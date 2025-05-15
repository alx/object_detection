import os
import sys
import time
import subprocess
import numpy as np
import cv2
import logging
import json
import threading
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from darknetpy.detector import Detector
import random
from collections import deque

# Logging setup
logger = logging.getLogger("object_detection")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

DEBUG_IMG_BBOX = (os.getenv('DEBUG_IMG_BBOX', 'False') == 'True')
INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://influxdb:8086")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "my-token")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "my-org")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "object_detection")
DETECTION_INTERVAL = int(os.environ.get("DETECTION_INTERVAL", 2))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))
YOUTUBE_STREAMS = os.environ.get("YOUTUBE_STREAMS", "[]")
CLASS_WHITELIST = set(map(str.strip, os.environ.get("CLASS_WHITELIST", "").split(","))) if os.environ.get("CLASS_WHITELIST") else None
SAVE_CROPPED_IMG = (os.getenv('SAVE_CROP_IMG', 'False') == 'True')

# Paths to YOLO model files (adjust as needed)
DARKNET_DATA = os.environ.get("DARKNET_DATA", "/models/coco.data")
DARKNET_CFG = os.environ.get("DARKNET_CFG", "/models/yolov4.cfg")
DARKNET_WEIGHTS = os.environ.get("DARKNET_WEIGHTS", "/models/yolov4.weights")
detector = Detector(DARKNET_DATA, DARKNET_CFG, DARKNET_WEIGHTS)

def open_ytdlp_ffmpeg_pipe(youtube_url, width, height):
    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "-o", "-", # output to stdout
        youtube_url
    ]
    ytdlp_proc = subprocess.Popen(
        ytdlp_cmd, stdout=subprocess.PIPE, bufsize=10**8
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "-", # Read from stdin
        "-vf", f"scale={width}:{height}", # Force scaling to desired size
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=ytdlp_proc.stdout, stdout=subprocess.PIPE, bufsize=10**8
    )
    return ytdlp_proc, ffmpeg_proc

def read_frame(ffmpeg_proc, width, height, save_dir=None, frame_idx=None):
    frame_size = width * height * 3
    raw_frame = ffmpeg_proc.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        return None
    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
    if save_dir is not None and frame_idx is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(filename, frame)
    return frame

def connect_influxdb():
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
            timeout=5000
        )
        health = client.health()
        if health.status != "pass":
            logger.error(f"InfluxDB health check failed: {health.message}")
            return None, None
        write_api = client.write_api(write_options=SYNCHRONOUS)
        logger.info("Connected to InfluxDB.")
        return client, write_api
    except Exception as e:
        logger.error(f"Could not connect to InfluxDB: {e}")
        return None, None

def random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

def draw_bounding_boxes(image, results):
    """
    Draw colored bounding boxes and labels on the image.
    results: list of dicts from darknetpy
    """
    for obj in results:
        left, top, right, bottom = obj['left'], obj['top'], obj['right'], obj['bottom']
        label = f"{obj['class']} {obj['prob']:.2f}"
        color = random_color()
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (left, top - th - baseline), (left + tw, top), color, -1)
        cv2.putText(image, label, (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return image

def iou(boxA, boxB):
    """
    Intersection over Union
    spatial proximity check based on bounding box overlap

    box = (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_val

def filter_duplicate_detections(detections, prev_detections, iou_threshold=0.5):
    """
    detections: list of dicts, each with keys 'class', 'left', 'top', 'right', 'bottom'
    prev_detections: list of dicts from previous detection cycle
    Returns: filtered list of detections
    """
    filtered = []
    for det in detections:
        boxA = (det['left'], det['top'], det['right'], det['bottom'])
        duplicate = False
        for prev in prev_detections:
            if det['class'] == prev['class']:
                boxB = (prev['left'], prev['top'], prev['right'], prev['bottom'])
                if iou(boxA, boxB) > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            filtered.append(det)
    return filtered

def save_cropped_detection(frame, x1, y1, x2, y2, class_name, stream__url):
    """
    Crop the detected region from the frame and save it to
    output/stream_slug/class_name/DDMMYYYY/epoch_timestamp.jpg
    """
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return
    date_str = time.strftime("%d%m%Y")
    out_dir = os.path.join("output", stream_slug, class_name, date_str)
    os.makedirs(out_dir, exist_ok=True)
    epoch_ts = int(time.time())
    out_path = os.path.join(out_dir, f"{epoch_ts}.jpg")
    cv2.imwrite(out_path, cropped)

def process_stream(stream, write_api):
    url = stream['url']
    title = stream['title']
    slug = stream['slug']
    logger.info(f"Opening yt-dlp+FFmpeg pipeline for '{title}' ({slug})...")

    ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
    last_detection = time.time()
    frame_count = 0

    WINDOW_SIZE = 10
    prev_detections_window = deque(maxlen=WINDOW_SIZE)

    while True:
        if ytdlp_proc.poll() is not None or ffmpeg_proc.poll() is not None:
            logger.warning(f"Process ended unexpectedly for '{title}' ({slug}), restarting pipeline...")
            try:
                ytdlp_proc.terminate()
            except Exception:
                pass
            try:
                ffmpeg_proc.terminate()
            except Exception:
                pass
            ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)

        frame = read_frame(ffmpeg_proc, FRAME_WIDTH, FRAME_HEIGHT)
        if frame is None:
            logger.warning(f"Stream ended or cannot fetch frame for '{title}' ({slug}).")
            break

        now = time.time()
        if now - last_detection >= DETECTION_INTERVAL:

            results = detector.detect(frame)

            # Filter based on classnames
            if CLASS_WHITELIST is not None:
                results = [obj for obj in results if obj['class'] in CLASS_WHITELIST]

            # Filter out duplicates by position/class
            all_prev_detections = [det for frame_dets in prev_detections_window for det in frame_dets]
            filtered_results = filter_duplicate_detections(results, all_prev_detections, iou_threshold=0.5)

            logger.info(f"[{slug}][Frame {frame_count}] Prediction results at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now))}:")
            for obj in filtered_results:
                cls = obj['class']
                conf = float(obj['prob'])
                x1, y1, x2, y2 = obj['left'], obj['top'], obj['right'], obj['bottom']
                logger.info(f" [{slug}] Detected: {cls} (confidence: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

                # Save cropped detection
                if SAVE_CROPPED_IMG:
                    save_cropped_detection(frame, x1, y1, x2, y2, cls, slug)

                if write_api:
                    pt = (
                        Point("object_detections")
                        .tag("object", cls)
                        .tag("stream_slug", slug)
                        .tag("stream_title", title)
                        .tag("stream_url", url)
                        .field("confidence", conf)
                        .field("x1", x1)
                        .field("y1", y1)
                        .field("x2", x2)
                        .field("y2", y2)
                        .time(int(now * 1e9), WritePrecision.NS)
                    )
                    try:
                        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=pt)
                    except Exception as e:
                        logger.warning(f" [{slug}] Could not write to InfluxDB: {e}")
                else:
                    logger.warning(f" [{slug}] InfluxDB write_api not available, skipping DB write.")

            # Draw colored bounding boxes and save the image if debug mode
            if filtered_results and DEBUG_IMG_BBOX:
                annotated_frame = draw_bounding_boxes(frame.copy(), results)
                os.makedirs("output_images", exist_ok=True)
                epoch_ts = int(now)
                out_path = os.path.join("output_images", f"{slug}_frame_{frame_count:06d}_{epoch_ts}.jpg")
                cv2.imwrite(out_path, annotated_frame)

            last_detection = now
            frame_count += 1
            prev_detections_window.append(results)

    ffmpeg_proc.terminate()
    ytdlp_proc.terminate()

def main():
    try:
        streams = json.loads(YOUTUBE_STREAMS)
    except Exception as e:
        logger.error(f"Could not parse YOUTUBE_STREAMS: {e}")
        sys.exit(1)
    if not streams:
        logger.error("No streams specified in YOUTUBE_STREAMS.")
        sys.exit(1)
    client, write_api = connect_influxdb()
    threads = []
    for stream in streams:
        t = threading.Thread(target=process_stream, args=(stream, write_api))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if client:
        client.close()

if __name__ == "__main__":
    main()
