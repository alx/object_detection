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
YOUTUBE_STREAMS = os.environ.get("YOUTUBE_STREAMS", "[]") # JSON array

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

def process_stream(stream, write_api):
    url = stream['url']
    title = stream['title']
    slug = stream['slug']
    logger.info(f"Opening yt-dlp+FFmpeg pipeline for '{title}' ({slug})...")

    ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
    last_detection = time.time()
    frame_count = 0

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
            logger.info(f"[{slug}][Frame {frame_count}] Prediction results at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now))}:")
            for obj in results:
                cls = obj['class']
                conf = float(obj['prob'])
                x1, y1, x2, y2 = obj['left'], obj['top'], obj['right'], obj['bottom']
                logger.info(f" [{slug}] Detected: {cls} (confidence: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
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
            if results and DEBUG_IMG_BBOX:
                annotated_frame = draw_bounding_boxes(frame.copy(), results)
                os.makedirs("output_images", exist_ok=True)
                epoch_ts = int(now)
                out_path = os.path.join("output_images", f"{slug}_frame_{frame_count:06d}_{epoch_ts}.jpg")
                cv2.imwrite(out_path, annotated_frame)

            last_detection = now
        frame_count += 1

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
