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
from ultralytics import YOLO
import random

# Logging setup
logger = logging.getLogger("object_detection")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://influxdb:8086")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "my-token")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "my-org")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "object_detection")
DETECTION_INTERVAL = int(os.environ.get("DETECTION_INTERVAL", 2))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))
YOUTUBE_STREAMS = os.environ.get("YOUTUBE_STREAMS", "[]") # JSON array

def open_ytdlp_ffmpeg_pipe(youtube_url, width, height):
    # Start yt-dlp process to output 360p video+audio to stdout
    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "-o", "-", # output to stdout
        youtube_url
    ]
    ytdlp_proc = subprocess.Popen(
        ytdlp_cmd, stdout=subprocess.PIPE, bufsize=10**8
    )
    # Start FFmpeg process, reading from yt-dlp's stdout
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "-", # Read from stdin
        "-vf", f"scale={width}:{height}",  # Force scaling to desired size
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
        import os
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

def draw_bounding_boxes(image, boxes, names):
    """
    Draw colored bounding boxes and labels on the image.
    boxes: list of (box, cls, conf)
    """
    for box, cls, conf in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{names[int(box.cls[0])]} {conf:.2f}"
        color = random_color()
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Draw label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        # Draw label text
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return image

def process_stream(stream, write_api):
    url = stream['url']
    title = stream['title']
    slug = stream['slug']
    logger.info(f"Opening yt-dlp+FFmpeg pipeline for '{title}' ({slug})...")
    ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
    model = YOLO("yolov8n.pt")
    last_detection = time.time()
    frame_count = 0

    while True:
        # Check if processes are still running, if not, re-init
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
            results = model(frame)
            logger.info(f"[{slug}][Frame {frame_count}] Prediction results at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now))}:")
            detected_boxes = []
            for result in results:
                for box in result.boxes:
                    cls = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    logger.info(f" [{slug}] Detected: {cls} (confidence: {conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                    detected_boxes.append((box, result.names, conf))
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

            # Draw colored bounding boxes and save the image
            if detected_boxes:
                # Use result.names for class names
                annotated_frame = draw_bounding_boxes(frame.copy(), [(box, box.cls, conf) for box, _, conf in detected_boxes], result.names)
                os.makedirs("output_images", exist_ok=True)
                out_path = os.path.join("output_images", f"{slug}_frame_{frame_count:06d}.jpg")
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
