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
SAVE_CROPPED_IMG = (os.getenv('SAVE_CROPPED_IMG', 'False') == 'True') # Corrected from SAVE_CROP_IMG

# Paths for Darknetpy model
DARKNET_DATA_PATH = os.environ.get("DARKNET_DATA_PATH")
DARKNET_CFG_PATH = os.environ.get("DARKNET_CFG_PATH")
DARKNET_WEIGHTS_PATH = os.environ.get("DARKNET_WEIGHTS_PATH")

detector_instance = None

if not all([DARKNET_DATA_PATH, DARKNET_CFG_PATH, DARKNET_WEIGHTS_PATH]):
    logger.error("Missing Darknet paths: Ensure DARKNET_DATA_PATH, DARKNET_CFG_PATH, and DARKNET_WEIGHTS_PATH environment variables are set.")
    sys.exit(1)
else:
    try:
        # Ensure the temp_frames directory exists for darknetpy processing
        os.makedirs("temp_frames", exist_ok=True)
        logger.info("Created 'temp_frames' directory for temporary image files.")

        detector_instance = Detector(DARKNET_DATA_PATH, DARKNET_CFG_PATH, DARKNET_WEIGHTS_PATH)
        logger.info(f"Darknetpy YOLO model loaded using:\n  Data: {DARKNET_DATA_PATH}\n  CFG: {DARKNET_CFG_PATH}\n  Weights: {DARKNET_WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Error loading Darknetpy YOLO model: {e}")
        logger.error("Please ensure darknetpy is installed correctly and model paths are valid (absolute paths might be required by darknetpy).")
        sys.exit(1)

def open_ytdlp_ffmpeg_pipe(youtube_url, width, height):
    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "--quiet",
        "-o", "-",
        youtube_url
    ]
    ytdlp_proc = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-i", "-",
        "-vf", f"scale={width}:{height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=ytdlp_proc.stdout, stdout=subprocess.PIPE, bufsize=10**8)
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
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=5000)
        health = client.health()
        if health.status != "pass": # type: ignore
            logger.error(f"InfluxDB health check failed: {health.message}") # type: ignore
            return None, None
        write_api = client.write_api(write_options=SYNCHRONOUS)
        logger.info("Connected to InfluxDB.")
        return client, write_api
    except Exception as e:
        logger.error(f"Could not connect to InfluxDB: {e}")
        return None, None

def random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

def draw_bounding_boxes(image, detection_dicts):
    """
    Draw colored bounding boxes and labels on the image.
    detection_dicts: list of dicts, each {'class': name, 'prob': confidence, 'left': x1, 'top': y1, 'right': x2, 'bottom': y2}
    """
    if not detection_dicts:
        return image

    for det in detection_dicts:
        x1 = det['left']
        y1 = det['top']
        x2 = det['right']
        y2 = det['bottom']
        conf = det['prob']
        label_name = det['class']

        label = f"{label_name} {conf:.2f}"
        color = random_color()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_bg_y1 = max(0, y1 - th - baseline)
        text_bg_y2 = y1
        cv2.rectangle(image, (x1, text_bg_y1), (x1 + tw, text_bg_y2), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return image

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_val

def filter_duplicate_detections(detections_dicts, prev_detections_dicts, iou_threshold=0.5):
    filtered = []
    for det in detections_dicts:
        boxA = (det['left'], det['top'], det['right'], det['bottom'])
        duplicate = False
        for prev in prev_detections_dicts:
            if det['class'] == prev['class']:
                boxB = (prev['left'], prev['top'], prev['right'], prev['bottom'])
                if iou(boxA, boxB) > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            filtered.append(det)
    return filtered

def save_fullframe_detection(frame, class_name, stream_slug):
    safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
    safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
    out_dir = os.path.join("output", "fullframe", safe_stream_slug, safe_class_name)
    os.makedirs(out_dir, exist_ok=True)
    epoch_ts = int(time.time() * 1000)
    out_path = os.path.join(out_dir, f"{epoch_ts}.jpg")
    try:
        cv2.imwrite(out_path, frame)
        logger.info(f"Saved fullframe image: {out_path}")
    except Exception as e:
        logger.error(f"Error saving fullframe image {out_path}: {e}")

def save_cropped_detection(frame, x1, y1, x2, y2, class_name, stream_slug):
    if y1 >= y2 or x1 >= x2:
        logger.warning(f"Skipping save_cropped_detection due to invalid box dimensions: [{x1},{y1},{x2},{y2}] for {class_name}")
        return
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        logger.warning(f"Skipping save_cropped_detection due to empty crop for {class_name}")
        return
    safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
    safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
    out_dir = os.path.join("output", "cropped", safe_stream_slug, safe_class_name)
    os.makedirs(out_dir, exist_ok=True)
    epoch_ts = int(time.time() * 1000)
    out_path = os.path.join(out_dir, f"{epoch_ts}.jpg")
    try:
        cv2.imwrite(out_path, cropped)
        logger.info(f"Saved cropped image: {out_path}")
    except Exception as e:
        logger.error(f"Error saving cropped image {out_path}: {e}")

def save_empty_detection_frame(frame, stream_slug, current_timestamp, last_saved_timestamp):
    time_since_last_save = current_timestamp - last_saved_timestamp
    if time_since_last_save >= 3600:
        safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
        out_dir = os.path.join("output", "fullframe", "empty_images", safe_stream_slug)
        os.makedirs(out_dir, exist_ok=True)
        epoch_ts_ms = int(current_timestamp * 1000)
        out_path = os.path.join(out_dir, f"{epoch_ts_ms}.jpg")
        try:
            cv2.imwrite(out_path, frame)
            logger.info(f"[{stream_slug}] Saved empty detection frame: {out_path}")
            return current_timestamp
        except Exception as e:
            logger.error(f"[{stream_slug}] Error saving empty detection frame {out_path}: {e}")
            return last_saved_timestamp
    return last_saved_timestamp

def process_stream(stream_info, influx_write_api, local_detector):
    if local_detector is None:
        logger.error(f"Detector not initialized for stream '{stream_info.get('slug', 'unknown_stream')}'. Skipping.")
        return

    url = stream_info['url']
    title = stream_info['title']
    slug = stream_info['slug']
    logger.info(f"Opening yt-dlp+FFmpeg pipeline for '{title}' ({slug})...")

    ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
    last_detection_time = time.time()
    frame_count = 0
    last_empty_saved_time = 0.0

    WINDOW_SIZE = 10
    prev_detections_window_dicts = deque(maxlen=WINDOW_SIZE)

    # Path for temporary frame file for darknetpy
    temp_frame_base_dir = "temp_frames"
    os.makedirs(temp_frame_base_dir, exist_ok=True) # Ensure base temp dir exists
    temp_frame_path = os.path.join(temp_frame_base_dir, f"{slug}_current_frame.jpg")


    while True:
        if ytdlp_proc.poll() is not None or ffmpeg_proc.poll() is not None:
            logger.warning(f"Process ended unexpectedly for '{title}' ({slug}), restarting pipeline...")
            try:
                if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate()
                if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate()
            except Exception as e:
                logger.error(f"Error terminating old processes: {e}")
            time.sleep(5)
            ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
            last_empty_saved_time = 0.0
            continue

        frame = read_frame(ffmpeg_proc, FRAME_WIDTH, FRAME_HEIGHT)
        if frame is None:
            logger.warning(f"Stream ended or cannot fetch frame for '{title}' ({slug}). Attempting restart.")
            time.sleep(5)
            try:
                if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate()
                if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate()
            except Exception as e:
                 logger.error(f"Error terminating processes before restart: {e}")
            ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
            last_empty_saved_time = 0.0
            continue

        current_time = time.time()
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            darknet_raw_results = []
            try:
                # Save frame to temp file for darknetpy
                cv2.imwrite(temp_frame_path, frame)
                # Perform detection using Darknetpy
                darknet_raw_results = local_detector.detect(temp_frame_path)
            except Exception as e:
                logger.error(f"[{slug}] Error during darknetpy detection: {e}")
                # Optional: clean up temp file if it exists and an error occurred
                # if os.path.exists(temp_frame_path):
                #    os.remove(temp_frame_path)

            current_detections_dicts = []
            if darknet_raw_results:
                for res in darknet_raw_results:
                    class_name = res.get('class')
                    conf = float(res.get('prob', 0.0))
                    x1 = int(res.get('left', 0))
                    y1 = int(res.get('top', 0))
                    x2 = int(res.get('right', 0))
                    y2 = int(res.get('bottom', 0))

                    if not class_name: # Skip if class name is missing
                        continue

                    if CLASS_WHITELIST is not None and class_name not in CLASS_WHITELIST:
                        continue

                    det_dict = {
                        'class': class_name, 'prob': conf,
                        'left': x1, 'top': y1, 'right': x2, 'bottom': y2
                    }
                    current_detections_dicts.append(det_dict)

            all_prev_detections_dicts = [det for frame_dets_list in prev_detections_window_dicts for det in frame_dets_list]
            final_results_dicts = filter_duplicate_detections(current_detections_dicts, all_prev_detections_dicts, iou_threshold=0.4)

            if len(final_results_dicts) > 0:
                logger.info(f"[{slug}][Frame {frame_count}] Prediction results at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(current_time))}: {len(final_results_dicts)} objects after filtering.")
                for det_dict in final_results_dicts:
                    cls = det_dict['class']
                    conf = det_dict['prob']
                    x1, y1, x2, y2 = det_dict['left'], det_dict['top'], det_dict['right'], det_dict['bottom']
                    logger.info(f" [{slug}] Detected: {cls} (confidence: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

                    save_fullframe_detection(frame, cls, slug)
                    if SAVE_CROPPED_IMG:
                        save_cropped_detection(frame, x1, y1, x2, y2, cls, slug)

                    if influx_write_api:
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
                            .time(int(current_time * 1e9), WritePrecision.NS)
                        )
                        try:
                            influx_write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=pt)
                        except Exception as e:
                            logger.warning(f" [{slug}] Could not write to InfluxDB: {e}")
                    else:
                        logger.warning(f" [{slug}] InfluxDB write_api not available, skipping DB write.")
            else:
                logger.info(f"[{slug}][Frame {frame_count}] No objects detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(current_time))}.")
                last_empty_saved_time = save_empty_detection_frame(frame, slug, current_time, last_empty_saved_time)

            if final_results_dicts and DEBUG_IMG_BBOX:
                annotated_frame = draw_bounding_boxes(frame.copy(), final_results_dicts)
                os.makedirs("output_images", exist_ok=True)
                epoch_ts = int(current_time)
                out_path = os.path.join("output_images", f"{slug}_frame_{frame_count:06d}_{epoch_ts}.jpg")
                cv2.imwrite(out_path, annotated_frame)

            last_detection_time = current_time
            prev_detections_window_dicts.append(current_detections_dicts)

        frame_count += 1
        if DETECTION_INTERVAL > 0.1 : time.sleep(0.01)

    # Clean up temp file for this stream when exiting (though loop is infinite)
    if os.path.exists(temp_frame_path):
        try:
            os.remove(temp_frame_path)
        except OSError as e:
            logger.warning(f"Error removing temp file {temp_frame_path}: {e}")

    logger.info(f"Exiting process_stream for '{title}' ({slug}).")
    if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate()
    if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate()

def main():
    global detector_instance # Make sure main uses the globally initialized detector

    if detector_instance is None:
        logger.error("Darknetpy detector was not successfully initialized. Exiting.")
        sys.exit(1)

    try:
        streams_data = json.loads(YOUTUBE_STREAMS)
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse YOUTUBE_STREAMS JSON: {e}. Value was: {YOUTUBE_STREAMS}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing YOUTUBE_STREAMS: {e}")
        sys.exit(1)

    if not streams_data:
        logger.error("No streams specified in YOUTUBE_STREAMS. Please provide a valid JSON list of streams.")
        sys.exit(1)

    influx_client, influx_write_api = connect_influxdb()

    threads = []
    for stream_config in streams_data:
        if not all(k in stream_config for k in ("url", "title", "slug")):
            logger.warning(f"Skipping stream due to missing 'url', 'title', or 'slug': {stream_config}")
            continue
        thread = threading.Thread(target=process_stream, args=(stream_config, influx_write_api, detector_instance), daemon=True)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    if influx_client:
        influx_client.close()
    logger.info("All stream processing finished. Exiting main application.")

if __name__ == "__main__":
    main()
