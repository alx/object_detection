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
from ultralytics import YOLO # Replaced darknetpy
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
SAVE_CROPPED_IMG = (os.getenv('SAVE_CROP_IMG', 'False') == 'True') # Note: Original had SAVE_CROP_IMG, docs suggest SAVE_CROPPED_IMG

# Path to YOLO model file for Ultralytics
ULTRALYTICS_MODEL_PATH = os.environ.get("ULTRALYTICS_MODEL_PATH", "yolov8n.pt") # Default to yolov8n.pt
try:
    detector = YOLO(ULTRALYTICS_MODEL_PATH)
    logger.info(f"Ultralytics YOLO model loaded from {ULTRALYTICS_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading Ultralytics YOLO model: {e}")
    sys.exit(1)


def open_ytdlp_ffmpeg_pipe(youtube_url, width, height):
    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "--quiet",
        "-o", "-", # output to stdout
        youtube_url
    ]
    ytdlp_proc = subprocess.Popen(
        ytdlp_cmd, stdout=subprocess.PIPE, bufsize=10**8
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "quiet",
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
            timeout=5000 # milliseconds
        )
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

def draw_bounding_boxes(image, detection_results, model_names):
    """
    Draw colored bounding boxes and labels on the image.
    detection_results: ultralytics results object (typically results[0] for single image)
    model_names: list of class names from detector.names
    """
    if detection_results is None or detection_results.boxes is None:
        return image

    for box in detection_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        try:
            label_name = model_names[cls_id]
        except IndexError:
            label_name = f"ClassID {cls_id}" # Fallback if class ID is out of bounds
            logger.warning(f"Class ID {cls_id} out of bounds for model_names. Max ID: {len(model_names)-1}")


        label = f"{label_name} {conf:.2f}"
        color = random_color()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Ensure text background doesn't go out of bounds
        text_bg_y1 = max(0, y1 - th - baseline)
        text_bg_y2 = y1
        cv2.rectangle(image, (x1, text_bg_y1), (x1 + tw, text_bg_y2), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return image

def iou(boxA, boxB):
    """
    Intersection over Union
    box = (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Added epsilon for stability
    return iou_val

def filter_duplicate_detections(detections_dicts, prev_detections_dicts, iou_threshold=0.5):
    """
    detections_dicts: list of dicts, each with keys 'class', 'left', 'top', 'right', 'bottom'
    prev_detections_dicts: list of dicts from previous detection cycle
    Returns: filtered list of detection dicts
    """
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
    """
    Crop the detected region from the frame and save it.
    """
    # Sanitize stream_slug and class_name for directory creation if necessary
    safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
    safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)

    out_dir = os.path.join("output", "fullframe", safe_stream_slug, safe_class_name)
    os.makedirs(out_dir, exist_ok=True)
    epoch_ts = int(time.time() * 1000) # Milliseconds for more uniqueness
    out_path = os.path.join(out_dir, f"{epoch_ts}.jpg")
    try:
        cv2.imwrite(out_path, frame)
        logger.info(f"Saved fullframe image: {out_path}")
    except Exception as e:
        logger.error(f"Error saving cropped image {out_path}: {e}")

def save_cropped_detection(frame, x1, y1, x2, y2, class_name, stream_slug):
    """
    Crop the detected region from the frame and save it.
    """
    if y1 >= y2 or x1 >= x2: # Check for invalid box dimensions
        logger.warning(f"Skipping save_cropped_detection due to invalid box dimensions: [{x1},{y1},{x2},{y2}] for {class_name}")
        return
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        logger.warning(f"Skipping save_cropped_detection due to empty crop for {class_name}")
        return
    # Sanitize stream_slug and class_name for directory creation if necessary
    safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
    safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)

    out_dir = os.path.join("output", "cropped", safe_stream_slug, safe_class_name)
    os.makedirs(out_dir, exist_ok=True)
    epoch_ts = int(time.time() * 1000) # Milliseconds for more uniqueness
    out_path = os.path.join(out_dir, f"{epoch_ts}.jpg")
    try:
        cv2.imwrite(out_path, cropped)
        logger.info(f"Saved cropped image: {out_path}")
    except Exception as e:
        logger.error(f"Error saving cropped image {out_path}: {e}")

def save_empty_detection_frame(frame, stream_slug, current_timestamp, last_saved_timestamp):
    """
    Saves the full frame to 'output/fullframe/empty_images/{stream_slug}/'
    if no objects are detected and at least an hour has passed since the last save.
    Returns the updated last_saved_timestamp.
    """
    time_since_last_save = current_timestamp - last_saved_timestamp
    if time_since_last_save >= 3600: # 3600 seconds = 1 hour
        safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
        out_dir = os.path.join("output", "fullframe", "empty_images", safe_stream_slug)
        os.makedirs(out_dir, exist_ok=True)
        epoch_ts_ms = int(current_timestamp * 1000) # Milliseconds for uniqueness
        out_path = os.path.join(out_dir, f"{epoch_ts_ms}.jpg")
        try:
            cv2.imwrite(out_path, frame)
            logger.info(f"[{stream_slug}] Saved empty detection frame: {out_path}")
            return current_timestamp # Update the last saved timestamp
        except Exception as e:
            logger.error(f"[{stream_slug}] Error saving empty detection frame {out_path}: {e}")
            return last_saved_timestamp # Don't update if save failed
    return last_saved_timestamp # No save needed, return original timestamp


def process_stream(stream_info, influx_write_api):
    url = stream_info['url']
    title = stream_info['title']
    slug = stream_info['slug']
    logger.info(f"Opening yt-dlp+FFmpeg pipeline for '{title}' ({slug})...")

    ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
    last_detection_time = time.time()
    frame_count = 0
    last_empty_saved_time = 0.0 # Timestamp of the last saved empty frame for this stream

    WINDOW_SIZE = 10 # Number of past detection sets to consider for duplicate filtering
    prev_detections_window_dicts = deque(maxlen=WINDOW_SIZE) # Stores lists of detection dicts

    while True:
        if ytdlp_proc.poll() is not None or ffmpeg_proc.poll() is not None:
            logger.warning(f"Process ended unexpectedly for '{title}' ({slug}), restarting pipeline...")
            try:
                if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate() # type: ignore
                if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate() # type: ignore
            except Exception as e:
                logger.error(f"Error terminating old processes: {e}")
            time.sleep(5) # Wait a bit before restarting
            ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
            last_empty_saved_time = 0.0 # Reset timer on restart
            continue # Restart the loop

        frame = read_frame(ffmpeg_proc, FRAME_WIDTH, FRAME_HEIGHT)
        if frame is None:
            logger.warning(f"Stream ended or cannot fetch frame for '{title}' ({slug}). Attempting restart.")
            time.sleep(5) # Wait before trying to reopen
            try:
                if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate() # type: ignore
                if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate() # type: ignore
            except Exception as e:
                 logger.error(f"Error terminating processes before restart: {e}")
            ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
            last_empty_saved_time = 0.0 # Reset timer on restart
            continue

        current_time = time.time()
        if current_time - last_detection_time >= DETECTION_INTERVAL:

            ultralytics_results = detector(frame, verbose=False)
            current_detections_dicts = []
            processed_for_influx = []

            if ultralytics_results and len(ultralytics_results) > 0:
                detection_result_item = ultralytics_results[0]
                if detection_result_item.boxes is not None:
                    for box in detection_result_item.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        try:
                            class_name = detector.names[cls_id]
                        except (IndexError, KeyError) as e:
                            logger.warning(f"[{slug}] Class ID {cls_id} not found in model names. Error: {e}")
                            class_name = f"UnknownClass_{cls_id}"

                        if CLASS_WHITELIST is not None and class_name not in CLASS_WHITELIST:
                            continue
                        det_dict = {
                            'class': class_name,
                            'prob': conf,
                            'left': x1, 'top': y1, 'right': x2, 'bottom': y2
                        }
                        current_detections_dicts.append(det_dict)
                        processed_for_influx.append(det_dict)

            all_prev_detections_dicts = [det for frame_dets_list in prev_detections_window_dicts for det in frame_dets_list]
            final_results_dicts = filter_duplicate_detections(current_detections_dicts, all_prev_detections_dicts, iou_threshold=0.4)

            if len(final_results_dicts) > 0:
                logger.info(f"[{slug}][Frame {frame_count}] Prediction results at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(current_time))}: {len(final_results_dicts)} objects after filtering.")
                for det_dict in final_results_dicts:
                    cls = det_dict['class']
                    conf = float(det_dict['prob'])
                    x1, y1, x2, y2 = det_dict['left'], det_dict['top'], det_dict['right'], det_dict['bottom']
                    logger.info(f" [{slug}] Detected: {cls} (confidence: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

                    save_fullframe_detection(frame, cls, slug)
                    # if SAVE_CROPPED_IMG: # Check environment variable before saving cropped image
                    #     save_cropped_detection(frame, x1, y1, x2, y2, cls, slug)

                    if influx_write_api:
                        pt = (
                            Point("object_detections")
                            .tag("object", cls)
                            .tag("stream_slug", slug)
                            .tag("stream_title", title)
                            .tag("stream_url", url)
                            .field("confidence", conf)
                            .field("x1", int(x1))
                            .field("y1", int(y1))
                            .field("x2", int(x2))
                            .field("y2", int(y2))
                            .time(int(current_time * 1e9), WritePrecision.NS)
                        )
                        try:
                            influx_write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=pt)
                        except Exception as e:
                            logger.warning(f" [{slug}] Could not write to InfluxDB: {e}")
                    else:
                        logger.warning(f" [{slug}] InfluxDB write_api not available, skipping DB write.")
            else: # No detections found in final_results_dicts
                logger.info(f"[{slug}][Frame {frame_count}] No objects detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(current_time))}.")
                # Save full frame if no detections and an hour has passed
                last_empty_saved_time = save_empty_detection_frame(frame, slug, current_time, last_empty_saved_time)


            if final_results_dicts and DEBUG_IMG_BBOX and ultralytics_results and len(ultralytics_results) > 0:
                annotated_frame = draw_bounding_boxes(frame.copy(), ultralytics_results[0], detector.names)
                os.makedirs("output_images", exist_ok=True)
                epoch_ts = int(current_time)
                out_path = os.path.join("output_images", f"{slug}_frame_{frame_count:06d}_{epoch_ts}.jpg")
                cv2.imwrite(out_path, annotated_frame)

            last_detection_time = current_time
            prev_detections_window_dicts.append(current_detections_dicts)

        frame_count += 1
        if DETECTION_INTERVAL > 0.1 : time.sleep(0.01)

    logger.info(f"Exiting process_stream for '{title}' ({slug}).")
    if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate() # type: ignore
    if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate() # type: ignore

def main():
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
        thread = threading.Thread(target=process_stream, args=(stream_config, influx_write_api), daemon=True)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    if influx_client:
        influx_client.close()
    logger.info("All stream processing finished. Exiting main application.")

if __name__ == "__main__":
    main()
