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
import darknet # From hank-ai/darknet
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

# --- Environment Variables ---
DEBUG_IMG_BBOX = (os.getenv('DEBUG_IMG_BBOX', 'False') == 'True')
INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://influxdb:8086")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "my-token")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "my-org")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "object_detection")
DETECTION_INTERVAL = int(os.environ.get("DETECTION_INTERVAL", 2))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280)) # For yt-dlp output, actual processing uses net dims
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))# For yt-dlp output, actual processing uses net dims
YOUTUBE_STREAMS = os.environ.get("YOUTUBE_STREAMS", "[]")
CLASS_WHITELIST = set(map(str.strip, os.environ.get("CLASS_WHITELIST", "").split(","))) if os.environ.get("CLASS_WHITELIST") else None
SAVE_CROPPED_IMG = (os.getenv('SAVE_CROPPED_IMG', 'False') == 'True')

# Paths for Darknet model (from hank-ai/darknet)
DARKNET_DATA_PATH = os.environ.get("DARKNET_DATA_PATH")
DARKNET_CFG_PATH = os.environ.get("DARKNET_CFG_PATH")
DARKNET_WEIGHTS_PATH = os.environ.get("DARKNET_WEIGHTS_PATH")
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", 0.25)) # Detection confidence threshold

# --- Global Darknet Model Variables ---
network = None
class_names = None
class_colors = None # Will be populated by darknet.load_network

def load_darknet_model():
    """Loads the Darknet model using paths from environment variables."""
    global network, class_names, class_colors
    if not all([DARKNET_DATA_PATH, DARKNET_CFG_PATH, DARKNET_WEIGHTS_PATH]):
        logger.error("Darknet model paths not fully set. Ensure DARKNET_DATA_PATH, DARKNET_CFG_PATH, DARKNET_WEIGHTS_PATH are defined.")
        return False
    try:
        # darknet.load_network expects string paths and handles encoding internally
        network, class_names, class_colors = darknet.load_network(
            DARKNET_CFG_PATH,
            DARKNET_DATA_PATH,
            DARKNET_WEIGHTS_PATH,
            batch_size=1  # Process one image at a time
        )
        logger.info(f"Darknet model loaded successfully:\n  CFG: {DARKNET_CFG_PATH}\n  Data: {DARKNET_DATA_PATH}\n  Weights: {DARKNET_WEIGHTS_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error loading Darknet model: {e}")
        logger.error("Ensure Darknet is compiled correctly (libdarknet.so available) and paths are valid.")
        return False

def open_ytdlp_ffmpeg_pipe(youtube_url, width, height):
    """Opens yt-dlp and ffmpeg processes to stream video frames."""
    ytdlp_cmd = [
        "yt-dlp", "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "--quiet", "-o", "-", youtube_url
    ]
    ytdlp_proc = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    ffmpeg_cmd = [
        "ffmpeg", "-loglevel", "quiet", "-i", "-",
        "-vf", f"scale={width}:{height}", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-"
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=ytdlp_proc.stdout, stdout=subprocess.PIPE, bufsize=10**8)
    return ytdlp_proc, ffmpeg_proc

def read_frame(ffmpeg_proc, width, height):
    """Reads a frame from the ffmpeg pipe."""
    frame_size = width * height * 3
    raw_frame = ffmpeg_proc.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        return None
    return np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

def connect_influxdb():
    """Connects to InfluxDB."""
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

def draw_boxes_on_frame(image_cv, detections_list, colors_dict):
    """
    Draws bounding boxes from darknet.detect_image output onto an OpenCV image.
    Args:
        image_cv: OpenCV image (BGR format).
        detections_list: List of tuples (label, confidence, (x_center, y_center, width, height)).
                         Coordinates are relative to the image dimensions passed for detection.
        colors_dict: Dictionary mapping class labels to (R, G, B) color tuples.
    """
    for label, confidence, bbox in detections_list:
        x_center, y_center, width, height = bbox
        x1 = int(round(x_center - (width / 2)))
        y1 = int(round(y_center - (height / 2)))
        x2 = int(round(x_center + (width / 2)))
        y2 = int(round(y_center + (height / 2)))

        # Ensure coordinates are within frame dimensions
        img_h, img_w, _ = image_cv.shape
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if x1 >= x2 or y1 >= y2: continue # Skip invalid boxes

        color_bgr = tuple(reversed(colors_dict.get(label, (random.randint(0,255), random.randint(0,255), random.randint(0,255))))) # Darknet colors are RGB

        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color_bgr, 2)
        label_text = f"{label} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image_cv, (x1, y1 - th - baseline), (x1 + tw, y1 - baseline), color_bgr, -1)
        cv2.putText(image_cv, label_text, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image_cv


def iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
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
    """Filters duplicate detections based on IoU with previous detections."""
    filtered = []
    for det in detections_dicts:
        boxA = (det['left'], det['top'], det['right'], det['bottom'])
        is_duplicate = False
        for prev_det_list in prev_detections_dicts: # prev_detections_dicts is a deque of lists
            for prev in prev_det_list:
                if det['class'] == prev['class']:
                    boxB = (prev['left'], prev['top'], prev['right'], prev['bottom'])
                    if iou(boxA, boxB) > iou_threshold:
                        is_duplicate = True
                        break
            if is_duplicate:
                break
        if not is_duplicate:
            filtered.append(det)
    return filtered

def save_image_with_subdir(base_dir, stream_slug, class_name, image_data, filename_prefix=""):
    """Saves an image, creating subdirectories as needed."""
    safe_stream_slug = "".join(c if c.isalnum() else "_" for c in stream_slug)
    safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name) if class_name else "unknown"

    out_dir = os.path.join("output", base_dir, safe_stream_slug, safe_class_name)
    if class_name is None: # For empty_images, class_name is not applicable
        out_dir = os.path.join("output", base_dir, "empty_images", safe_stream_slug)

    os.makedirs(out_dir, exist_ok=True)
    epoch_ts_ms = int(time.time() * 1000)
    out_path = os.path.join(out_dir, f"{filename_prefix}{epoch_ts_ms}.jpg")
    try:
        cv2.imwrite(out_path, image_data)
        logger.info(f"Saved image: {out_path}")
    except Exception as e:
        logger.error(f"Error saving image {out_path}: {e}")

def save_fullframe_detection(frame, class_name, stream_slug):
    save_image_with_subdir("fullframe", stream_slug, class_name, frame)

def save_cropped_detection(frame, x1, y1, x2, y2, class_name, stream_slug):
    if y1 >= y2 or x1 >= x2:
        logger.warning(f"Skipping save_cropped_detection: invalid box [{x1},{y1},{x2},{y2}] for {class_name}")
        return
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        logger.warning(f"Skipping save_cropped_detection: empty crop for {class_name}")
        return
    save_image_with_subdir("cropped", stream_slug, class_name, cropped)

def save_empty_detection_frame(frame, stream_slug, current_timestamp, last_saved_timestamp):
    if current_timestamp - last_saved_timestamp >= 3600: # 1 hour
        save_image_with_subdir("fullframe", stream_slug, None, frame, filename_prefix="empty_") # class_name is None
        return current_timestamp
    return last_saved_timestamp

def process_stream(stream_info, influx_write_api):
    """Processes a single video stream for object detection."""
    global network, class_names, class_colors # Use global model variables

    if network is None or class_names is None:
        logger.error(f"Darknet model not loaded. Skipping stream '{stream_info.get('slug', 'unknown_stream')}'.")
        return

    url = stream_info['url']
    title = stream_info['title']
    slug = stream_info['slug']
    logger.info(f"Opening yt-dlp+FFmpeg pipeline for '{title}' ({slug})...")

    ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
    last_detection_time = time.time()
    frame_count = 0
    last_empty_saved_time = 0.0
    prev_detections_window_dicts = deque(maxlen=10) # Window for duplicate filtering

    # Get network input dimensions once
    net_width = darknet.network_width(network)
    net_height = darknet.network_height(network)

    while True:
        original_frame = read_frame(ffmpeg_proc, FRAME_WIDTH, FRAME_HEIGHT)
        if original_frame is None:
            logger.warning(f"Stream ended or cannot fetch frame for '{title}' ({slug}). Attempting restart.")
            # Cleanup and restart logic
            try:
                if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate()
                if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate()
            except Exception as e: logger.error(f"Error terminating old processes: {e}")
            time.sleep(5)
            ytdlp_proc, ffmpeg_proc = open_ytdlp_ffmpeg_pipe(url, FRAME_WIDTH, FRAME_HEIGHT)
            last_empty_saved_time = 0.0 # Reset timer
            prev_detections_window_dicts.clear()
            continue

        current_time = time.time()
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            # Prepare frame for Darknet
            frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

            darknet_image = darknet.make_image(net_width, net_height, 3) # 3 channels for RGB
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            # Perform detection
            raw_detections_net_coords = darknet.detect_image(network, class_names, darknet_image, thresh=DETECTION_THRESHOLD)
            darknet.free_image(darknet_image) # Crucial to free Darknet image memory

            current_detections_dicts_orig_coords = []
            orig_h, orig_w, _ = original_frame.shape

            for label, confidence_float, bbox_net in raw_detections_net_coords:
                x_center_net, y_center_net, w_net, h_net = bbox_net # Coords relative to net_width, net_height

                # Scale bbox coordinates from network input size back to original frame dimensions
                x1 = int(round((x_center_net - w_net / 2.0) * (orig_w / net_width)))
                y1 = int(round((y_center_net - h_net / 2.0) * (orig_h / net_height)))
                x2 = int(round((x_center_net + w_net / 2.0) * (orig_w / net_width)))
                y2 = int(round((y_center_net + h_net / 2.0) * (orig_h / net_height)))

                # Clamp coordinates to original frame dimensions
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                if CLASS_WHITELIST is not None and label not in CLASS_WHITELIST:
                    continue

                if x1 >= x2 or y1 >= y2: # Skip invalid boxes after scaling
                    continue

                det_dict = {
                    'class': label, 'prob': float(confidence_float),
                    'left': x1, 'top': y1, 'right': x2, 'bottom': y2
                }
                current_detections_dicts_orig_coords.append(det_dict)

            # Filter duplicates based on IoU with detections in the sliding window
            final_results_dicts = filter_duplicate_detections(current_detections_dicts_orig_coords, prev_detections_window_dicts)
            prev_detections_window_dicts.append(current_detections_dicts_orig_coords) # Add current raw (unfiltered by IoU) detections to window

            if len(final_results_dicts) > 0:
                logger.info(f"[{slug}][Frame {frame_count}] Detections: {len(final_results_dicts)} (after filtering)")
                for det_dict in final_results_dicts:
                    cls, conf, x1, y1, x2, y2 = det_dict['class'], det_dict['prob'], det_dict['left'], det_dict['top'], det_dict['right'], det_dict['bottom']
                    logger.info(f" [{slug}] Detected: {cls} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")
                    save_fullframe_detection(original_frame, cls, slug)
                    if SAVE_CROPPED_IMG:
                        save_cropped_detection(original_frame, x1, y1, x2, y2, cls, slug)
                    if influx_write_api:
                        pt = (Point("object_detections").tag("object", cls).tag("stream_slug", slug)
                              .tag("stream_title", title).tag("stream_url", url)
                              .field("confidence", conf).field("x1", x1).field("y1", y1)
                              .field("x2", x2).field("y2", y2)
                              .time(int(current_time * 1e9), WritePrecision.NS))
                        try:
                            influx_write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=pt)
                        except Exception as e:
                            logger.warning(f" [{slug}] InfluxDB write error: {e}")
            else:
                logger.info(f"[{slug}][Frame {frame_count}] No new objects detected.")
                last_empty_saved_time = save_empty_detection_frame(original_frame, slug, current_time, last_empty_saved_time)

            if DEBUG_IMG_BBOX and len(raw_detections_net_coords) > 0: # Draw based on raw detections on resized frame
                # For debug display, draw on a copy of the frame_resized (network input)
                # The raw_detections_net_coords are for this resized frame.
                debug_frame_to_draw = frame_resized.copy() # This is RGB
                debug_frame_to_draw_bgr = cv2.cvtColor(debug_frame_to_draw, cv2.COLOR_RGB2BGR)

                annotated_debug_frame = draw_boxes_on_frame(debug_frame_to_draw_bgr, raw_detections_net_coords, class_colors)

                os.makedirs("output_images", exist_ok=True)
                epoch_ts = int(current_time)
                out_path = os.path.join("output_images", f"{slug}_debug_net_input_{frame_count:06d}_{epoch_ts}.jpg")
                cv2.imwrite(out_path, annotated_debug_frame)

            last_detection_time = current_time
        frame_count += 1
        if DETECTION_INTERVAL > 0.1: time.sleep(0.01)

    logger.info(f"Exiting process_stream for '{title}' ({slug}).")
    if ffmpeg_proc and ffmpeg_proc.poll() is None: ffmpeg_proc.terminate()
    if ytdlp_proc and ytdlp_proc.poll() is None: ytdlp_proc.terminate()

def main():
    """Main function to initialize model and start stream processing threads."""
    if not load_darknet_model():
        logger.error("Failed to load Darknet model. Exiting application.")
        sys.exit(1)

    try:
        streams_data = json.loads(YOUTUBE_STREAMS)
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse YOUTUBE_STREAMS JSON: {e}. Value: {YOUTUBE_STREAMS}")
        sys.exit(1)

    if not streams_data:
        logger.error("No streams specified in YOUTUBE_STREAMS.")
        sys.exit(1)

    influx_client, influx_write_api = connect_influxdb()
    threads = []
    for stream_config in streams_data:
        if not all(k in stream_config for k in ("url", "title", "slug")):
            logger.warning(f"Skipping stream due to missing keys: {stream_config}")
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
