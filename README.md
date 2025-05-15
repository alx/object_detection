# Object Detection from YouTube Streams with YOLOv4 and InfluxDB

This project provides a robust pipeline for real-time object detection on YouTube video streams using the YOLOv4 model. It leverages `yt-dlp` and `ffmpeg` for video capture, `darknetpy` for object detection, and logs detection results to InfluxDB for analytics and monitoring. The system is designed for multi-stream processing, scalability, and easy integration with visualization tools.

## **Features**

- **Real-Time Object Detection:**
Processes live or recorded YouTube streams, detecting objects in real-time using YOLOv4.

- **Multi-Stream Support:**
Handles multiple YouTube streams concurrently using threading.

- **Detection Logging:**
Stores detection results (object class, confidence, bounding box, stream info, timestamp) in InfluxDB for time-series analysis.

- **Debug Visualization:**
Optionally saves annotated frames with bounding boxes for debugging and model validation.

- **Configurable via Environment Variables:**
Easily adjust model paths, stream sources, detection interval, frame size, and database settings.

## **Architecture Overview**

- **Video Ingestion:**
Uses `yt-dlp` to fetch YouTube streams and pipes them to `ffmpeg` for frame extraction and resizing.

- **Object Detection:**
Each frame is processed by a YOLOv4 detector via `darknetpy`.

- **Result Logging:**
Detected objects are logged into InfluxDB with rich metadata.

- **Debug Output:**
If enabled, saves images with bounding boxes to disk.

## **Setup**

### **Dependencies**

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- ffmpeg
- numpy
- opencv-python
- influxdb-client
- [darknetpy](https://github.com/hank-ai/darknet)
- threading, logging, json

### **Environment Variables**

| Variable | Description | Default Value |
| :-- | :-- | :-- |
| INFLUXDB_URL | InfluxDB server URL | http://influxdb:8086 |
| INFLUXDB_TOKEN | InfluxDB API token | my-token |
| INFLUXDB_ORG | InfluxDB organization | my-org |
| INFLUXDB_BUCKET | InfluxDB bucket for detections | object_detection |
| YOUTUBE_STREAMS | JSON array of stream configs | [] |
| DETECTION_INTERVAL | Seconds between detections per stream | 2 |
| FRAME_WIDTH | Width of processed video frames | 1280 |
| FRAME_HEIGHT | Height of processed video frames | 720 |
| DARKNET_DATA | Path to YOLO data file | /models/coco.data |
| DARKNET_CFG | Path to YOLO config file | /models/yolov4.cfg |
| DARKNET_WEIGHTS | Path to YOLO weights file | /models/yolov4.weights |
| DEBUG_IMG_BBOX | Save debug images with bounding boxes | False |

### **YouTube Streams Format**

Set `YOUTUBE_STREAMS` as a JSON array of objects, each with:

```json
[
  {
    "url": "https://youtube.com/stream_url",
    "title": "Stream Title",
    "slug": "stream_slug"
  }
]
```


## **Usage**

1. **Install dependencies** and ensure `yt-dlp` and `ffmpeg` are in your PATH.
2. **Set environment variables** as needed.
3. **Run the application:**

```bash
python object_detection/app.py
```

4. **Monitor logs** for detection outputs and InfluxDB writes.
5. **(Optional)**: Enable `DEBUG_IMG_BBOX=True` to save annotated frames for inspection.

## **docker compose**

1. **Set environment variables** in `docker-compose.yml` as needed.
2. **Run the application:**

```bash
docker compose up -d
```

## **Detection Output in InfluxDB**

Each detection is written as a point with the following fields and tags:

- Tags: `object`, `stream_slug`, `stream_title`, `stream_url`
- Fields: `confidence`, `x1`, `y1`, `x2`, `y2`
- Timestamp: nanosecond precision


## **Customization**

- **Model Files:**
Adjust YOLO model paths via environment variables to use custom datasets or configurations.

- **Detection Interval:**
Tune `DETECTION_INTERVAL` for performance vs. detection frequency.

- **Frame Size:**
Change `FRAME_WIDTH` and `FRAME_HEIGHT` for different resolutions.


## **Logging \& Debugging**

- Logs are output to stdout with timestamps and severity levels.
- Errors in stream capture, detection, or database writes are logged.
- Debug images with bounding boxes are saved if enabled.


## **Extending**

- Integrate with Grafana or other visualization tools for real-time dashboards.
- Add support for other video sources or models.
- Enhance with alerting or notification systems based on detection events.
