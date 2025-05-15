import os
import json
import numpy as np
import cv2
from influxdb_client import InfluxDBClient
from collections import defaultdict

INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://ouoite:8086")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "my-token")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "my-org")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "object_detection")
FRAME_WIDTH = 640
FRAME_HEIGHT = 384

YOUTUBE_STREAMS = os.environ.get("YOUTUBE_STREAMS", '[{"url":"https://www.youtube.com/watch?v=6MMXJrzT5c0","title":"Henry Africa\'s Bar & Café | Soi Green Mango","slug":"henry-green-mango"},{"url":"https://www.youtube.com/watch?v=w47yvCftkWQ","title":"Bondi Aussie Bar & Grill | Lamai","slug":"bondi-lamai"},{"url":"https://www.youtube.com/watch?v=U1LMXaV3sYI","title":"Tropical Murphy\'s Irish Pub | Chaweng","slug":"tropical-chaweng"},{"url":"https://www.youtube.com/watch?v=OFVpXuZOlPc","title":"The Shack | Fisherman\'s Village","slug":"shack-fisherman"}]')

def group_bbox_records(result):
    grouped = defaultdict(dict)
    for table in result:
        for record in table.records:
            t = record.get_time()
            obj = record.values.get("object")
            print(record)
            print(obj)
            field = record.get_field()
            value = record.get_value()
            if obj is not None:
                grouped[t]['object'] = obj
            if field in {"x1", "y1", "x2", "y2"}:
                grouped[t][field] = value
    bboxes = []
    for t, rec in grouped.items():
        if all(k in rec for k in ("object", "x1", "y1", "x2", "y2")):
            bboxes.append(rec)
    return bboxes

def main():
    streams = json.loads(YOUTUBE_STREAMS)
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    color_map = {}

    for stream in streams:
        stream_url = stream["url"]
        stream_slug = stream["slug"]
        stream_title = stream["title"]
        print(f"Processing stream: {stream_title}")

        query = f'''
from(bucket:"{INFLUXDB_BUCKET}")
  |> range(start: -1d)
  |> filter(fn: (r) => r._measurement == "object_detections")
  |> filter(fn: (r) => r.stream_url == "{stream_url}")
'''
        result = query_api.query(org=INFLUXDB_ORG, query=query)
        bboxes = group_bbox_records(result)
        if not bboxes:
            print(f"No detections found for {stream_title}")
            continue

        # Draw bounding boxes
        image = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
        for bbox in bboxes:
            obj = bbox["object"]
            if obj not in color_map:
                color_map[obj] = tuple(np.random.randint(0, 255, 3).tolist())
            color = color_map[obj]
            pt1 = (int(float(bbox["x1"])), int(float(bbox["y1"])))
            pt2 = (int(float(bbox["x2"])), int(float(bbox["y2"])))
            cv2.rectangle(image, pt1, pt2, color, 1)

        # Optionally add a legend
        y0 = 30
        for idx, (cls, color) in enumerate(color_map.items()):
            cv2.rectangle(image, (10, y0 + idx*25 - 15), (30, y0 + idx*25 + 5), color, -1)
            cv2.putText(image, cls, (40, y0 + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        out_path = f"{stream_slug}_detections.png"
        cv2.imwrite(out_path, image)
        print(f"Saved: {out_path}")

    client.close()

if __name__ == "__main__":
    main()
