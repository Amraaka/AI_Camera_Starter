from __future__ import annotations

from pathlib import Path

import supervision as sv
import numpy as np

try:
	from src.input.video import video_frames
	from src.input.rtsp import rtsp_frames
	from src.pipeline.detector import PersonDetector
	from src.pipeline.processor import ZoneCounter, ZoneDefinition
	from src.visualization.debug_view import ZoneDebugRenderer
except ModuleNotFoundError:
	from input.video import video_frames
	from input.rtsp import rtsp_frames
	from pipeline.detector import PersonDetector
	from pipeline.processor import ZoneCounter, ZoneDefinition
	from visualization.debug_view import ZoneDebugRenderer


ZONE_REF_W = 1920
ZONE_REF_H = 1080

BARISTA_ZONE = np.array(
	[
		[85, 321],
		[495, 117],
		[580, 93],
		[665, 154],
		[752, 252],
		[1054, 415],
		[1179, 355],
		[1531, 544],
		[1456, 752],
		[1820, 971],
		[7, 968],
		[3, 358],
		[4, 314],
		[44, 324],
	],
	dtype=np.float32,
)

CUSTOMER_ZONE = np.array(
	[
		[597, 1],
		[1753, 2],
		[1753, 848],
		[1586, 827],
		[1471, 755],
		[1555, 543],
		[1181, 346],
		[1065, 408],
		[657, 196],
		[617, 142],
		[596, 45],
		[596, 15],
	],
	dtype=np.float32,
)

MODEL_PATH = Path("models/yolo26x.onnx")
VIDEO_PATH = Path("video/walking_inout_zone.mp4")
SOURCE_KIND = "rtsp"  # "video" | "rtsp"
RTSP_URL = "rtsp://admin:q1w2e3r4@192.168.0.102:554/Streaming/Channels/301"


def get_frame_source():
	if SOURCE_KIND == "video":
		if not VIDEO_PATH.exists():
			raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
		return video_frames(VIDEO_PATH), sv.VideoInfo.from_video_path(str(VIDEO_PATH)).fps

	if SOURCE_KIND == "rtsp":
		return rtsp_frames(RTSP_URL), 30.0

	raise ValueError(f"Unsupported SOURCE_KIND: {SOURCE_KIND}")


def main() -> None:
	frames, source_fps = get_frame_source()
	fps = max(1, int(round(source_fps)))
	delay_ms = max(1, int(1000 / fps))

	zones = [
		ZoneDefinition(name="BARISTA_ZONE", polygon=BARISTA_ZONE, color=(40, 220, 20)),
		ZoneDefinition(name="CUSTOMER_ZONE", polygon=CUSTOMER_ZONE, color=(20, 180, 255)),
	]

	detector = PersonDetector(model_path=MODEL_PATH)
	processor = ZoneCounter(ref_width=ZONE_REF_W, ref_height=ZONE_REF_H, zones=zones)
	renderer = ZoneDebugRenderer(window_name="YOLO Zone Counting")

	for frame in frames:
		frame_height, frame_width = frame.shape[:2]
		detections = detector.detect(frame)
		zone_detections, zone_counts, polygons = processor.process(
			detections=detections,
			frame_width=frame_width,
			frame_height=frame_height,
		)

		annotated = renderer.render(
			frame=frame,
			detections=zone_detections,
			zones=zones,
			polygons=polygons,
			zone_counts=zone_counts,
		)
		key = renderer.show(annotated, delay_ms=delay_ms)
		if key == 27 or key == ord("q"):
			break

	renderer.close()


if __name__ == "__main__":
	main()
