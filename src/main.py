from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

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
SourceKind = Literal["video", "rtsp"]

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

# Frame size: 1920 x 1080
CUSTOMER_ZONE = np.array([
    [673, 74],
    [763, 93],
    [985, 160],
    [1025, 128],
    [1126, 129],
    [1182, 169],
    [1250, 164],
    [1380, 140],
    [1510, 164],
    [1588, 187],
    [1601, 136],
    [1663, 7],
    [1918, 5],
    [1917, 998],
    [1636, 828],
    [1594, 834],
    [1471, 752],
    [1552, 548],
    [1181, 349],
    [1085, 410],
    [796, 266],
    [734, 226],
    [707, 161],
    [668, 141],
], 
dtype=np.float32
)

MODEL_PATH = Path("models/yolo26x.onnx")
VIDEO_PATH = Path("video/walking_inout_zone.mp4")
SOURCE_KIND: SourceKind = "video"
RTSP_URL = "rtsp://admin:q1w2e3r4@192.168.0.102:554/Streaming/Channels/301"


def get_frame_source() -> tuple[Iterator[np.ndarray], float]:
	if SOURCE_KIND == "video":
		if not VIDEO_PATH.exists():
			raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
		return video_frames(VIDEO_PATH), sv.VideoInfo.from_video_path(str(VIDEO_PATH)).fps

	if SOURCE_KIND == "rtsp":
		return rtsp_frames(RTSP_URL), 30.0

	raise ValueError(f"Unsupported SOURCE_KIND: {SOURCE_KIND}")


def build_zones() -> list[ZoneDefinition]:
	return [
		ZoneDefinition(name="BARISTA_ZONE", polygon=BARISTA_ZONE, color=(40, 220, 20)),
		ZoneDefinition(name="CUSTOMER_ZONE", polygon=CUSTOMER_ZONE, color=(20, 180, 255)),
	]


def main() -> None:
	frames, source_fps = get_frame_source()
	fps = max(1, int(round(source_fps)))
	delay_ms = max(1, int(1000 / fps))

	zones = build_zones()

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
		if key in (27, ord("q")):
			break

	renderer.close()


if __name__ == "__main__":
	main()
