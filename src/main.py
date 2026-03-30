from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal
import logging
import time

import supervision as sv
import numpy as np

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


try:
    from src.input.video import video_frames
    from src.input.rtsp import rtsp_frames
    from src.pipeline.detector import PersonDetector
    from src.pipeline.processor import ZoneCounter, ZoneDefinition
    from src.pipeline.tracker import PersonTracker
    from src.services.firestore_service import FirestoreZoneCountPublisher
    from src.visualization.debug_view import ZoneDebugRenderer
except ModuleNotFoundError:
    from input.video import video_frames
    from input.rtsp import rtsp_frames
    from pipeline.detector import PersonDetector
    from pipeline.processor import ZoneCounter, ZoneDefinition
    from pipeline.tracker import PersonTracker
    from services.firestore_service import FirestoreZoneCountPublisher
    from visualization.debug_view import ZoneDebugRenderer


ZONE_REF_W = 1920
ZONE_REF_H = 1080
SourceKind = Literal["video", "rtsp"]

BARISTA_ZONE = np.array(
    [
        [115, 376],
        [487, 203],
        [476, 15],
        [582, 5],
        [645, 170],
        [964, 369],
        [1452, 732],
        [1813, 993],
        [1026, 969],
        [918, 987],
        [473, 984],
        [266, 706],
    ],
    dtype=np.float32,
)

# Frame size: 1920 x 1080
CUSTOMER_ZONE = np.array(
    [
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
    dtype=np.float32,
)

MODEL_PATH = Path("models/yolo26x.onnx")
REID_MODEL_PATH = Path("models/osnet_x1_0_msmt17.pt")
VIDEO_PATH = Path("video/walking_inout_zone.mp4")
SOURCE_KIND: SourceKind = "rtsp"
RTSP_URL = "rtsp://admin:q1w2e3r4@192.168.0.102:554/Streaming/Channels/301"
BARISTA_ZONE_NAME = "BARISTA_ZONE"
BARISTA_ABSENCE_CONFIRM_S = 10.0
BARISTA_ALERT_COOLDOWN_S = 60.0


def get_frame_source() -> tuple[Iterator[np.ndarray], float]:
    if SOURCE_KIND == "video":
        if not VIDEO_PATH.exists():
            raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
        return (
            video_frames(VIDEO_PATH),
            sv.VideoInfo.from_video_path(str(VIDEO_PATH)).fps,
        )

    if SOURCE_KIND == "rtsp":
        return rtsp_frames(RTSP_URL), 30.0

    raise ValueError(f"Unsupported SOURCE_KIND: {SOURCE_KIND}")


def build_zones() -> list[ZoneDefinition]:
    return [
        ZoneDefinition(name="BARISTA_ZONE", polygon=BARISTA_ZONE, color=(40, 220, 20)),
        ZoneDefinition(
            name="CUSTOMER_ZONE", polygon=CUSTOMER_ZONE, color=(20, 180, 255)
        ),
    ]


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    frames, source_fps = get_frame_source()
    fps = max(1, int(round(source_fps)))
    delay_ms = 1 if SOURCE_KIND == "rtsp" else max(1, int(1000 / fps))

    zones = build_zones()

    detector = PersonDetector(model_path=MODEL_PATH)
    tracker = PersonTracker(
        reid_model_path=REID_MODEL_PATH,
        det_threshold=0.25,
        with_reid=False,
    )
    processor = ZoneCounter(ref_width=ZONE_REF_W, ref_height=ZONE_REF_H, zones=zones)
    renderer = ZoneDebugRenderer(window_name="YOLO Zone Counting")
    firestore_publisher = FirestoreZoneCountPublisher.from_env()

    barista_missing_since_ts: float | None = None
    absence_alert_sent = False
    last_absence_alert_ts = 0.0

    for frame in frames:
        frame_height, frame_width = frame.shape[:2]
        detections = detector.detect(frame)
        tracked_detections = tracker.update(frame=frame, detections=detections)
        zone_detections, zone_counts, polygons = processor.process(
            detections=tracked_detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        firestore_publisher.publish(zone_counts=zone_counts)

        now = time.monotonic()
        barista_count = int(zone_counts.get(BARISTA_ZONE_NAME, 0))

        if barista_count <= 0:
            if barista_missing_since_ts is None:
                barista_missing_since_ts = now

            absent_for_s = now - barista_missing_since_ts
            should_send_alert = (
                absent_for_s >= BARISTA_ABSENCE_CONFIRM_S
                and (
                    not absence_alert_sent
                    or (now - last_absence_alert_ts) >= BARISTA_ALERT_COOLDOWN_S
                )
            )
            if should_send_alert:
                firestore_publisher.publish_alert(
                    event_type="barista_absent",
                    zone_name=BARISTA_ZONE_NAME,
                    details={
                        "barista_count": 0,
                        "absent_for_s": round(absent_for_s, 1),
                    },
                )
                absence_alert_sent = True
                last_absence_alert_ts = now
        else:
            if barista_missing_since_ts is not None and absence_alert_sent:
                firestore_publisher.publish_alert(
                    event_type="barista_returned",
                    zone_name=BARISTA_ZONE_NAME,
                    details={"barista_count": barista_count},
                )

            barista_missing_since_ts = None
            absence_alert_sent = False

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
