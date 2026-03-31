from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal
import logging
import time
import argparse
import os
from collections import defaultdict, deque

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
        [107, 335],
        [488, 171],
        [471, 5],
        [651, 5],
        [664, 187],
        [1080, 413],
        [1183, 356],
        [1542, 550],
        [1458, 756],
        [1616, 855],
        [1822, 474],
        [1919, 476],
        [1908, 962],
        [321, 955],
        [254, 800],
        [211, 690],
        [195, 649],
        [177, 608],
        [159, 563],
        [150, 516],
        [139, 478],
        [132, 454],
        [120, 411],
        [114, 373],
    ],
    dtype=np.float32,
)

# Frame size: 1920 x 1080
CUSTOMER_ZONE = np.array(
    [
        [673, 77],
        [750, 91],
        [900, 122],
        [1003, 114],
        [1105, 137],
        [1280, 131],
        [1404, 124],
        [1502, 120],
        [1608, 150],
        [1674, 12],
        [1912, 21],
        [1851, 381],
        [1587, 824],
        [1474, 748],
        [1557, 550],
        [1182, 346],
        [1080, 399],
        [753, 227],
        [713, 160],
        [675, 135],
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


def get_frame_source(
    source_kind: SourceKind,
    video_path: Path,
    rtsp_url: str,
    rtsp_source_fps: float,
) -> tuple[Iterator[np.ndarray], float]:
    if source_kind == "video":
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return (
            video_frames(video_path),
            sv.VideoInfo.from_video_path(str(video_path)).fps,
        )

    if source_kind == "rtsp":
        return rtsp_frames(rtsp_url), max(1.0, rtsp_source_fps)

    raise ValueError(f"Unsupported SOURCE_KIND: {source_kind}")


def build_zones() -> list[ZoneDefinition]:
    return [
        ZoneDefinition(name="BARISTA_ZONE", polygon=BARISTA_ZONE, color=(40, 220, 20)),
        ZoneDefinition(
            name="CUSTOMER_ZONE", polygon=CUSTOMER_ZONE, color=(20, 180, 255)
        ),
    ]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(min_value, float(raw))
    except ValueError:
        return default


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="People detection and zone counting")
    parser.add_argument(
        "--source-kind",
        choices=["video", "rtsp"],
        default=None,
        help="Input source type.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to local video file when source-kind=video.",
    )
    parser.add_argument(
        "--rtsp-url",
        type=str,
        default=None,
        help="RTSP URL when source-kind=rtsp.",
    )
    parser.add_argument(
        "--rtsp-source-fps",
        type=float,
        default=None,
        help="Expected RTSP FPS if stream metadata is unavailable.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Person detector model path.",
    )
    parser.add_argument(
        "--reid-model-path",
        type=str,
        default=None,
        help="ReID model path (used only if ReID is enabled).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable OpenCV display and run in production mode.",
    )
    parser.add_argument(
        "--with-reid",
        action="store_true",
        help="Enable ReID in tracker (more stable IDs, slower).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Process every Nth frame (e.g. 5 or 10).",
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=None,
        help="Detector input size. Lower value is faster, higher may be more accurate.",
    )
    parser.add_argument(
        "--tracker-det-threshold",
        type=float,
        default=None,
        help="Tracker detection threshold.",
    )
    parser.add_argument(
        "--fps-log-interval-s",
        type=float,
        default=None,
        help="How often to log runtime performance metrics.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    run_mode = os.getenv("RUN_MODE", "debug").strip().lower()
    source_kind: SourceKind = (
        args.source_kind or os.getenv("SOURCE_KIND", str(SOURCE_KIND))
    ).strip().lower()  # type: ignore[assignment]
    if source_kind not in {"video", "rtsp"}:
        raise ValueError(f"Invalid SOURCE_KIND: {source_kind}")
    video_path = Path(args.video_path or os.getenv("VIDEO_PATH", str(VIDEO_PATH)))
    rtsp_url = args.rtsp_url or os.getenv("RTSP_URL", RTSP_URL)
    rtsp_source_fps = (
        args.rtsp_source_fps
        if args.rtsp_source_fps is not None
        else _env_float("RTSP_SOURCE_FPS", default=30.0, min_value=1.0)
    )
    model_path = Path(args.model_path or os.getenv("MODEL_PATH", str(MODEL_PATH)))
    reid_model_path = Path(
        args.reid_model_path or os.getenv("REID_MODEL_PATH", str(REID_MODEL_PATH))
    )
    display_enabled = (not args.headless) and _env_bool(
        "DISPLAY_ENABLED", default=run_mode != "production"
    )
    frame_stride = args.frame_stride or _env_int("FRAME_STRIDE", default=1, min_value=1)
    detector_imgsz = args.detector_imgsz or _env_int(
        "DETECTOR_IMGSZ", default=960, min_value=320
    )
    with_reid = args.with_reid or _env_bool("WITH_REID", default=False)
    tracker_det_threshold = (
        args.tracker_det_threshold
        if args.tracker_det_threshold is not None
        else _env_float("TRACKER_DET_THRESHOLD", default=0.25, min_value=0.01)
    )
    zone_smoothing_window = _env_int("ZONE_COUNT_SMOOTHING_WINDOW", default=3, min_value=1)
    fps_log_interval_s = (
        args.fps_log_interval_s
        if args.fps_log_interval_s is not None
        else _env_float("FPS_LOG_INTERVAL_S", default=5.0, min_value=1.0)
    )

    logging.info(
        "Runtime config: source_kind=%s display_enabled=%s frame_stride=%s detector_imgsz=%s with_reid=%s zone_smoothing_window=%s",
        source_kind,
        display_enabled,
        frame_stride,
        detector_imgsz,
        with_reid,
        zone_smoothing_window,
    )

    frames, source_fps = get_frame_source(
        source_kind=source_kind,
        video_path=video_path,
        rtsp_url=rtsp_url,
        rtsp_source_fps=rtsp_source_fps,
    )
    fps = max(1, int(round(source_fps)))
    delay_ms = 1 if source_kind == "rtsp" else max(1, int(1000 / fps))

    zones = build_zones()

    detector = PersonDetector(model_path=model_path, imgsz=detector_imgsz)
    tracker = PersonTracker(
        reid_model_path=reid_model_path,
        det_threshold=tracker_det_threshold,
        with_reid=with_reid,
    )
    processor = ZoneCounter(ref_width=ZONE_REF_W, ref_height=ZONE_REF_H, zones=zones)
    renderer = ZoneDebugRenderer(window_name="YOLO Zone Counting") if display_enabled else None
    firestore_publisher = FirestoreZoneCountPublisher.from_env()
    smoothers: dict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=zone_smoothing_window)
    )

    barista_missing_since_ts: float | None = None
    absence_alert_sent = False
    last_absence_alert_ts = 0.0
    wall_start_ts = time.monotonic()
    last_perf_log_ts = wall_start_ts
    processed_frames = 0
    last_processed_frames = 0
    source_frames_seen = 0

    for frame_idx, frame in enumerate(frames, start=1):
        source_frames_seen = frame_idx
        if frame_stride > 1 and (frame_idx % frame_stride) != 0:
            continue

        processed_frames += 1
        frame_height, frame_width = frame.shape[:2]
        detections = detector.detect(frame)
        tracked_detections = tracker.update(frame=frame, detections=detections)
        zone_detections, zone_counts, polygons = processor.process(
            detections=tracked_detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        stable_zone_counts: dict[str, int] = {}
        for zone_name, count in zone_counts.items():
            smoothers[zone_name].append(int(count))
            # Median over recent processed frames reduces count flicker.
            stable_zone_counts[zone_name] = int(np.median(np.array(smoothers[zone_name])))
        firestore_publisher.publish(zone_counts=stable_zone_counts)

        now = time.monotonic()
        barista_count = int(stable_zone_counts.get(BARISTA_ZONE_NAME, 0))

        if barista_count <= 0:
            if barista_missing_since_ts is None:
                barista_missing_since_ts = now

            absent_for_s = now - barista_missing_since_ts
            should_send_alert = absent_for_s >= BARISTA_ABSENCE_CONFIRM_S and (
                not absence_alert_sent
                or (now - last_absence_alert_ts) >= BARISTA_ALERT_COOLDOWN_S
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

        if renderer is not None:
            annotated = renderer.render(
                frame=frame,
                detections=zone_detections,
                zones=zones,
                polygons=polygons,
                zone_counts=stable_zone_counts,
            )
            key = renderer.show(annotated, delay_ms=delay_ms)
            if key in (27, ord("q")):
                break

        now_perf = time.monotonic()
        if now_perf - last_perf_log_ts >= fps_log_interval_s:
            elapsed_total_s = max(1e-6, now_perf - wall_start_ts)
            elapsed_window_s = max(1e-6, now_perf - last_perf_log_ts)
            effective_fps = source_frames_seen / elapsed_total_s
            processed_fps = processed_frames / elapsed_total_s
            processed_fps_window = (processed_frames - last_processed_frames) / elapsed_window_s
            skipped_frames = source_frames_seen - processed_frames
            skip_ratio = skipped_frames / max(1, source_frames_seen)
            logging.info(
                "Perf: source_frames=%s processed_frames=%s skipped_frames=%s skip_ratio=%.2f effective_fps=%.2f processed_fps=%.2f processed_fps_window=%.2f detections=%s tracks=%s",
                source_frames_seen,
                processed_frames,
                skipped_frames,
                skip_ratio,
                effective_fps,
                processed_fps,
                processed_fps_window,
                len(detections),
                len(tracked_detections),
            )
            last_perf_log_ts = now_perf
            last_processed_frames = processed_frames

    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
