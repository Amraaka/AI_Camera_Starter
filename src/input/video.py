from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def video_frames(video_path: str | Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()
