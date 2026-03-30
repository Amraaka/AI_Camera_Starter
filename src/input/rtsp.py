from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def rtsp_frames(rtsp_url: str) -> Iterator[np.ndarray]:
	cap = cv2.VideoCapture(rtsp_url)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break
			yield frame
	finally:
		cap.release()
