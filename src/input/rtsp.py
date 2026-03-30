from __future__ import annotations

import threading
import time
from typing import Iterator

import cv2
import numpy as np


def rtsp_frames(rtsp_url: str) -> Iterator[np.ndarray]:
	cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	latest_frame: np.ndarray | None = None
	latest_seq = -1
	read_error = False
	stop_event = threading.Event()
	lock = threading.Lock()

	def _reader() -> None:
		nonlocal latest_frame, latest_seq, read_error
		while not stop_event.is_set():
			ok, frame = cap.read()
			if not ok:
				read_error = True
				time.sleep(0.01)
				continue
			with lock:
				latest_frame = frame
				latest_seq += 1
				read_error = False

	reader_thread = threading.Thread(target=_reader, name="rtsp-reader", daemon=True)
	reader_thread.start()

	last_yielded_seq = -1

	try:
		while True:
			with lock:
				seq = latest_seq
				frame = None if latest_frame is None else latest_frame.copy()

			if frame is None:
				if read_error:
					time.sleep(0.01)
					continue
				time.sleep(0.001)
				continue

			if seq == last_yielded_seq:
				time.sleep(0.001)
				continue

			last_yielded_seq = seq
			yield frame
	finally:
		stop_event.set()
		reader_thread.join(timeout=0.5)
		cap.release()
