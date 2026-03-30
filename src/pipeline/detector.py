from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class PersonDetector:
	def __init__(
		self,
		model_path: str | Path,
		conf_threshold: float = 0.25,
		iou_threshold: float = 0.45,
	) -> None:
		self.model_path = Path(model_path)
		if not self.model_path.exists():
			raise FileNotFoundError(f"Model not found: {self.model_path}")

		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold
		self.model = YOLO(str(self.model_path))

	def detect(self, frame: np.ndarray) -> np.ndarray:
		result = self.model.predict(
			frame,
			conf=self.conf_threshold,
			iou=self.iou_threshold,
			classes=[0],
			verbose=False,
		)[0]

		boxes = result.boxes
		if boxes is None or len(boxes) == 0:
			return np.empty((0, 6), dtype=np.float32)

		xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
		conf = boxes.conf.detach().cpu().numpy().astype(np.float32)[:, np.newaxis]
		cls = boxes.cls.detach().cpu().numpy().astype(np.float32)[:, np.newaxis]
		return np.concatenate([xyxy, conf, cls], axis=1)
