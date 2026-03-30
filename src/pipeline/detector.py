from __future__ import annotations

from pathlib import Path

import supervision as sv
from ultralytics import YOLO


class PersonDetector:
	def __init__(self, model_path: str | Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> None:
		self.model_path = Path(model_path)
		if not self.model_path.exists():
			raise FileNotFoundError(f"Model not found: {self.model_path}")

		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold
		self.model = YOLO(str(self.model_path))

	def detect(self, frame) -> sv.Detections:
		result = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
		detections = sv.Detections.from_ultralytics(result)

		if detections.class_id is None:
			return detections

		person_mask = detections.class_id == 0
		return detections[person_mask]
