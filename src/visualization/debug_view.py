from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

try:
	from src.pipeline.processor import ZoneDefinition
except ModuleNotFoundError:
	from pipeline.processor import ZoneDefinition


class ZoneDebugRenderer:
	def __init__(self, window_name: str = "YOLO Zone Counting") -> None:
		self.window_name = window_name
		self.box_annotator = sv.BoxAnnotator()
		self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

	def render(
		self,
		frame: np.ndarray,
		detections: sv.Detections,
		zones: list[ZoneDefinition],
		polygons: dict[str, np.ndarray],
		zone_counts: dict[str, int],
	) -> np.ndarray:
		labels = []
		if detections.confidence is not None:
			labels = [f"person {conf:.2f}" for conf in detections.confidence]

		annotated = frame.copy()
		annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
		annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

		y_offset = 45
		for zone in zones:
			polygon = polygons[zone.name]
			cv2.polylines(annotated, [polygon], isClosed=True, color=zone.color, thickness=3)
			cv2.putText(
				annotated,
				f"{zone.name} people: {zone_counts.get(zone.name, 0)}",
				(30, y_offset),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				zone.color,
				2,
				cv2.LINE_AA,
			)
			y_offset += 40

		return annotated

	def show(self, frame: np.ndarray, delay_ms: int = 1) -> int:
		cv2.imshow(self.window_name, frame)
		return cv2.waitKey(delay_ms) & 0xFF

	@staticmethod
	def close() -> None:
		cv2.destroyAllWindows()
