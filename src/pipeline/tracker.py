from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
import torch
from boxmot import BotSort


class PersonTracker:
	def __init__(
		self,
		reid_model_path: str | Path,
		det_threshold: float = 0.25,
		with_reid: bool = False,
	) -> None:
		reid_path = Path(reid_model_path)
		if with_reid and not reid_path.exists():
			raise FileNotFoundError(f"ReID model not found: {reid_path}")

		self._tracker = BotSort(
			reid_weights=reid_path if with_reid else None,
			device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
			half=False,
			track_high_thresh=float(det_threshold),
			with_reid=with_reid,
		)

	@staticmethod
	def _empty_detections() -> sv.Detections:
		return sv.Detections(
			xyxy=np.empty((0, 4), dtype=np.float32),
			confidence=np.empty((0,), dtype=np.float32),
			class_id=np.empty((0,), dtype=np.int32),
			tracker_id=np.empty((0,), dtype=np.int32),
		)

	def update(self, frame: np.ndarray, detections: np.ndarray) -> sv.Detections:
		if detections.size == 0:
			tracks = self._tracker.update(np.empty((0, 6), dtype=np.float32), frame)
			_ = tracks
			return self._empty_detections()

		tracks = self._tracker.update(detections, frame)
		if tracks is None or len(tracks) == 0:
			return self._empty_detections()

		tracks_np = np.asarray(tracks, dtype=np.float32)
		xyxy = tracks_np[:, 0:4]
		tracker_id = tracks_np[:, 4].astype(np.int32)
		confidence = tracks_np[:, 5].astype(np.float32) if tracks_np.shape[1] > 5 else np.ones((len(tracks_np),), dtype=np.float32)
		class_id = tracks_np[:, 6].astype(np.int32) if tracks_np.shape[1] > 6 else np.zeros((len(tracks_np),), dtype=np.int32)

		return sv.Detections(
			xyxy=xyxy,
			confidence=confidence,
			class_id=class_id,
			tracker_id=tracker_id,
		)
