from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv


@dataclass(frozen=True)
class ZoneDefinition:
	name: str
	polygon: np.ndarray
	color: tuple[int, int, int]


class ZoneCounter:
	def __init__(self, ref_width: int, ref_height: int, zones: list[ZoneDefinition]) -> None:
		self.ref_width = ref_width
		self.ref_height = ref_height
		self.zones = zones
		self._cache_frame_size: tuple[int, int] | None = None
		self._cache_polygons: dict[str, np.ndarray] = {}

	def _scale_polygon(self, polygon: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
		scale_x = frame_width / self.ref_width
		scale_y = frame_height / self.ref_height
		scaled = polygon.copy()
		scaled[:, 0] *= scale_x
		scaled[:, 1] *= scale_y
		return scaled.astype(np.int32)

	def _scaled_polygons(self, frame_width: int, frame_height: int) -> dict[str, np.ndarray]:
		frame_size = (frame_width, frame_height)
		if self._cache_frame_size == frame_size:
			return self._cache_polygons

		self._cache_polygons = {
			zone.name: self._scale_polygon(zone.polygon, frame_width, frame_height)
			for zone in self.zones
		}
		self._cache_frame_size = frame_size
		return self._cache_polygons

	@staticmethod
	def _in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
		if points_xy.size == 0:
			return np.zeros((0,), dtype=bool)

		mask = np.zeros((points_xy.shape[0],), dtype=bool)
		for idx, (x, y) in enumerate(points_xy):
			mask[idx] = cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
		return mask

	def process(
		self,
		detections: sv.Detections,
		frame_width: int,
		frame_height: int,
	) -> tuple[sv.Detections, dict[str, int], dict[str, np.ndarray]]:
		polygons = self._scaled_polygons(frame_width, frame_height)

		anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
		in_any_zone = np.zeros((len(detections),), dtype=bool)
		zone_counts: dict[str, int] = {}

		for zone in self.zones:
			mask = self._in_polygon(anchors, polygons[zone.name])
			zone_counts[zone.name] = int(np.sum(mask))
			in_any_zone |= mask

		zone_only_detections = detections[in_any_zone]
		return zone_only_detections, zone_counts, polygons
