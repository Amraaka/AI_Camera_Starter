from __future__ import annotations

import logging
import importlib
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FirestoreServiceConfig:
	enabled: bool = False
	credentials_path: str | None = None
	collection: str = "zone_counts"
	alert_collection: str = "zone_alerts"
	document_id: str = "live"
	camera_id: str = "camera_1"
	min_publish_interval_s: float = 0.2
	only_on_change: bool = True


class FirestoreZoneCountPublisher:
	def __init__(self, config: FirestoreServiceConfig) -> None:
		self._config = config
		self._log = logging.getLogger(__name__)
		self._enabled = False
		self._db = None
		self._last_zone_counts: dict[str, int] | None = None
		self._last_publish_ts: float = 0.0

		if not config.enabled:
			self._log.info("Firestore publisher is disabled.")
			return

		try:
			firebase_admin = importlib.import_module("firebase_admin")
			credentials = importlib.import_module("firebase_admin.credentials")
			firestore = importlib.import_module("firebase_admin.firestore")
		except Exception as exc:  # pragma: no cover - import path only
			self._log.warning(
				"Firestore enabled but firebase-admin is not available: %s", exc
			)
			return

		try:
			if config.credentials_path:
				cred = credentials.Certificate(config.credentials_path)
				firebase_admin.initialize_app(cred)
			else:
				firebase_admin.initialize_app()
		except ValueError:
			# App is already initialized in this process.
			pass
		except Exception as exc:  # pragma: no cover - external credential path
			self._log.warning("Failed to initialize Firebase app: %s", exc)
			return

		try:
			self._db = firestore.client()
			self._enabled = True
			self._log.info(
				"Firestore publisher enabled (collection=%s, document=%s).",
				config.collection,
				config.document_id,
			)
		except Exception as exc:  # pragma: no cover - external service path
			self._log.warning("Failed to create Firestore client: %s", exc)

	@classmethod
	def from_env(cls) -> "FirestoreZoneCountPublisher":
		enabled_raw = os.getenv("FIRESTORE_ENABLED", "false").strip().lower()
		enabled = enabled_raw in {"1", "true", "yes", "on"}

		credentials_path = os.getenv("FIREBASE_CREDENTIALS")
		collection = os.getenv("FIRESTORE_COLLECTION", "zone_counts")
		alert_collection = os.getenv("FIRESTORE_ALERT_COLLECTION", "zone_alerts")
		document_id = os.getenv("FIRESTORE_DOCUMENT_ID", "live")
		camera_id = os.getenv("FIRESTORE_CAMERA_ID", "camera_1")
		min_interval_raw = os.getenv("FIRESTORE_MIN_PUBLISH_INTERVAL_S", "0.2")
		only_on_change_raw = os.getenv("FIRESTORE_ONLY_ON_CHANGE", "true").strip().lower()

		try:
			min_publish_interval_s = max(0.0, float(min_interval_raw))
		except ValueError:
			min_publish_interval_s = 0.2

		only_on_change = only_on_change_raw in {"1", "true", "yes", "on"}

		config = FirestoreServiceConfig(
			enabled=enabled,
			credentials_path=credentials_path,
			collection=collection,
			alert_collection=alert_collection,
			document_id=document_id,
			camera_id=camera_id,
			min_publish_interval_s=min_publish_interval_s,
			only_on_change=only_on_change,
		)
		return cls(config)

	def publish(self, zone_counts: Mapping[str, int]) -> None:
		if not self._enabled or self._db is None:
			return

		zone_counts_dict = {name: int(count) for name, count in zone_counts.items()}
		now = time.monotonic()

		if self._config.only_on_change and self._last_zone_counts == zone_counts_dict:
			if now - self._last_publish_ts < self._config.min_publish_interval_s:
				return

		if now - self._last_publish_ts < self._config.min_publish_interval_s:
			return

		payload = {
			"camera_id": self._config.camera_id,
			"zone_counts": zone_counts_dict,
			"updated_at_unix_ms": int(time.time() * 1000),
		}

		try:
			(
				self._db.collection(self._config.collection)
				.document(self._config.document_id)
				.set(payload, merge=True)
			)
			self._last_zone_counts = zone_counts_dict
			self._last_publish_ts = now
		except Exception as exc:  # pragma: no cover - external service path
			self._log.warning("Firestore publish failed: %s", exc)

	def publish_alert(
		self,
		event_type: str,
		zone_name: str,
		details: Mapping[str, Any] | None = None,
	) -> None:
		if not self._enabled or self._db is None:
			return

		payload: dict[str, Any] = {
			"camera_id": self._config.camera_id,
			"event_type": event_type,
			"zone_name": zone_name,
			"created_at_unix_ms": int(time.time() * 1000),
		}
		if details:
			payload.update(dict(details))

		try:
			self._db.collection(self._config.alert_collection).add(payload)
		except Exception as exc:  # pragma: no cover - external service path
			self._log.warning("Firestore alert publish failed: %s", exc)
