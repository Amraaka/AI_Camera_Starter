# Realtime Video AI Pipeline

Scalable camera-centric pipeline for RTSP ingest, AI processing, and business event generation.

## Services

- `services/stream-reader`: Pulls RTSP streams and emits normalized frame messages.
- `services/ai-workers`: Runs detector, tracker, and re-identification stages.
- `services/event-builder`: Converts raw tracks/detections into business events.
- `services/api`: Camera lifecycle and event query APIs.

## Shared Libraries

- `libs/contracts`: Versioned JSON schemas for inter-service messages.
- `libs/common`: Shared configuration, logging, and utility modules.

## Data Flow

`RTSP -> Frame Queue -> Workers -> Event Queue -> DB`

Each `cameraId` owns an independent logical pipeline state.

## Firestore Live Zone Counts

The pipeline can publish per-zone people counts to Firestore on every live change.

### 1) Install Python dependency

```bash
pip install firebase-admin
```

### 2) Set environment variables

```bash
export FIRESTORE_ENABLED=true
export FIREBASE_CREDENTIALS=/absolute/path/to/firebase-service-account.json
export FIRESTORE_COLLECTION=zone_counts
export FIRESTORE_ALERT_COLLECTION=zone_alerts
export FIRESTORE_DOCUMENT_ID=live
export FIRESTORE_CAMERA_ID=camera_1
export FIRESTORE_MIN_PUBLISH_INTERVAL_S=0.2
export FIRESTORE_ONLY_ON_CHANGE=true
```

Or use a `.env` file in the project root.

```bash
cp .env.example .env
# Edit .env and set FIREBASE_CREDENTIALS to your real absolute path
```

`src/main.py` now auto-loads `.env` at startup.

### 3) Run the app

`src/main.py` initializes `FirestoreZoneCountPublisher` and writes the latest payload to:

`<collection>/<document_id>`

Payload shape:

```json
{
	"camera_id": "camera_1",
	"zone_counts": {
		"BARISTA_ZONE": 2,
		"CUSTOMER_ZONE": 5
	},
	"updated_at_unix_ms": 1711800000000
}
```

## Stable Person IDs (OSNet ReID)

The pipeline now uses appearance-assisted tracking for more stable IDs across frames.

- Detector: YOLO person detections
- Tracker: IoU + OSNet embedding matching (Hungarian assignment)
- ReID model: `models/osnet_x1_0_msmt17.pt`

Install dependencies:

```bash
pip install torchreid tensorboard scipy
```

## Barista Absence Alert

The pipeline now sends alert events when no one is detected in `BARISTA_ZONE` for a sustained period.

- Confirm duration before alert: `10s`
- Repeat cooldown while still absent: `60s`
- Event types: `barista_absent`, `barista_returned`

Alert events are written to:

`<FIRESTORE_ALERT_COLLECTION>` (default: `zone_alerts`)

Example event document:

```json
{
	"camera_id": "camera_1",
	"event_type": "barista_absent",
	"zone_name": "BARISTA_ZONE",
	"barista_count": 0,
	"absent_for_s": 12.4,
	"created_at_unix_ms": 1711800000000
}
```

## Frontend Integration (Firebase Web SDK)

Use your frontend to subscribe to `zone_alerts` and show live notifications.

```javascript
import { initializeApp } from "firebase/app";
import {
	getFirestore,
	collection,
	query,
	where,
	orderBy,
	limit,
	onSnapshot,
} from "firebase/firestore";

const firebaseConfig = {
	apiKey: "...",
	authDomain: "...",
	projectId: "...",
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

const alertsRef = collection(db, "zone_alerts");
const q = query(
	alertsRef,
	where("camera_id", "==", "camera_1"),
	orderBy("created_at_unix_ms", "desc"),
	limit(20),
);

const seen = new Set();

onSnapshot(q, (snap) => {
	snap.docChanges().forEach((change) => {
		if (change.type !== "added") return;
		if (seen.has(change.doc.id)) return;
		seen.add(change.doc.id);

		const event = change.doc.data();
		if (event.event_type === "barista_absent") {
			console.warn("ALERT: No barista in zone", event);
			// Show toast, modal, bell badge, sound, etc.
		}

		if (event.event_type === "barista_returned") {
			console.info("Recovery: Barista returned", event);
		}
	});
});
```

Recommended frontend behavior:

1. Show red alert toast/banner for `barista_absent`.
2. Show green recovery toast for `barista_returned`.
3. Keep a notification history panel from the latest alert docs.
4. Filter by `camera_id` if you have multiple cameras.
