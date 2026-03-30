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
