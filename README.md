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
