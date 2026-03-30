"""
Get video information and compare multiple videos.
Uses OpenCV for metadata and os for file size.
Run: python tool/video_info.py
"""
import os
import sys
from pathlib import Path

import cv2


def get_video_info(path: str) -> dict | None:
    """Extract video metadata and file size. Returns None if video cannot be opened."""
    path = str(path)
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {path}"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join(chr((codec >> (8 * i)) & 0xFF) for i in range(4)) if codec else "?"

    duration_sec = frame_count / fps if fps and fps > 0 else 0

    cap.release()

    file_size = os.path.getsize(path)

    return {
        "path": path,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "duration_str": _format_duration(duration_sec),
        "codec": codec_str,
        "file_size_bytes": file_size,
        "file_size_str": _format_size(file_size),
    }


def _format_duration(sec: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    if sec <= 0:
        return "0:00.000"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"{m}:{s:06.3f}"


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_info(info: dict) -> None:
    """Print video info in a readable format."""
    if "error" in info:
        print(f"  ERROR: {info['error']}")
        return
    print(f"  Path:       {info['path']}")
    print(f"  Resolution: {info['resolution']} (width x height)")
    print(f"  FPS:        {info['fps']}")
    print(f"  Frames:     {info['frame_count']}")
    print(f"  Duration:   {info['duration_str']} ({info['duration_sec']:.2f} sec)")
    print(f"  Codec:      {info['codec']}")
    print(f"  File size:  {info['file_size_str']} ({info['file_size_bytes']} bytes)")


def print_comparison(infos: list[dict]) -> None:
    """Print a side-by-side comparison table."""
    # Filter out errors
    valid = [i for i in infos if "error" not in i]
    if not valid:
        print("No valid videos to compare.")
        return

    keys = ["resolution", "width", "height", "fps", "frame_count", "duration_sec", "file_size_bytes", "file_size_str"]
    labels = ["Resolution", "Width", "Height", "FPS", "Frames", "Duration (sec)", "File size (bytes)", "File size"]

    # Use first row for display keys
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Property':<20} " + " | ".join(f"{Path(i['path']).name[:30]:<32}" for i in valid))
    print("-" * 80)

    for key, label in zip(keys, labels):
        vals = []
        for i in valid:
            v = i.get(key, "?")
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(f"{label:<20} " + " | ".join(f"{v:<32}" for v in vals))

    print("=" * 80)

    # Highlight differences
    if len(valid) >= 2:
        a, b = valid[0], valid[1]
        diffs = []
        if a.get("width") != b.get("width") or a.get("height") != b.get("height"):
            diffs.append("Resolution differs")
        if abs((a.get("fps") or 0) - (b.get("fps") or 0)) > 0.01:
            diffs.append("FPS differs")
        if diffs:
            print("\nDifferences:", ", ".join(diffs))


def main() -> None:
    videos = [
        "video/tek.mp4",
        "video/tek.mp4",
    ]

    if len(sys.argv) > 1:
        videos = sys.argv[1:]

    print("VIDEO INFORMATION")
    print("=" * 80)

    infos = []
    for v in videos:
        print(f"\n--- {v} ---")
        info = get_video_info(v)
        infos.append(info)
        print_info(info)

    if len(infos) >= 2:
        print_comparison(infos)


if __name__ == "__main__":
    main()
