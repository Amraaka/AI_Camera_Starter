import cv2
import numpy as np

def load_image_or_frame(path: str, frame_index: int = 0) -> tuple[np.ndarray | None, int, int]:
    path_lower = path.strip().lower()
    if path_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        frame = cv2.imread(path)
        if frame is None:
            return None, 0, 0
        h, w = frame.shape[:2]
        return frame, w, h
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, 0, 0
    h, w = frame.shape[:2]
    return frame, w, h


def main() -> None:
    n_edges = input("How many edges (vertices) for the polygon? ").strip()
    try:
        n = int(n_edges)
        if n < 3:
            raise SystemExit("Need at least 3 edges.")
    except ValueError:
        raise SystemExit("Enter a number (e.g. 4 for a quadrilateral).")

    path = input("Path to photo or video: ").strip()
    if not path:
        raise SystemExit("No path given.")

    frame_index = 0
    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        idx = input("Video: use frame index (0 = first frame, or press Enter for 0): ").strip()
        if idx:
            try:
                frame_index = int(idx)
            except ValueError:
                pass

    frame, w, h = load_image_or_frame(path, frame_index)
    if frame is None:
        raise SystemExit(f"Cannot load image or video: {path}")

    points: list[list[int]] = []
    display = frame.copy()

    def redraw() -> None:
        nonlocal display
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, tuple(points[i - 1]), tuple(pt), (0, 255, 0), 2)
        if len(points) == n:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)

    def on_mouse(event: int, x: int, y: int, *_args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append([x, y])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    win = "Polygon picker: left-click points, right-click undo, Enter when done"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    print(f"Click {n} points in order. Right-click to undo last. Press Enter when done.")
    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("\r") or key == ord("\n"):
            break
        if key == 27:  
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")
        if key == 8 and points:  
            points.pop()
            redraw()

    cv2.destroyAllWindows()

    if len(points) != n:
        raise SystemExit(f"Expected {n} points, got {len(points)}. Run again.")

    polygon = np.array(points)
    print()
    print("--- Paste into supervision_test.py ---")
    print(f"# Frame size: {w} x {h}")
    print("polygon = np.array([")
    for pt in polygon:
        print(f"    [{pt[0]}, {pt[1]}],")
    print("])")
    print("---")


if __name__ == "__main__":
    main()
