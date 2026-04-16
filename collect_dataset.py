"""
collect_dataset.py
------------------
Run this script to capture facial images from your webcam.
Usage:
  python collect_dataset.py --label ME --count 100
  python collect_dataset.py --label NOT_ME --count 100

Controls:
  SPACE  → capture a frame
  A      → auto-capture mode (captures every 0.5s)
  Q      → quit
"""

import cv2
import os
import argparse
import time

DATASET_DIR = "dataset"

def collect(label: str, count: int) -> None:
    save_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    print(f"\n[INFO] Saving to: {save_dir}")
    print(f"[INFO] Already have {existing} images. Capturing {count} more.")
    print("[INFO] Press SPACE to capture | A for auto | Q to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    captured = 0
    auto_mode = False
    last_capture = 0

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        status = f"Label: {label} | Captured: {captured}/{count} | {'AUTO' if auto_mode else 'MANUAL'}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 200, 255), 2)
        cv2.imshow("Dataset Collector", display)

        key = cv2.waitKey(1) & 0xFF

        should_capture = False
        if key == ord(' '):
            should_capture = True
        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"[INFO] Auto mode {'ON' if auto_mode else 'OFF'}")
        elif key == ord('q'):
            break

        if auto_mode and (time.time() - last_capture) > 0.5:
            should_capture = True

        if should_capture and len(faces) > 0:
            idx = existing + captured
            fname = os.path.join(save_dir, f"{label}_{idx:04d}.jpg")
            cv2.imwrite(fname, frame)
            captured += 1
            last_capture = time.time()
            print(f"  Saved: {fname}")
        elif should_capture and len(faces) == 0:
            print("  [WARN] No face detected — move closer or improve lighting.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Captured {captured} images for label '{label}'.")
    print(f"       Total in folder: {existing + captured}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam dataset collector")
    parser.add_argument("--label", choices=["ME", "NOT_ME"], required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    collect(args.label, args.count)
