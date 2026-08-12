"""이미 촬영한 검사 영상 분석 도구."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from factory_app import HAND_MODEL, RESULTS, ensure_model, safe_name
from inspection_engine import InspectionEngine
from motion_tracker import WorkerMotionTracker, draw_motion_points


def unique_result_path(barcode: str) -> Path:
    path = RESULTS / f"{safe_name(barcode)}_saved_result.json"
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS / f"{safe_name(barcode)}_{stamp}_saved_result.json"
    return path


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    video = filedialog.askopenfilename(
        title="검사할 저장 영상을 선택하세요",
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    if not video:
        root.destroy()
        return
    barcode = simpledialog.askstring("제품 번호", "영상의 바코드/제품 번호를 입력하세요:")
    if not barcode:
        root.destroy()
        return

    ensure_model()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        messagebox.showerror("오류", "선택한 영상을 열 수 없습니다.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 30.0

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_tracking_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    tracker = WorkerMotionTracker()
    engine = InspectionEngine(nominal_seconds=70.0, warning_seconds=18.0)
    engine.start(barcode.strip(), 0.0)
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_time = frame_no / fps
        timestamp_ms = max(frame_no, int(video_time * 1000))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        centers = []
        for landmarks in result.hand_landmarks or []:
            ids = (0, 5, 9, 13, 17)
            centers.append((sum(landmarks[i].x for i in ids) / len(ids),
                            sum(landmarks[i].y for i in ids) / len(ids)))
        points = tracker.detect(frame, centers)
        event = engine.update(frame, points, video_time)
        draw_motion_points(frame, points)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 82), (20, 20, 20), -1)
        cv2.putText(frame, f"SAVED VIDEO  {engine.step_index}/7  {video_time:.1f}s",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
        cv2.putText(frame, event.get("message", ""), (15, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (0, 80, 255) if event["event"] == "warning" else (50, 220, 120), 2)
        cv2.imshow("Saved Video Inspection - ESC to stop", cv2.resize(frame, (960, 540)))
        if event["event"] == "complete" or cv2.waitKey(max(1, int(1000 / fps))) == 27:
            break
        frame_no += 1

    end_time = frame_no / fps
    inspection = engine.finish(str(Path(video).resolve()), end_time,
                               forced=engine.step_index < 7)
    output = unique_result_path(barcode.strip())
    with output.open("w", encoding="utf-8") as file:
        json.dump(inspection.__dict__, file, ensure_ascii=False, indent=2)
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    messagebox.showinfo("분석 완료", f"{inspection.final_status}\n결과: {output}")
    root.destroy()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        messagebox.showerror("저장 영상 검사 오류", str(exc))
        sys.exit(1)
