"""이미 촬영한 검사 영상 분석 도구."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageDraw, ImageFont

from factory_app import HAND_MODEL, POSE_MODEL, RESULTS, ensure_model, safe_name
from inspection_engine import InspectionEngine, OFFICIAL_STEPS
from dataclasses import asdict
from armrest_tracker import ArmrestTracker, draw_armrest_box
from object_seg import best_grip, draw_box, draw_point
from motion_tracker import (
    WorkerMotionTracker, arm_metrics, draw_hand_skeleton, draw_motion_points,
    draw_pose_arms, merge_hand_sources,
)


def output_dir_metrics(barcode: str) -> Path:
    """프레임별 판정 지표를 남길 CSV 경로."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS / f"{safe_name(barcode)}_{stamp}_metrics.csv"


def unique_result_path(barcode: str) -> Path:
    path = RESULTS / f"{safe_name(barcode)}_saved_result.json"
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS / f"{safe_name(barcode)}_{stamp}_saved_result.json"
    return path


def korean_font(size: int):
    candidates = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def checklist_display(frame, engine: InspectionEngine, event: dict):
    """직원 기준영상의 제품별 반복 체크리스트를 표시한다."""
    video = cv2.resize(frame, (900, 506))
    canvas = np.full((620, 1280, 3), (16, 19, 24), dtype=np.uint8)
    canvas[:506, :900] = video
    panel = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    title_font = korean_font(25); item_font = korean_font(17); small_font = korean_font(14)
    total = len(OFFICIAL_STEPS)
    draw.text((925, 20), "직원 기준 작업 검사", font=title_font, fill=(245,247,250))
    draw.text((925, 57), f"완료: {engine.step_index}/{total}", font=item_font, fill=(60,210,130))
    pending_index = next((i for i, s in enumerate(engine.steps) if s.status != "PASS"), -1)
    # 17개 항목 전체를 한 화면에 억지로 넣지 않고 현재 단계 주변을 표시한다.
    start = max(0, min(pending_index - 2, total - 8)) if pending_index >= 0 else max(0, total-8)
    y = 100
    for index in range(start, min(total, start + 8)):
        definition = OFFICIAL_STEPS[index]
        status = engine.steps[index].status if index < len(engine.steps) else "WAIT"
        if status == "PASS":
            mark, color = "● 완료", (45,205,120)
        elif engine.active and index == pending_index:
            mark, color = "▶ 분석 중", (255,190,45)
        else:
            mark, color = "○ 탐색 중", (155,165,180)
        draw.text((925, y), mark, font=small_font, fill=color)
        # 번호는 mark에 이미 있으므로 긴 문구가 패널을 넘지 않도록 본문만 두 줄 처리한다.
        label = definition.label
        if len(label) > 24:
            split = 24
            draw.text((925, y+22), label[:split], font=small_font, fill=(235,238,242))
            draw.text((925, y+42), label[split:], font=small_font, fill=(235,238,242))
            y += 58
        else:
            draw.text((925, y+22), label, font=small_font, fill=(235,238,242))
            y += 48
    reason = event.get("reason", "")
    draw.text((20, 530), "현재 판정 근거", font=item_font, fill=(255,195,60))
    draw.text((20, 565), reason[:95], font=small_font, fill=(235,238,242))
    draw.text((960, 578), "ESC: 검사 종료 및 결과 저장", font=small_font, fill=(255,145,90))
    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


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
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.40,
        min_pose_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )
    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
    tracker = WorkerMotionTracker()
    product_tracker = ArmrestTracker()
    engine = InspectionEngine(nominal_seconds=70.0, warning_seconds=18.0)
    engine.start(barcode.strip(), 0.0)
    frame_no = 0

    # 임계값을 실측으로 조정할 수 있도록 프레임별 지표를 남긴다.
    metrics_path = output_dir_metrics(barcode.strip())
    metrics_file = metrics_path.open("w", encoding="utf-8", newline="")
    metrics_writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_time = frame_no / fps
        timestamp_ms = max(frame_no, int(video_time * 1000))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        points = tracker.detect(frame, merge_hand_sources(result, pose_result))
        product_box, product_confidence = product_tracker.detect(frame)
        # 제품 후보 추적은 시각적 보조로 사용한다. 비닐이나 손에 가려져 제품 상자가
        # 잠시 사라져도 공식 검사 행동 분석 자체는 계속한다.
        # 통과가 일어난 프레임을 그 단계 이름으로 기록해야 튜닝에 쓸 수 있다.
        # update() 뒤에 읽으면 이미 다음 단계로 넘어가 있다.
        step_before = (engine.current_definition.id
                       if engine.current_definition else "done")
        event = engine.update(frame, points, video_time, arm_metrics(pose_result),
                              best_grip(result))
        draw_pose_arms(frame, pose_result)
        draw_hand_skeleton(frame, result)
        draw_motion_points(frame, points)
        draw_armrest_box(frame, product_box, product_confidence)
        # 세그멘테이션 결과를 함께 표시해 어떤 근거로 통과했는지 눈으로 본다.
        tool = engine.recognizer.last_tool
        vinyl = engine.recognizer.last_vinyl
        if tool.found:
            draw_box(frame, tool.box,
                     f"TOOL {'CLEAR' if tool.clear else 'DIM'}", (60, 200, 255))
            draw_point(frame, tool.head, "HEAD", (60, 200, 255))
        if vinyl.found:
            draw_box(frame, vinyl.box,
                     f"VINYL x{vinyl.blobs}", (240, 160, 255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 108), (20, 20, 20), -1)
        cv2.putText(frame, f"SAVED VIDEO  {engine.step_index}/{len(OFFICIAL_STEPS)}  {video_time:.1f}s",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
        cv2.putText(frame, event.get("message", ""), (15, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (0, 80, 255) if event["event"] == "warning" else (50, 220, 120), 2)
        cv2.putText(frame, event.get("reason", "Tracking verified hands only"), (15, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (80, 210, 255), 1)
        row = {"time": round(video_time, 2),
               "step": engine.step_index,
               "step_id": step_before,
               "event": event["event"]}
        row.update({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in asdict(engine.recognizer.last_metrics).items()})
        if metrics_writer is None:
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=list(row))
            metrics_writer.writeheader()
        metrics_writer.writerow(row)

        display = checklist_display(frame, engine, event)
        cv2.imshow("Saved Video Inspection - ESC to stop", display)
        if event["event"] == "complete" or cv2.waitKey(max(1, int(1000 / fps))) == 27:
            break
        frame_no += 1

    end_time = frame_no / fps
    inspection = engine.finish(str(Path(video).resolve()), end_time,
                               forced=engine.step_index < len(OFFICIAL_STEPS))
    output = unique_result_path(barcode.strip())
    with output.open("w", encoding="utf-8") as file:
        json.dump(inspection.__dict__, file, ensure_ascii=False, indent=2)
    metrics_file.close()
    cap.release()
    landmarker.close()
    pose_landmarker.close()
    cv2.destroyAllWindows()
    summary = []
    for item in inspection.checklist:
        mark = "완료" if item["status"] == "PASS" else "누락"
        summary.append(f"[{mark}] {item['label']}")
    messagebox.showinfo("분석 완료",
                        f"최종 판정: {inspection.final_status}\n\n" + "\n".join(summary)
                        + f"\n\n결과: {output}"
                        + f"\n지표: {metrics_path}")
    root.destroy()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        messagebox.showerror("저장 영상 검사 오류", str(exc))
        sys.exit(1)
