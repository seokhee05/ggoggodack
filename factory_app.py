"""공장용 암레스트 실시간 검사 애플리케이션.

실행 즉시 카메라를 표시하며 바코드를 스캔하면 박스별 녹화와 실시간
행동 검사를 시작한다. 수동 ROI 설정과 C/Q 키 조작은 사용하지 않는다.
"""

from __future__ import annotations

import json
import os
import re
import sys
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from inspection_engine import InspectionEngine, OFFICIAL_STEPS
from object_seg import best_grip, draw_box, draw_point
from motion_tracker import (
    WorkerMotionTracker, arm_metrics, draw_hand_skeleton, draw_motion_points,
    draw_pose_arms, merge_hand_sources,
)


ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
HAND_MODEL = MODELS / "hand_landmarker.task"
POSE_MODEL = MODELS / "pose_landmarker_lite.task"
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FALLBACK_FPS = 30.0
DISPLAY_WIDTH = 960

BG = "#101318"
CARD = "#1b2028"
TEXT = "#f4f6f8"
SUB = "#a7b0bc"
GREEN = "#21c77a"
RED = "#ef5350"
YELLOW = "#ffb020"
BLUE = "#3b82f6"
FONT = "Malgun Gothic"


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value.strip())
    return value[:80] or "UNKNOWN"


def ensure_model() -> None:
    RECORDINGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    if not (HAND_MODEL.exists() and HAND_MODEL.stat().st_size > 0):
        urllib.request.urlretrieve(HAND_URL, HAND_MODEL)
    if not (POSE_MODEL.exists() and POSE_MODEL.stat().st_size > 0):
        urllib.request.urlretrieve(POSE_URL, POSE_MODEL)


def alert_sound(window: tk.Tk) -> None:
    try:
        if sys.platform == "win32":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            window.bell()
    except Exception:
        pass


class FactoryApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("암레스트 실시간 검사 시스템")
        self.configure(bg=BG)
        self.geometry("1180x900")
        self.minsize(1020, 760)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.writer = None
        self.hand_lm = None
        self.pose_lm = None
        self.after_id = None
        self.last_timestamp_ms = -1
        self.engine = InspectionEngine(nominal_seconds=70.0, warning_seconds=18.0)
        self.current_video_path = ""
        self.current_result_path = ""
        self.total = self.passed = self.failed = 0
        self.last_event = "idle"
        self.motion_tracker = WorkerMotionTracker()

        self._build_ui()
        self.after(100, self._start_camera)

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=28, pady=(18, 8))
        tk.Label(header, text="암레스트 실시간 검사",
                 font=(FONT, 25, "bold"), bg=BG, fg=TEXT).pack(side="left")
        self.clock = tk.Label(header, text="", font=(FONT, 13), bg=BG, fg=SUB)
        self.clock.pack(side="right")

        stats = tk.Frame(self, bg=CARD, pady=10)
        stats.pack(fill="x", padx=28)
        self.stats_label = tk.Label(stats, text="", font=(FONT, 16, "bold"),
                                    bg=CARD, fg=TEXT)
        self.stats_label.pack()

        barcode = tk.Frame(self, bg=BG)
        barcode.pack(pady=10)
        tk.Label(barcode, text="바코드", font=(FONT, 15, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", padx=8)
        self.barcode_entry = tk.Entry(barcode, width=32, font=(FONT, 18),
                                      justify="center")
        self.barcode_entry.pack(side="left", padx=6)
        self.barcode_entry.bind("<Return>", self._on_barcode)
        tk.Button(barcode, text="검사 시작", font=(FONT, 14, "bold"),
                  bg=GREEN, fg="white", relief="flat", padx=20, pady=5,
                  command=self._on_barcode).pack(side="left", padx=8)

        self.video_label = tk.Label(self, bg="black")
        self.video_label.pack(padx=28, pady=4)

        self.status_label = tk.Label(self, text="카메라 연결 중...",
                                     font=(FONT, 17, "bold"), bg=CARD,
                                     fg=YELLOW, pady=10)
        self.status_label.pack(fill="x", padx=28, pady=(5, 5))

        self.check_labels = []
        checklist = tk.Frame(self, bg=BG)
        checklist.pack(fill="x", padx=34, pady=3)
        checklist.grid_columnconfigure(0, weight=1)
        checklist.grid_columnconfigure(1, weight=1)
        for index, step in enumerate(OFFICIAL_STEPS):
            label = tk.Label(checklist, text=f"○ {step.label}",
                             font=(FONT, 10), bg=BG, fg=SUB, anchor="w")
            label.grid(row=index // 2, column=index % 2, sticky="ew", padx=5, pady=1)
            self.check_labels.append(label)

        buttons = tk.Frame(self, bg=BG)
        buttons.pack(pady=8)
        tk.Button(buttons, text="현재 박스 강제 종료", font=(FONT, 12, "bold"),
                  bg=RED, fg="white", relief="flat", padx=18, pady=6,
                  command=self._force_finish).pack(side="left", padx=6)
        tk.Button(buttons, text="저장 영상 검사", font=(FONT, 12, "bold"),
                  bg=BLUE, fg="white", relief="flat", padx=18, pady=6,
                  command=self._open_saved_video).pack(side="left", padx=6)
        tk.Button(buttons, text="프로그램 종료", font=(FONT, 12, "bold"),
                  bg=CARD, fg=TEXT, relief="flat", padx=18, pady=6,
                  command=self.on_close).pack(side="left", padx=6)

        self._update_stats()
        self._tick_clock()

    def _tick_clock(self) -> None:
        self.clock.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def _start_camera(self) -> None:
        try:
            ensure_model()
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(CAMERA_INDEX, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not self.cap.isOpened():
                raise RuntimeError("카메라를 열 수 없습니다.")

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_tracking_confidence=0.45,
                min_hand_presence_confidence=0.45,
            )
            self.hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(options)
            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self.pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
            self._set_status("카메라 정상 — 바코드를 스캔해 주세요.", GREEN)
            self.barcode_entry.focus_set()
            self._loop()
        except Exception as exc:
            messagebox.showerror("시작 오류", str(exc))
            self._set_status(f"시작 오류: {exc}", RED)

    def _camera_info(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1 or fps > 120:
            fps = FALLBACK_FPS
        return width, height, fps

    def _unique_paths(self, barcode: str):
        name = safe_name(barcode)
        video = RECORDINGS / f"{name}.mp4"
        result = RESULTS / f"{name}_result.json"
        if video.exists() or result.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            video = RECORDINGS / f"{name}_{suffix}.mp4"
            result = RESULTS / f"{name}_{suffix}_result.json"
        return video, result

    def _on_barcode(self, event=None):
        barcode = self.barcode_entry.get().strip()
        self.barcode_entry.delete(0, "end")
        if not barcode:
            self._set_status("바코드를 입력해 주세요.", RED, True)
            return "break"
        if self.cap is None:
            self._set_status("카메라가 준비되지 않았습니다.", RED, True)
            return "break"
        if self.engine.active:
            if not messagebox.askyesno(
                    "다음 바코드",
                    "현재 박스를 NG로 저장하고 다음 박스 검사를 시작할까요?"):
                self.barcode_entry.focus_set()
                return "break"
            self._finish_box(forced=True)

        video_path, result_path = self._unique_paths(barcode)
        width, height, fps = self._camera_info()
        self.writer = cv2.VideoWriter(str(video_path),
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (width, height))
        if not self.writer.isOpened():
            self.writer = None
            self._set_status("영상 저장 파일을 만들 수 없습니다.", RED, True)
            return "break"
        self.current_video_path = str(video_path)
        self.current_result_path = str(result_path)
        self.motion_tracker.reset()
        self.engine.start(barcode, time.monotonic())
        self._set_status(f"[{barcode}] 검사 시작 — {OFFICIAL_STEPS[0].label}", GREEN)
        self._refresh_checklist()
        self._update_stats()
        self.barcode_entry.focus_set()
        return "break"

    def _draw_overlay(self, frame, event):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 96), (18, 18, 18), -1)
        product = self.engine.barcode if self.engine.active else "WAITING"
        step = self.engine.step_index
        cv2.putText(frame, f"PRODUCT: {product}", (16, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"CHECKLIST: {step}/{len(OFFICIAL_STEPS)}", (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 220, 130), 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (w - 290, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (230, 230, 230), 2)
        if event.get("event") == "warning":
            cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 170), -1)
            cv2.putText(frame, "WARNING: CHECK ORDER / MISSING STEP",
                        (16, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    def _loop(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.after_id = self.after(20, self._loop)
            return
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()
        ms = int(now * 1000)
        if ms <= self.last_timestamp_ms:
            ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self.hand_lm.detect_for_video(image, ms)
        pose_result = self.pose_lm.detect_for_video(image, ms)
        hands = self.motion_tracker.detect(
            frame, merge_hand_sources(hand_result, pose_result))

        # 판정은 오버레이가 없는 원본 프레임으로 한다. 스켈레톤 선 색이 공구
        # 색상 검출 범위와 겹쳐 오검출을 만들기 때문이다.
        event = self.engine.update(frame, hands, now, arm_metrics(pose_result),
                                   best_grip(hand_result))
        if self.engine.active and self.writer is not None:
            self.writer.write(frame)

        draw_pose_arms(frame, pose_result)
        draw_hand_skeleton(frame, hand_result)
        draw_motion_points(frame, hands)
        tool = self.engine.recognizer.last_tool
        vinyl = self.engine.recognizer.last_vinyl
        if tool.found:
            draw_box(frame, tool.box,
                     f"TOOL {'CLEAR' if tool.clear else 'DIM'}", (60, 200, 255))
            draw_point(frame, tool.head, "HEAD", (60, 200, 255))
        if vinyl.found:
            draw_box(frame, vinyl.box, f"VINYL x{vinyl.blobs}", (240, 160, 255))

        if event["event"] != self.last_event:
            if event["event"] == "warning":
                self._set_status(event["message"], RED, True)
            elif event["event"] in {"step_complete", "hit"}:
                message = event["message"]
                if event.get("next"):
                    message += f" → 다음: {event['next']}"
                self._set_status(message, GREEN)
                self._refresh_checklist()
            elif event["event"] == "complete":
                self._set_status(event["message"] + " 저장 중...", GREEN)
                self._finish_box(forced=False)
            self.last_event = event["event"]

        self._draw_overlay(frame, event)
        scale = DISPLAY_WIDTH / w
        display = cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale)))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(display))
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        self.after_id = self.after(1, self._loop)

    def _refresh_checklist(self) -> None:
        for i, label in enumerate(self.check_labels):
            if i < len(self.engine.steps) and self.engine.steps[i].status == "PASS":
                label.config(text=f"● {OFFICIAL_STEPS[i].label}  완료", fg=GREEN)
            elif self.engine.active and i == self.engine.step_index:
                label.config(text=f"▶ {OFFICIAL_STEPS[i].label}  진행 중", fg=YELLOW)
            else:
                label.config(text=f"○ {OFFICIAL_STEPS[i].label}", fg=SUB)

    def _finish_box(self, forced: bool) -> None:
        if not self.engine.active:
            return
        now = time.monotonic()
        result = self.engine.finish(self.current_video_path, now, forced)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        with open(self.current_result_path, "w", encoding="utf-8") as file:
            json.dump(result.__dict__, file, ensure_ascii=False, indent=2)
        self.total += 1
        if result.final_status == "PASS":
            self.passed += 1
            self._set_status(f"[{result.barcode}] PASS 저장 완료 — 다음 바코드 대기", GREEN)
        else:
            self.failed += 1
            missing = [x["label"] for x in result.checklist if x["status"] != "PASS"]
            self._set_status(
                f"[{result.barcode}] NG 저장 — 누락: {', '.join(missing)}", RED, True)
        self.current_video_path = ""
        self.current_result_path = ""
        self.last_event = "idle"
        self._refresh_checklist()
        self._update_stats()
        self.barcode_entry.focus_set()

    def _force_finish(self) -> None:
        if not self.engine.active:
            self._set_status("진행 중인 박스가 없습니다.", RED, True)
            return
        if messagebox.askyesno("강제 종료", "현재 박스를 NG로 저장할까요?"):
            self._finish_box(forced=True)

    def _open_saved_video(self) -> None:
        """실시간 카메라를 유지한 채 저장 영상 분석기를 별도 실행한다."""
        script = ROOT / "saved_video_check.py"
        subprocess.Popen([sys.executable, str(script)], cwd=str(ROOT))

    def _set_status(self, message: str, color: str, sound: bool = False) -> None:
        if sound:
            self.status_label.config(
                text=f"⚠ 경고 ⚠   {message}", fg="white", bg=RED,
                font=(FONT, 22, "bold"), pady=18)
            self.after(1800, self._restore_status_style)
        else:
            self.status_label.config(text=message, fg=color, bg=CARD,
                                     font=(FONT, 17, "bold"), pady=10)
        if sound:
            alert_sound(self)

    def _restore_status_style(self) -> None:
        self.status_label.config(bg=CARD, font=(FONT, 17, "bold"), pady=10)

    def _update_stats(self) -> None:
        current = self.engine.barcode if self.engine.active else "대기 중"
        self.stats_label.config(
            text=f"총 {self.total}   PASS {self.passed}   NG {self.failed}"
                 f"          현재 제품: {current}")

    def on_close(self) -> None:
        if self.engine.active and not messagebox.askyesno(
                "프로그램 종료", "현재 검사를 NG로 저장하고 종료할까요?"):
            return
        if self.engine.active:
            self._finish_box(forced=True)
        if self.after_id:
            self.after_cancel(self.after_id)
        if self.writer is not None:
            self.writer.release()
        if self.hand_lm is not None:
            self.hand_lm.close()
        if self.pose_lm is not None:
            self.pose_lm.close()
        if self.cap is not None:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    FactoryApp().mainloop()
