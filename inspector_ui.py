# -*- coding: utf-8 -*-
"""
검사원용 GUI (Tkinter) - 연속 녹화 + 실시간 판정 버전
- test.py 의 검사 로직/유틸을 재사용합니다.
- 검사 세션 전체를 '하나의 연속 영상'으로 촬영합니다.
- 바코드(스캐너) 입력 또는 Enter / '박스 완료' 버튼으로 박스 한 개(사이클)를 구분합니다.
- 바코드 찍는 순간 그 박스의 체크리스트 통과 여부로 성공/실패를 집계합니다.
- 세션을 새로 시작하면 카운트는 0부터 초기화됩니다. (관리자 ROI 설정은 유지)
- 좌측 상단에 총/성공/실패, 우측 상단에 날짜/시간을 화면과 영상에 함께 표시합니다.
실행:  python inspector_ui.py   (사전 설치: pip install pillow)
"""

import os
import json
import time
from datetime import datetime

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import test as core  # 기존 검사 로직 재사용 (import 시 실행되지 않음)


# =============================================================================
# 색상 / 폰트 테마
# =============================================================================
BG = "#111418"
CARD = "#1b2027"
FG = "#f2f5f8"
SUB = "#9aa4b2"
ACCENT = "#2f7cf6"
GREEN = "#28c76f"
RED = "#ea5455"
YELLOW = "#ff9f43"

FONT = "Malgun Gothic"          # 윈도우 기본 한글 폰트 (UI용)
DISPLAY_W = 960                 # 화면에 보여줄 영상 가로 크기(px)

KOR_LABELS = {
    "knob": "손잡이 (Knob)",
    "hinge": "힌지 (Hinge)",
    "screw_1": "나사 1 (Screw 1)",
    "screw_2": "나사 2 (Screw 2)",
    "damper_1": "댐퍼 1 (Damper 1)",
    "damper_2": "댐퍼 2 (Damper 2)",
    "rib_joint": "리브·이음부 (Rib/Joint)",
    "gap": "라이닝-스킨 간격 (Gap)",
}

WEEKDAYS_KR = ["월", "화", "수", "목", "금", "토", "일"]


def now_str():
    n = datetime.now()
    return f"{n:%Y-%m-%d} ({WEEKDAYS_KR[n.weekday()]}) {n:%H:%M:%S}"


# =============================================================================
# 작은 유틸
# =============================================================================
def cv2_to_tk(frame, target_w=DISPLAY_W):
    h, w = frame.shape[:2]
    scale = target_w / w
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), scale


def big_button(parent, text, command, bg=ACCENT, fg="white", w=18, h=2, size=20):
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
        font=(FONT, size, "bold"), relief="flat", bd=0,
        width=w, height=h, cursor="hand2",
    )


# =============================================================================
# 영상에 새겨넣는 HUD (한글 tofu 방지를 위해 영문/숫자로 표기)
# =============================================================================
def draw_hud(frame, total, ok, ng, box_no, step_idx, total_steps, current_id):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (430, 132), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"TOTAL {total}", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OK {ok}", (200, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, f"NG {ng}", (310, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 90, 240), 2, cv2.LINE_AA)

    cv2.putText(frame, f"BOX #{box_no}   step {step_idx}/{total_steps}",
                (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)
    if current_id:
        cv2.putText(frame, f"WAIT: {current_id}", (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)

    for i in range(total_steps):
        cx = 18 + i * 40
        color = (60, 220, 130) if i < step_idx else (90, 90, 90)
        cv2.circle(frame, (cx, 118), 8, color, -1, cv2.LINE_AA)

    # 우측 상단 날짜/시간 (숫자·기호라 cv2 로 잘 렌더링됨)
    stamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, stamp, (w - tw - 16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)


# =============================================================================
# 메인 애플리케이션
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("부품 검사 시스템")
        self.configure(bg=BG)
        self.geometry("1120x820")
        self.minsize(960, 700)

        self.cap = None
        self.writer = None
        self.after_id = None
        self.clock_id = None
        self.clock_label = None

        core.ensure_directories()

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)

        self.show_standby()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------------------------------------------- 공통
    def clear(self):
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.clock_id:
            self.after_cancel(self.clock_id)
            self.clock_id = None
        self.clock_label = None
        for seq in ("<Return>", "<space>", "<Key>"):
            self.unbind(seq)
        for w in self.container.winfo_children():
            w.destroy()

    def release_cam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.release_cam()
        if self.writer is not None:
            self.writer.release()
        self.destroy()

    def _tick_clock(self):
        if self.clock_label is not None:
            try:
                self.clock_label.config(text=now_str())
            except tk.TclError:
                return
            self.clock_id = self.after(1000, self._tick_clock)

    # ---------------------------------------------------------------- 대기화면
    def show_standby(self):
        self.clear()
        self.release_cam()

        # 우측 상단 시계
        self.clock_label = tk.Label(self.container, text=now_str(),
                                    font=(FONT, 14), bg=BG, fg=SUB)
        self.clock_label.place(relx=0.98, rely=0.03, anchor="ne")
        self._tick_clock()

        wrap = tk.Frame(self.container, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(wrap, text="부품 검사 시스템", font=(FONT, 40, "bold"),
                 bg=BG, fg=FG).pack(pady=(0, 6))
        tk.Label(wrap, text="Enter 또는 스페이스바를 누르면 검사가 시작됩니다",
                 font=(FONT, 16), bg=BG, fg=SUB).pack(pady=(0, 30))

        big_button(wrap, "▶  검사 시작", self.start_session,
                   bg=GREEN, w=22, h=3, size=26).pack(pady=10)

        tk.Label(wrap, text="새 검사를 시작하면 개수는 0부터 다시 셉니다",
                 font=(FONT, 13), bg=BG, fg=SUB).pack(pady=(24, 0))

        roi_ok = core.load_roi_config() is not None
        status = "부품 위치 설정 완료" if roi_ok else "⚠ 부품 위치 설정 필요 (관리자 설정)"
        tk.Label(wrap, text=status, font=(FONT, 12),
                 bg=BG, fg=(GREEN if roi_ok else YELLOW)).pack(pady=(20, 0))

        gear = tk.Button(self.container, text="⚙ 관리자 설정",
                         command=self.show_setup, bg=CARD, fg=SUB,
                         font=(FONT, 11), relief="flat", bd=0, cursor="hand2")
        gear.place(relx=0.98, rely=0.97, anchor="se")

        self.bind("<Return>", lambda e: self.start_session())
        self.bind("<space>", lambda e: self.start_session())

    # ---------------------------------------------------------------- 관리자 ROI 설정
    def show_setup(self):
        self.clear()
        self.release_cam()

        self.cap = core.open_camera()
        frame = None
        for _ in range(30):
            ok, cur = self.cap.read()
            if ok:
                frame = cur
        self.release_cam()
        if frame is None:
            messagebox.showerror("오류", "웹캠 영상을 읽을 수 없습니다.\n카메라 연결을 확인하세요.")
            self.show_standby()
            return

        self.setup_frame = cv2.flip(frame, 1)
        self.setup_h, self.setup_w = self.setup_frame.shape[:2]
        self.setup_keys = [k for k, _ in core.ROI_DEFINITIONS]
        self.setup_idx = 0
        self.setup_rois = {}

        top = tk.Frame(self.container, bg=BG)
        top.pack(fill="x", pady=10)
        self.setup_title = tk.Label(top, text="", font=(FONT, 22, "bold"), bg=BG, fg=FG)
        self.setup_title.pack()
        tk.Label(top, text="마우스로 부품 영역을 드래그해서 네모를 그리세요",
                 font=(FONT, 14), bg=BG, fg=SUB).pack(pady=(2, 0))

        photo, self.setup_scale = cv2_to_tk(self.setup_frame)
        self.setup_photo = photo
        self.canvas = tk.Canvas(self.container, width=photo.width(),
                                height=photo.height(), bg="black",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(pady=8)
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.bind("<Button-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self._rect = None
        self._x0 = self._y0 = 0

        btns = tk.Frame(self.container, bg=BG)
        btns.pack(pady=8)
        big_button(btns, "다시 그리기", self._redo_current,
                   bg=YELLOW, w=12, h=1, size=15).pack(side="left", padx=6)
        big_button(btns, "취소", self.show_standby,
                   bg=RED, w=10, h=1, size=15).pack(side="left", padx=6)

        self._update_setup_title()

    def _update_setup_title(self):
        key = self.setup_keys[self.setup_idx]
        self.setup_title.config(
            text=f"[{self.setup_idx + 1}/8]  {KOR_LABELS[key]} 위치를 지정하세요")

    def _drag_start(self, e):
        self._x0, self._y0 = e.x, e.y
        if self._rect:
            self.canvas.delete(self._rect)
        self._rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                  outline=YELLOW, width=3)

    def _drag_move(self, e):
        if self._rect:
            self.canvas.coords(self._rect, self._x0, self._y0, e.x, e.y)

    def _drag_end(self, e):
        x0, y0 = min(self._x0, e.x), min(self._y0, e.y)
        x1, y1 = max(self._x0, e.x), max(self._y0, e.y)
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            return
        s = self.setup_scale
        box = (x0 / s, y0 / s, (x1 - x0) / s, (y1 - y0) / s)
        key = self.setup_keys[self.setup_idx]
        self.setup_rois[key] = core.normalize_roi(box, self.setup_w, self.setup_h)

        self.setup_idx += 1
        if self.setup_idx >= len(self.setup_keys):
            core.save_roi_config(self.setup_rois)
            messagebox.showinfo("완료", "부품 위치 8곳 설정을 저장했습니다.")
            self.show_standby()
        else:
            self._update_setup_title()

    def _redo_current(self):
        if self._rect:
            self.canvas.delete(self._rect)
            self._rect = None

    # ---------------------------------------------------------------- 검사 세션 (연속)
    def start_session(self):
        rois = core.load_roi_config()
        if rois is None:
            messagebox.showwarning(
                "설정 필요", "먼저 관리자 설정에서 부품 8곳 위치를 지정하세요.")
            return
        self.rois = rois
        self.steps = core.CHECKLIST_STEPS
        self.total_steps = len(self.steps)

        self.clear()
        self.release_cam()
        try:
            self.cap = core.open_camera()
        except Exception as err:
            messagebox.showerror("오류", str(err))
            self.show_standby()
            return

        width, height, fps = core.camera_properties(self.cap)
        self.sess_wh = (width, height)
        self.sess_fps = fps if fps and fps > 1 else core.FALLBACK_FPS

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"session_{ts}"
        self.session_started_at = datetime.now().isoformat(timespec="seconds")
        self.video_path = os.path.join(core.RECORDINGS_DIR, f"{self.session_name}.mp4")
        self.writer = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            self.sess_fps, (width, height))

        # 세션 집계 상태 (매번 0 부터 초기화)
        self.total = 0
        self.ok = 0
        self.ng = 0
        self.box_no = 1
        self.cur_idx = 0
        self.dwell_start = None
        self.barcode_buf = ""
        self.box_records = []
        self.session_start = time.monotonic()
        self.box_start_t = 0.0
        self.last_ms = -1

        # MediaPipe (실시간용: 부하 낮춤)
        pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=core.POSE_MODEL),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
            min_pose_presence_confidence=0.5)
        hand_opt = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=core.HAND_MODEL),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5,
            min_hand_presence_confidence=0.5)
        self.pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt)
        self.hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(hand_opt)

        # ------- UI -------
        self.clock_label = tk.Label(self.container, text=now_str(),
                                    font=(FONT, 14), bg=BG, fg=SUB)
        self.clock_label.place(relx=0.98, rely=0.02, anchor="ne")
        self._tick_clock()

        top = tk.Frame(self.container, bg=BG)
        top.pack(fill="x", pady=(8, 2))
        self.hud_label = tk.Label(top, text="", font=(FONT, 18, "bold"), bg=BG, fg=FG)
        self.hud_label.pack()
        tk.Label(top,
                 text="박스 검사를 마치면 바코드를 찍거나 Enter(또는 '박스 완료')를 누르세요",
                 font=(FONT, 12), bg=BG, fg=SUB).pack(pady=(2, 0))

        self.video_label = tk.Label(self.container, bg="black")
        self.video_label.pack(pady=6)

        btns = tk.Frame(self.container, bg=BG)
        btns.pack(pady=6)
        big_button(btns, "✓  박스 완료 (다음)", self.commit_box,
                   bg=ACCENT, w=18, h=2, size=18).pack(side="left", padx=8)
        big_button(btns, "■  검사 종료", self.end_session,
                   bg=RED, w=14, h=2, size=18).pack(side="left", padx=8)

        self.bind("<Return>", lambda e: self.commit_box())
        self.bind("<Key>", self._on_key)

        self._update_hud_label()
        self._loop()

    def _on_key(self, e):
        if e.char and e.char.isprintable() and e.keysym != "Return":
            self.barcode_buf += e.char

    def _update_hud_label(self):
        self.hud_label.config(
            text=f"총 {self.total}   성공 {self.ok}   실패 {self.ng}      "
                 f"|   현재 박스 #{self.box_no}  진행 {self.cur_idx}/{self.total_steps}")

    def _loop(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.after_id = self.after(15, self._loop)
            return
        frame = cv2.flip(frame, 1)
        width, height = self.sess_wh

        ms = int((time.monotonic() - self.session_start) * 1000)
        if ms <= self.last_ms:
            ms = self.last_ms + 1
        self.last_ms = ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_res = self.pose_lm.detect_for_video(img, ms)
        hand_res = self.hand_lm.detect_for_video(img, ms)

        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks:
                core.draw_body(frame, lm)

        centers = []
        if hand_res.hand_landmarks:
            for lm in hand_res.hand_landmarks:
                core.draw_hand(frame, lm)
                c = core.hand_center(lm, width, height)
                centers.append(c)
                cv2.circle(frame, c, 8, (255, 255, 255), 2, cv2.LINE_AA)

        active = None
        if self.cur_idx < self.total_steps:
            active = self.steps[self.cur_idx]["roi"]
            inside = any(core.point_in_roi(c, self.rois[active], width, height)
                         for c in centers)
            now = time.monotonic()
            if inside:
                if self.dwell_start is None:
                    self.dwell_start = now
                elif now - self.dwell_start >= core.DWELL_SECONDS:
                    self.cur_idx += 1
                    self.dwell_start = None
                    self._update_hud_label()
            else:
                self.dwell_start = None

        core.draw_rois(frame, self.rois, active_roi=active)
        draw_hud(frame, self.total, self.ok, self.ng, self.box_no,
                 self.cur_idx, self.total_steps, active)

        self.writer.write(frame)

        photo, _ = cv2_to_tk(frame)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

        self.after_id = self.after(1, self._loop)

    def commit_box(self):
        """바코드/Enter/버튼 → 현재 박스 마감, 성공·실패 집계, 다음 박스 시작."""
        if self.cap is None:
            return
        passed = self.cur_idx >= self.total_steps
        status = "성공" if passed else "실패"
        self.total += 1
        if passed:
            self.ok += 1
        else:
            self.ng += 1

        barcode = self.barcode_buf.strip() or f"제품{self.total:03d}"
        now_t = time.monotonic() - self.session_start
        self.box_records.append({
            "box_no": self.box_no,
            "barcode": barcode,
            "status": status,
            "completed_steps": self.cur_idx,
            "total_steps": self.total_steps,
            "checked_at": now_str(),
            "start_sec": round(self.box_start_t, 2),
            "end_sec": round(now_t, 2),
        })

        self.box_no += 1
        self.cur_idx = 0
        self.dwell_start = None
        self.barcode_buf = ""
        self.box_start_t = now_t
        self._update_hud_label()

    def end_session(self):
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.release_cam()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        try:
            self.pose_lm.close()
            self.hand_lm.close()
        except Exception:
            pass

        summary = {
            "session_name": self.session_name,
            "video": self.video_path,
            "started_at": self.session_started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "total": self.total,
            "success": self.ok,
            "fail": self.ng,
            "boxes": self.box_records,
        }
        result_path = os.path.join(core.RESULTS_DIR, f"{self.session_name}_summary.json")
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        self.show_summary(summary)

    # ---------------------------------------------------------------- 종료 요약
    def show_summary(self, summary):
        self.clear()

        self.clock_label = tk.Label(self.container, text=now_str(),
                                    font=(FONT, 14), bg=BG, fg=SUB)
        self.clock_label.place(relx=0.98, rely=0.03, anchor="ne")
        self._tick_clock()

        wrap = tk.Frame(self.container, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="검사 종료", font=(FONT, 34, "bold"),
                 bg=BG, fg=FG).pack(pady=(30, 6))
        tk.Label(wrap,
                 text=f"총 {summary['total']}개   |   성공 {summary['success']}   |   실패 {summary['fail']}",
                 font=(FONT, 22, "bold"), bg=BG, fg=ACCENT).pack(pady=6)

        listf = tk.Frame(wrap, bg=BG)
        listf.pack(pady=14)
        for rec in summary["boxes"][-12:]:
            c = GREEN if rec["status"] == "성공" else RED
            row = tk.Frame(listf, bg=BG)
            row.pack(anchor="w", pady=1)
            tk.Label(row, text=f"박스 #{rec['box_no']}", font=(FONT, 13),
                     bg=BG, fg=FG, width=8, anchor="w").pack(side="left")
            tk.Label(row, text=rec["barcode"], font=(FONT, 13),
                     bg=BG, fg=SUB, width=14, anchor="w").pack(side="left")
            tk.Label(row, text=f"{rec['completed_steps']}/{rec['total_steps']}",
                     font=(FONT, 13), bg=BG, fg=SUB, width=8).pack(side="left")
            tk.Label(row, text=rec["status"], font=(FONT, 13, "bold"),
                     bg=BG, fg=c).pack(side="left")

        tk.Label(wrap, text=f"영상 저장: {os.path.basename(summary['video'])}",
                 font=(FONT, 11), bg=BG, fg=SUB).pack(pady=(6, 0))

        big_button(wrap, "▶  대기화면으로", self.show_standby,
                   bg=ACCENT, w=20, h=2, size=20).pack(pady=20)


if __name__ == "__main__":
    App().mainloop()
