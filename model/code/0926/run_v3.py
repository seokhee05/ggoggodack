"""
========================================================================
Armrest 불량 검사 - 검사원 행동 추적 파이프라인 v3.3
========================================================================
v3.3 변경사항:
  - 클래스별 표시 임계값 분리: side는 0.8, inside/top은 0.7
    (side가 오검출이 잦아서 더 엄격하게 요구)
  - 1단계 순서 강제: "안쪽면 먼저 완료 → 그 다음에만 옆면이 카운트됨"
    (옆면이 안쪽면보다 먼저 감지돼도 안쪽면 끝나기 전까지는 무시)
  - 중앙 배지를 매 프레임 "OO 감지"가 아니라 1단계 진행 단계로 표시:
    대기 중 → 안쪽면 확인 중 → 옆면 확인 중 → 1단계 완료
========================================================================
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from tqdm import tqdm

# ======================================================================
# 디바이스 자동 감지 (M4 Pro는 mps로 잡힘)
# ======================================================================
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
print(f"[INFO] 사용 디바이스: {DEVICE}")

# ======================================================================
# 모델 (3-class)
# ======================================================================
MODEL_PATH = "/Users/songminseo/Desktop/kkokkodack/model/code/armrest_3class_best.pt"
model = YOLO(MODEL_PATH)
model.to(DEVICE)
CLASS_NAMES = model.names  # {0: 'inside', 1: 'top', 2: 'side'}

# ======================================================================
# 체크리스트 / 임계값
# ======================================================================
CONF_TH = 0.4             # 체크리스트에 "반영"할 최소 확신도 (내부 판정용, 기존 유지)
AREA_TH = 0.03
MERGE_GAP = 0.5
PARTS_PER_BOX = 3

# 클래스별 "화면에 표시"할 최소 확신도 — side는 오검출이 잦아 더 엄격하게
DISPLAY_CONF_TH = {
    "inside": 0.7,
    "top": 0.7,
    "side": 0.8,
}

# ======================================================================
# 한글 폰트
# ======================================================================
_FONT_PATH = "/Users/songminseo/Desktop/kkokkodack/fonts/NotoSansCJK-Regular.ttc"
_FONT_BOLD = "/Users/songminseo/Desktop/kkokkodack/fonts/NotoSansCJK-Bold.ttc"
_fc = {}
def _font(sz, bold=False):
    k = (sz, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(_FONT_BOLD if bold else _FONT_PATH, sz)
    return _fc[k]

# ======================================================================
# StepTracker v3.3 — 순서 강제: inside 완료 전까지 side는 카운트 안 됨
# ======================================================================
@dataclass
class StepTracker:
    fps: float
    acc_inside: float = 0.0
    acc_side: float = 0.0
    acc_top: float = 0.0
    _last_t: dict = field(default_factory=lambda: {"side": None, "inside": None, "top": None})

    MIN_DUR_INSIDE = 0.3   # 윗홈(안쪽면) 확인 — 먼저 끝나야 함
    MIN_DUR_SIDE = 0.3     # 아래홈(옆면) 확인 — inside 완료 후에만 누적 시작
    MIN_DUR_TOP = 0.4      # 윗면 확인 (4단계)

    def _accumulate(self, view, t):
        last = self._last_t[view]
        delta = 0.0
        if last is not None and (t - last) <= MERGE_GAP:
            delta = t - last
        self._last_t[view] = t
        return delta

    def step1_inside_done(self):
        return self.acc_inside >= self.MIN_DUR_INSIDE

    def step1_side_done(self):
        return self.acc_side >= self.MIN_DUR_SIDE

    def step1_done(self):
        """1단계 완료 = 안쪽면이 먼저 끝나고, 그 다음 옆면도 끝남 (순서 강제)"""
        return self.step1_inside_done() and self.step1_side_done()

    def step4_done(self):
        return self.acc_top >= self.MIN_DUR_TOP

    def phase(self):
        """현재 1단계 진행 단계 — 중앙 배지 표시용"""
        if self.step1_done():
            return "STEP1_DONE"
        if self.step1_inside_done():
            return "WAIT_SIDE"      # 안쪽면 끝남, 옆면 확인 중/대기
        if self.acc_inside > 0:
            return "WAIT_INSIDE"    # 안쪽면 확인 진행 중
        return "IDLE"               # 아직 아무것도 감지 안 됨 (대기)

    def update(self, view, t):
        """
        반환값:
          'accepted'        → 체크리스트에 누적됨
          'waiting_inside'  → side 감지됐지만 inside가 아직 안 끝나서 무시(순서 위반)
          'detected'        → top 감지됐지만 1단계 미완료라 누적 보류
          'ignored'         → 해당 없음
        """
        if view == "inside":
            self.acc_inside += self._accumulate("inside", t)
            return "accepted"

        elif view == "side":
            if not self.step1_inside_done():
                return "waiting_inside"
            self.acc_side += self._accumulate("side", t)
            return "accepted"

        elif view == "top":
            if not self.step1_done():
                return "detected"
            self.acc_top += self._accumulate("top", t)
            return "accepted"

        return "ignored"

    def missing_steps(self):
        missing = []
        if not self.step1_done():
            missing.append("1단계: 안쪽면 기스·파손 확인")
        if not self.step4_done():
            missing.append("4단계: 비닐 제거 후 윗면 확인")
        return missing

# ======================================================================
# 인식
# ======================================================================
MASK_COLORS = {
    "inside": np.array([0, 220, 80]),
    "top": np.array([80, 180, 255]),
    "side": np.array([255, 180, 0]),
}

PHASE_DISPLAY = {
    "IDLE": ("대기 중", (120, 120, 120)),
    "WAIT_INSIDE": ("안쪽면 확인 중", (0, 220, 80)),
    "WAIT_SIDE": ("옆면 확인 중", (255, 180, 0)),
    "STEP1_DONE": ("1단계 완료 ✓", (0, 200, 100)),
}

def detect_frame(frame):
    results = model(frame, conf=0.35, verbose=False, device=DEVICE)[0]
    H, W = frame.shape[:2]
    if results.boxes is None or len(results.boxes) == 0:
        return "none", 0.0, 0.0, None
    idx = int(results.boxes.conf.argmax())
    cls_id = int(results.boxes.cls[idx])
    conf = float(results.boxes.conf[idx])
    view = CLASS_NAMES.get(cls_id, "none")
    mask_overlay = None
    area_ratio = 0.0
    if results.masks is not None and len(results.masks) > idx:
        mask = results.masks.data[idx].cpu().numpy()
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        area_ratio = mask.sum() / (W * H)
        mask_overlay = (mask > 0.5).astype(np.uint8)
    return view, conf, area_ratio, mask_overlay


def passes_display_threshold(view, conf):
    th = DISPLAY_CONF_TH.get(view, 0.7)
    return conf >= th

# ======================================================================
# 렌더링
# ======================================================================
def render(frame, W, H, pw, tracker, view, conf, area_ratio,
           mask_overlay, update_result, completed_parts, warnings_log,
           show_on_screen):
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)
    disp = frame.copy()

    if show_on_screen and mask_overlay is not None and view in MASK_COLORS:
        color = MASK_COLORS[view]
        disp[mask_overlay > 0] = (disp[mask_overlay > 0] * 0.45 + color * 0.55).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, tuple(int(x) for x in color), 3)

    canvas[:, :W] = disp
    canvas[:, W:] = (32, 32, 32)

    phase = tracker.phase()
    vlabel, vcolor = PHASE_DISPLAY[phase]
    bw2, bh2 = 400, 60
    bx = (W - bw2) // 2
    cv2.rectangle(canvas, (bx, 18), (bx + bw2, 18 + bh2), vcolor, -1)
    cv2.rectangle(canvas, (bx, 18), (bx + bw2, 18 + bh2), (255, 255, 255), 2)
    cv2.rectangle(canvas, (W - 300, 20), (W - 20, 72), (0, 0, 0), -1)

    bar_h = 90
    cv2.rectangle(canvas, (0, H - bar_h), (W, H), (20, 20, 20), -1)

    checks = [
        ("1-a", "윗홈 확인 (안쪽면)", tracker.step1_inside_done()),
        ("1-b", "아래홈 확인 (옆면)", tracker.step1_side_done()),
        ("4", "윗면 확인", tracker.step4_done()),
    ]
    y = 120
    for _, _, done in checks:
        y += 70
        col = (0, 200, 100) if done else (90, 90, 90)
        bx2 = W + 28
        cv2.rectangle(canvas, (bx2, y - 14), (bx2 + 30, y + 14), col, -1)
        cv2.rectangle(canvas, (bx2, y - 14), (bx2 + 30, y + 14), (200, 200, 200), 2)
        if done:
            cv2.line(canvas, (bx2 + 5, y), (bx2 + 12, y + 9), (255, 255, 255), 3)
            cv2.line(canvas, (bx2 + 12, y + 9), (bx2 + 24, y - 7), (255, 255, 255), 3)

    step1_y = 120 + 70 * 2 + 30
    step1_done = tracker.step1_done()
    cv2.rectangle(canvas, (W + 15, step1_y), (W + pw - 15, step1_y + 28),
                  (0, 140, 80) if step1_done else (60, 60, 60), -1)

    warn = None
    if warnings_log:
        lp, miss = warnings_log[-1]
        y0 = H - 210
        cv2.rectangle(canvas, (W + 12, y0), (W + pw - 12, y0 + 150), (0, 0, 160), -1)
        cv2.rectangle(canvas, (W + 12, y0), (W + pw - 12, y0 + 150), (0, 100, 255), 2)
        warn = (lp, miss, y0)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    def txt(xy, s, sz=20, c=(255, 255, 255), b=False):
        d.text(xy, s, font=_font(sz, b), fill=c)

    bbox = _font(26, True).getbbox(vlabel)
    tw = bbox[2] - bbox[0]
    tx = (W - tw) // 2
    txt((tx, 28), vlabel, 26, (255, 255, 255), True)

    if update_result == "waiting_inside":
        txt((bx, 82), "⚠ 안쪽면 먼저 확인 필요 (옆면 무시됨)", 16, (255, 150, 150), True)
    elif update_result == "detected":
        txt((bx, 82), "⏸ 1단계 완료 후 반영", 16, (255, 200, 100), True)

    txt((W - 285, 30), f"검수 완료  {completed_parts} / {PARTS_PER_BOX}", 26, (0, 255, 180), True)

    by = H - bar_h + 10
    disp_conf = conf if show_on_screen else 0.0
    th_for_view = DISPLAY_CONF_TH.get(view, 0.7)
    cc = (100, 255, 150) if disp_conf >= th_for_view else (200, 200, 200)
    raw_label = view if view != "none" else "-"
    txt((20, by), f"[디버그] raw감지: {raw_label}  확신도: {conf:.0%} (기준 {th_for_view:.0%})", 18, cc)
    txt((20, by + 30), f"면적: {area_ratio * 100:.1f}%   반영결과: {update_result}", 16, (180, 200, 220))

    txt((W + 18, 88), "검사 체크리스트", 22, (255, 255, 255), True)

    y = 120
    for code, label, done in checks:
        y += 70
        sc = (100, 255, 150) if done else (180, 180, 180)
        txt((W + 70, y - 16), f"[{code}]", 16, (200, 200, 200), True)
        txt((W + 70, y + 2), label, 15, (220, 220, 220))
        txt((W + 70, y + 22), "✓ 완료" if done else "미확인", 13, sc, True)

    s1txt = "1단계 완료 ✓" if step1_done else "1단계 진행 중..."
    s1c = (150, 255, 200) if step1_done else (200, 200, 200)
    txt((W + 25, step1_y + 4), s1txt, 16, s1c, True)

    if warn:
        lp, miss, y0 = warn
        txt((W + 25, y0 + 12), "⚠ 검사 누락 경고", 20, (0, 255, 255), True)
        txt((W + 25, y0 + 46), f"부품 #{lp} 누락:", 15, (255, 255, 255))
        yy = y0 + 72
        for m in miss:
            txt((W + 30, yy), f"· {m}", 14, (200, 200, 255))
            yy += 24

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ======================================================================
# 메인
# ======================================================================
def run_on_video(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = 420
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + pw, H))

    tracker = StepTracker(fps=fps)
    completed = 0
    warnings_log = []
    absent = 0
    ABSENT_RESET = int(fps * 5.0)
    had_act = False
    fi = 0

    pbar = tqdm(total=total, desc=f"{video_path.split('/')[-1]}", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        view, conf, area, mask = detect_frame(frame)

        result = "ignored"
        if view in ("inside", "side", "top") and conf >= CONF_TH and area >= AREA_TH:
            result = tracker.update(view, t)
            if result in ("accepted", "detected", "waiting_inside"):
                had_act = True
                absent = 0
            else:
                absent += 1
        else:
            absent += 1

        show_on_screen = (
            view in ("inside", "side", "top")
            and passes_display_threshold(view, conf)
            and area >= AREA_TH
        )

        if had_act and absent >= ABSENT_RESET:
            missing = tracker.missing_steps()
            if missing:
                warnings_log.append((completed + 1, missing))
            completed += 1
            tracker = StepTracker(fps=fps)
            had_act = False
            absent = 0

        canvas = render(frame, W, H, pw, tracker, view, conf, area, mask,
                        result, completed, warnings_log, show_on_screen)
        writer.write(canvas)
        fi += 1
        pbar.update(1)

    pbar.close()

    if had_act:
        missing = tracker.missing_steps()
        if missing:
            warnings_log.append((completed + 1, missing))
        completed += 1

    cap.release()
    writer.release()
    print(f"완료: {completed}개 부품, 경고 {len(warnings_log)}건")
    for p, ms in warnings_log:
        print(f"  부품#{p}: {ms}")
    return completed, warnings_log


def run_on_frames(frame_dir, out_path, fps=5):
    import glob
    frames = sorted(glob.glob(f"{frame_dir}/*.png"))
    if not frames:
        print("프레임 없음"); return

    sample = cv2.imread(frames[0])
    H, W = sample.shape[:2]
    pw = 420
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + pw, H))

    tracker = StepTracker(fps=fps)
    completed = 0
    warnings_log = []
    absent = 0
    ABSENT_RESET = int(fps * 5.0)
    had_act = False

    for i, fp in enumerate(tqdm(frames, desc="frames")):
        t = i / fps
        frame = cv2.imread(fp)
        view, conf, area, mask = detect_frame(frame)

        result = "ignored"
        if view in ("inside", "side", "top") and conf >= CONF_TH and area >= AREA_TH:
            result = tracker.update(view, t)
            if result in ("accepted", "detected", "waiting_inside"):
                had_act = True
                absent = 0
            else:
                absent += 1
        else:
            absent += 1

        show_on_screen = (
            view in ("inside", "side", "top")
            and passes_display_threshold(view, conf)
            and area >= AREA_TH
        )

        if had_act and absent >= ABSENT_RESET:
            missing = tracker.missing_steps()
            if missing:
                warnings_log.append((completed + 1, missing))
            completed += 1
            tracker = StepTracker(fps=fps)
            had_act = False
            absent = 0

        canvas = render(frame, W, H, pw, tracker, view, conf, area, mask,
                        result, completed, warnings_log, show_on_screen)
        writer.write(canvas)

    if had_act:
        missing = tracker.missing_steps()
        if missing:
            warnings_log.append((completed + 1, missing))
        completed += 1

    writer.release()
    print(f"완료: {completed}개 부품, 경고 {len(warnings_log)}건")
    for p, ms in warnings_log:
        print(f"  부품#{p}: {ms}")


if __name__ == "__main__":
    import sys, subprocess
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/songminseo/Desktop/kkokkodack/data/media/cycle_mp4/crop017.mp4"
    out = "/Users/songminseo/Desktop/kkokkodack/model/result_v3.mp4"

    if src.endswith(".mp4") or src.endswith(".mov"):
        run_on_video(src, out)
    else:
        run_on_frames(src, out)

    subprocess.run(["open", out])