"""
========================================================================
Armrest 불량 검사 - 검사원 행동 추적 파이프라인 v3.2
========================================================================
v3.2 변경사항:
  - 확신도 70% 미만은 화면에 표시 안 함 (DISPLAY_CONF_TH)
    ※ 체크리스트 반영 기준(CONF_TH=0.4)과는 별개. 반영 기준을 올리면
      실제 검사 순간도 놓칠 수 있어서, "화면에 보여줄지"만 따로 분리함.
  - GPU 자동 감지: Apple Silicon(MPS) > CUDA > CPU 순으로 자동 선택
  - tqdm으로 진행률 / 예상 남은 시간 표시
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
# 모델 (3-class) — 경로는 본인 환경에 맞게 수정
# ======================================================================
MODEL_PATH = "/Users/songminseo/Desktop/kkokkodack/model/code/armrest_3class_best.pt"
model = YOLO(MODEL_PATH)
model.to(DEVICE)
CLASS_NAMES = model.names  # {0: 'inside', 1: 'top', 2: 'side'}

# ======================================================================
# 체크리스트 / 임계값
# ======================================================================
CONF_TH = 0.4            # 체크리스트에 "반영"할 최소 확신도 (실제 판정 기준)
DISPLAY_CONF_TH = 0.7    # 화면에 "표시"할 최소 확신도 (오탐 노출 방지용, 더 엄격)
AREA_TH = 0.03
MERGE_GAP = 0.5
PARTS_PER_BOX = 3

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
# StepTracker
# ======================================================================
@dataclass
class StepTracker:
    fps: float
    acc_side: float = 0.0
    acc_inside: float = 0.0
    acc_top: float = 0.0
    _last_t: dict = field(default_factory=lambda: {"side": None, "inside": None, "top": None})

    MIN_DUR_SIDE = 0.3
    MIN_DUR_INSIDE = 0.3
    MIN_DUR_TOP = 0.4

    def _accumulate(self, view, t):
        last = self._last_t[view]
        delta = 0.0
        if last is not None and (t - last) <= MERGE_GAP:
            delta = t - last
        self._last_t[view] = t
        return delta

    def step1_side_done(self): return self.acc_side >= self.MIN_DUR_SIDE
    def step1_inside_done(self): return self.acc_inside >= self.MIN_DUR_INSIDE
    def step1_done(self): return self.step1_side_done() and self.step1_inside_done()
    def step4_done(self): return self.acc_top >= self.MIN_DUR_TOP

    def update(self, view, t):
        """
        반환값: 'accepted' / 'detected'(순서 보류) / 'ignored'
        ※ 이 함수는 CONF_TH(반영 기준)를 통과한 프레임에서만 호출됨.
          화면 표시 여부(DISPLAY_CONF_TH)와는 무관하게 체크리스트는 그대로 쌓임.
        """
        if view == "side":
            self.acc_side += self._accumulate("side", t)
            return "accepted"
        elif view == "inside":
            self.acc_inside += self._accumulate("inside", t)
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
VIEW_DISPLAY = {
    "inside": ("안쪽면 감지 (윗홈)", (0, 220, 80)),
    "top": ("윗면 감지", (80, 180, 255)),
    "side": ("옆면 감지 (아래홈)", (255, 180, 0)),
    "none": ("미감지", (120, 120, 120)),
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

# ======================================================================
# 렌더링
# ======================================================================
def render(frame, W, H, pw, tracker, view, conf, area_ratio,
           mask_overlay, update_result, completed_parts, warnings_log,
           show_on_screen):
    """
    show_on_screen: DISPLAY_CONF_TH를 통과했는지 여부.
      False면 마스크/배지를 그리지 않고 "미감지"로 표시 (체크리스트 반영과는 무관).
    """
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)
    disp = frame.copy()

    # 화면 표시 기준을 통과한 경우에만 마스크 오버레이
    disp_view = view if show_on_screen else "none"

    if show_on_screen and mask_overlay is not None and view in MASK_COLORS:
        color = MASK_COLORS[view]
        disp[mask_overlay > 0] = (disp[mask_overlay > 0] * 0.45 + color * 0.55).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, tuple(int(x) for x in color), 3)

    canvas[:, :W] = disp
    canvas[:, W:] = (32, 32, 32)

    vlabel, vcolor = VIEW_DISPLAY.get(disp_view, VIEW_DISPLAY["none"])
    bw2, bh2 = 400, 60
    bx = (W - bw2) // 2
    cv2.rectangle(canvas, (bx, 18), (bx + bw2, 18 + bh2), vcolor, -1)
    cv2.rectangle(canvas, (bx, 18), (bx + bw2, 18 + bh2), (255, 255, 255), 2)
    cv2.rectangle(canvas, (W - 300, 20), (W - 20, 72), (0, 0, 0), -1)

    bar_h = 80
    cv2.rectangle(canvas, (0, H - bar_h), (W, H), (20, 20, 20), -1)

    checks = [
        ("1-a", "아래홈 확인 (옆면)", tracker.step1_side_done()),
        ("1-b", "윗홈 확인 (안쪽면)", tracker.step1_inside_done()),
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

    if show_on_screen and update_result == "detected":
        txt((bx, 82), "⏸ 1단계 완료 후 반영", 16, (255, 200, 100), True)

    txt((W - 285, 30), f"검수 완료  {completed_parts} / {PARTS_PER_BOX}", 26, (0, 255, 180), True)

    by = H - bar_h + 12
    disp_conf = conf if show_on_screen else 0.0
    cc = (100, 255, 150) if disp_conf >= DISPLAY_CONF_TH else (200, 200, 200)
    txt((20, by), f"확신도: {disp_conf:.0%}", 24, cc, True)
    txt((220, by), f"면적: {(area_ratio if show_on_screen else 0.0) * 100:.1f}%", 24, (200, 220, 255), True)
    status_txt = {"accepted": "✓ 반영", "detected": "감지됨 (보류)", "ignored": "-"}
    txt((420, by), status_txt.get(update_result, "-") if show_on_screen else "-", 20, (255, 200, 100))
    txt((20, by + 32), f"감지: {vlabel}", 18, (0, 255, 180), True)

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
    pw = 380
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

        # 체크리스트 반영 여부 (기존 기준 CONF_TH 그대로 유지)
        result = "ignored"
        if view in ("inside", "side", "top") and conf >= CONF_TH and area >= AREA_TH:
            result = tracker.update(view, t)
            if result in ("accepted", "detected"):
                had_act = True
                absent = 0
            else:
                absent += 1
        else:
            absent += 1

        # 화면 표시 여부 (더 엄격한 DISPLAY_CONF_TH)
        show_on_screen = (
            view in ("inside", "side", "top")
            and conf >= DISPLAY_CONF_TH
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
    pw = 380
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
            if result in ("accepted", "detected"):
                had_act = True
                absent = 0
            else:
                absent += 1
        else:
            absent += 1

        show_on_screen = (
            view in ("inside", "side", "top")
            and conf >= DISPLAY_CONF_TH
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

    # 처리 끝나면 자동으로 기본 플레이어로 열기
    subprocess.run(["open", out])