"""
학습된 YOLOv8n-seg 모델을 inspection_pipeline에 연결해서
crop008 전체 영상에 대해 실제 추론 + 체크리스트 판정 + 오버레이 렌더링
"""
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from enum import Enum

# ======================================================================
# 모델 로드
# ======================================================================
MODEL_PATH = "/home/claude/runs/armrest/weights/best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # {0: 'inside', 1: 'top'}

# ======================================================================
# 체크리스트 정의 (민서 담당 부분: inside/top만)
# ======================================================================
class Step(Enum):
    INSIDE = "1단계: 안쪽면 기스·파손 육안 확인"
    TOP = "4단계: 비닐 제거 후 상단(윗면) 확인"

STEP_MIN_DURATION = {Step.INSIDE: 0.4, Step.TOP: 0.4}
VIEW_TO_STEP = {"inside": Step.INSIDE, "top": Step.TOP}
CONF_TH = 0.4
AREA_TH = 0.03
MERGE_GAP_SEC = 0.5
PARTS_PER_BOX = 3

# ======================================================================
# 한글 렌더링
# ======================================================================
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_fc = {}
def _font(sz, bold=False):
    k=(sz,bold)
    if k not in _fc: _fc[k]=ImageFont.truetype(_FONT_BOLD if bold else _FONT_PATH, sz)
    return _fc[k]

# ======================================================================
# StepTracker
# ======================================================================
@dataclass
class StepTracker:
    fps: float
    accumulated: dict = field(default_factory=lambda: {s: 0.0 for s in Step})
    _last_active_t: dict = field(default_factory=lambda: {s: None for s in Step})

    def update(self, step, t):
        last = self._last_active_t[step]
        if last is not None and (t - last) <= MERGE_GAP_SEC:
            self.accumulated[step] += (t - last)
        self._last_active_t[step] = t

    def is_done(self, step):
        return self.accumulated[step] >= STEP_MIN_DURATION[step]

    def missing_steps(self):
        return [s for s in Step if not self.is_done(s)]

# ======================================================================
# 실제 모델 기반 인식
# ======================================================================
def detect_frame(frame):
    """학습된 YOLOv8-seg로 추론. 반환: (view, conf, area_ratio, mask_overlay)"""
    results = model(frame, conf=CONF_TH, verbose=False)[0]
    H, W = frame.shape[:2]

    if results.boxes is None or len(results.boxes) == 0:
        return "none", 0.0, 0.0, None

    # 가장 큰(confidence 높은) 검출 사용
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
def render(frame, W, H, pw, tracker, active_step, view, conf, area_ratio,
           mask_overlay, completed_parts, warnings_log):
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)

    # 마스크 오버레이 (초록 반투명)
    disp = frame.copy()
    if mask_overlay is not None:
        color = np.array([0, 220, 80]) if view == "inside" else np.array([80, 180, 255])
        disp[mask_overlay > 0] = (disp[mask_overlay > 0] * 0.5 + color * 0.5).astype(np.uint8)

    canvas[:, :W] = disp
    canvas[:, W:] = (32, 32, 32)

    # 도형
    cv2.rectangle(canvas, (W - 320, 20), (W - 20, 72), (0, 0, 0), -1)
    bar_h = 100
    cv2.rectangle(canvas, (0, H - bar_h), (W, H), (0, 0, 0), -1)

    step_geo = []
    y = 140
    for step in Step:
        y += 80
        done = tracker.is_done(step)
        acc = tracker.accumulated[step]
        need = STEP_MIN_DURATION[step]
        color = (120, 220, 0) if done else (220, 80, 80)
        cv2.circle(canvas, (W + 38, y - 6), 14, color, -1)
        bx, bw = W + 62, pw - 95
        cv2.rectangle(canvas, (bx, y + 22), (bx + bw, y + 32), (70, 70, 70), -1)
        fw = int(min(1.0, acc / need) * bw)
        cv2.rectangle(canvas, (bx, y + 22), (bx + fw, y + 32), color, -1)
        step_geo.append((step, y, done, acc, need))

    warn = None
    if warnings_log:
        lp, miss = warnings_log[-1]
        y0 = H - 230
        cv2.rectangle(canvas, (W + 15, y0), (W + pw - 15, y0 + 160), (0, 0, 140), -1)
        warn = (lp, miss, y0)

    # 텍스트 (PIL)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    def txt(xy, s, sz=20, c=(255,255,255), b=False):
        d.text(xy, s, font=_font(sz, b), fill=c)

    txt((W - 305, 30), f"검수 완료  {completed_parts} / {PARTS_PER_BOX}", 28, (0, 255, 180), True)

    txt((15, H - bar_h + 8), "판단 기준", 16, (180, 180, 180))
    txt((15, H - bar_h + 32), f"view={view}   conf={conf:.2f}   area={area_ratio*100:.1f}%", 18, (255, 255, 255))
    now = "-" if active_step is None else active_step.value
    txt((15, H - bar_h + 60), f"현재 감지: {now}", 18, (0, 255, 180), True)

    txt((W + 20, 100), "체크리스트 (현재 부품)", 22, (255, 255, 255), True)
    for step, y, done, acc, need in step_geo:
        mark = "완료" if done else "대기"
        txt((W + 28, y - 16), mark, 13, (0, 0, 0), True)
        txt((W + 62, y - 22), step.value, 16, (235, 235, 235))
        txt((W + 62, y + 36), f"{acc:.1f}s / {need:.1f}s", 12, (160, 160, 160))

    if warn:
        lp, miss, y0 = warn
        txt((W + 30, y0 + 12), "⚠ 검사 누락 경고", 22, (0, 255, 255), True)
        txt((W + 30, y0 + 50), f"부품 #{lp} 누락 항목:", 16, (255, 255, 255))
        yy = y0 + 80
        for m in miss:
            txt((W + 30, yy), f"· {m.value}", 14, (200, 200, 255))
            yy += 26

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ======================================================================
# 메인
# ======================================================================
def run_real(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = 460
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + pw, H))

    tracker = StepTracker(fps=fps)
    completed_parts = 0
    warnings_log = []
    absent_frames = 0
    ABSENT_RESET = int(fps * 2.5)
    part_had_activity = False
    fi = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps

        view, conf, area_ratio, mask_overlay = detect_frame(frame)
        active_step = None

        if view in VIEW_TO_STEP and conf >= CONF_TH and area_ratio >= AREA_TH:
            step = VIEW_TO_STEP[view]
            tracker.update(step, t)
            active_step = step
            part_had_activity = True
            absent_frames = 0
        else:
            absent_frames += 1

        if part_had_activity and absent_frames >= ABSENT_RESET:
            missing = tracker.missing_steps()
            if missing:
                warnings_log.append((completed_parts + 1, missing))
            completed_parts += 1
            tracker = StepTracker(fps=fps)
            part_had_activity = False
            absent_frames = 0

        canvas = render(frame, W, H, pw, tracker, active_step, view, conf,
                        area_ratio, mask_overlay, completed_parts, warnings_log)
        writer.write(canvas)
        fi += 1
        if fi % 100 == 0:
            print(f"  {fi}/{total} ({100*fi/total:.0f}%)")

    # 마지막 부품 마무리
    if part_had_activity:
        missing = tracker.missing_steps()
        if missing:
            warnings_log.append((completed_parts + 1, missing))
        completed_parts += 1

    cap.release()
    writer.release()
    return completed_parts, warnings_log


if __name__ == "__main__":
    src = "/mnt/user-data/uploads/crop008.mp4"
    out = "/home/claude/crop008_real.mp4"
    print("추론 시작...")
    parts, warns = run_real(src, out)
    print(f"검수 완료: {parts}")
    print(f"경고: {warns}")
