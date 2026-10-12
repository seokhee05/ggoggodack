"""
05_run_inference.py
추론 파이프라인 v3.1

개선:
  - side 오탐 방지: side 판정 기준 강화 (conf 0.6 이상 + area 5% 이상 + aspect ratio 체크)
  - 애매하면 inside/top으로 폴백
  - 사이클 상태 머신 인터페이스 준비 (IDLE/INSPECTING)
"""

import cv2, os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
import argparse

# ======================================================================
# 설정
# ======================================================================
CONF_TH = 0.4           # inside/top 최소 확신도
SIDE_CONF_TH = 0.6      # side는 더 높은 확신도 요구
SIDE_AREA_TH = 0.05     # side는 더 큰 면적 요구
AREA_TH = 0.03          # inside/top 최소 면적
MERGE_GAP = 0.5
PARTS_PER_BOX = 3

# side 형태 제약: aspect ratio가 1.5 이상이어야 (길쭉해야) side로 인정
SIDE_MIN_ASPECT = 1.3

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

# ======================================================================
# 한글 폰트 (시스템에 맞게 수정 필요)
# ======================================================================
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",       # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",                    # macOS
    "/Library/Fonts/NanumGothic.ttf",                                # macOS
    "C:/Windows/Fonts/malgun.ttf",                                   # Windows
]
_fc = {}

def _find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

_FONT_PATH = _find_font()
_FONT_BOLD = None
for p in FONT_CANDIDATES:
    if "Bold" in p and os.path.exists(p):
        _FONT_BOLD = p
        break
if _FONT_BOLD is None:
    _FONT_BOLD = _FONT_PATH

def _font(sz, bold=False):
    if _FONT_PATH is None:
        return ImageFont.load_default()
    k = (sz, bold)
    if k not in _fc:
        path = _FONT_BOLD if bold else _FONT_PATH
        _fc[k] = ImageFont.truetype(path, sz)
    return _fc[k]

# ======================================================================
# StepTracker v3.1
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

    def _accum(self, view, t):
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
        if view == "side":
            self.acc_side += self._accum("side", t)
            return "accepted"
        elif view == "inside":
            self.acc_inside += self._accum("inside", t)
            return "accepted"
        elif view == "top":
            if not self.step1_done():
                return "detected"  # 마스크 표시는 하되 누적 보류
            self.acc_top += self._accum("top", t)
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
# 인식 + side 엄격 판정
# ======================================================================
def detect_frame(model, frame):
    results = model(frame, conf=0.35, verbose=False)[0]
    H, W = frame.shape[:2]
    if results.boxes is None or len(results.boxes) == 0:
        return "none", 0.0, 0.0, None

    idx = int(results.boxes.conf.argmax())
    cls_id = int(results.boxes.cls[idx])
    conf = float(results.boxes.conf[idx])
    view = model.names.get(cls_id, "none")

    mask_overlay = None
    area_ratio = 0.0
    aspect_ratio = 1.0

    if results.masks is not None and len(results.masks) > idx:
        mask = results.masks.data[idx].cpu().numpy()
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        area_ratio = mask.sum() / (W * H)
        mask_overlay = (mask > 0.5).astype(np.uint8)

        # aspect ratio 계산 (side 검증용)
        cnts, _ = cv2.findContours(mask_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w, h) / (min(w, h) or 1)

    # ---- side 엄격 필터 ----
    if view == "side":
        side_ok = (
            conf >= SIDE_CONF_TH and           # 확신도 높아야
            area_ratio >= SIDE_AREA_TH and      # 면적 충분해야
            aspect_ratio >= SIDE_MIN_ASPECT     # 길쭉해야 (각진 옆면)
        )
        if not side_ok:
            # side 기준 미달 → 두 번째 후보 확인, 없으면 conf 기반 폴백
            # 모델이 side라고 했지만 기준 미달이면 inside/top 중 더 가능성 높은 쪽으로
            if results.boxes is not None and len(results.boxes) > 1:
                # 두 번째 후보
                confs = results.boxes.conf.cpu().numpy()
                sorted_idx = np.argsort(-confs)
                for si in sorted_idx[1:]:
                    alt_view = model.names.get(int(results.boxes.cls[si]), "none")
                    alt_conf = float(confs[si])
                    if alt_view in ("inside", "top") and alt_conf >= CONF_TH:
                        view = alt_view
                        conf = alt_conf
                        break
                else:
                    view = "none"
                    conf = 0.0
            else:
                view = "none"
                conf = 0.0

    return view, conf, area_ratio, mask_overlay

# ======================================================================
# 렌더링
# ======================================================================
def render(frame, W, H, pw, tracker, view, conf, area_ratio,
           mask_overlay, update_result, completed_parts, warnings_log):
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)

    disp = frame.copy()
    if mask_overlay is not None and view in MASK_COLORS:
        color = MASK_COLORS[view]
        disp[mask_overlay > 0] = (disp[mask_overlay > 0] * 0.45 + color * 0.55).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, tuple(int(x) for x in color), 3)

    canvas[:, :W] = disp
    canvas[:, W:] = (32, 32, 32)

    vlabel, vcolor = VIEW_DISPLAY.get(view, VIEW_DISPLAY["none"])
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

    if update_result == "detected":
        txt((bx, 82), "⏸ 1단계 완료 후 반영", 16, (255, 200, 100), True)

    txt((W - 285, 30), f"검수 완료  {completed_parts} / {PARTS_PER_BOX}", 26, (0, 255, 180), True)

    by = H - bar_h + 12
    cc = (100, 255, 150) if conf >= CONF_TH else (200, 200, 200)
    txt((20, by), f"확신도: {conf:.0%}", 24, cc, True)
    txt((220, by), f"면적: {area_ratio * 100:.1f}%", 24, (200, 220, 255), True)
    status_txt = {"accepted": "✓ 반영", "detected": "감지됨 (보류)", "ignored": "-"}
    txt((420, by), status_txt.get(update_result, "-"), 20, (255, 200, 100))
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
def run(video_path, weights_path, out_path=None):
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pw = 380

    if out_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f"{base}_result.mp4"

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + pw, H))
    tracker = StepTracker(fps=fps)
    completed = 0
    warnings_log = []
    absent = 0
    ABSENT_RESET = int(fps * 5.0)  # 5초로 늘림 (오분할 방지)
    had_act = False
    fi = 0

    print(f"영상: {video_path} ({total} frames, {total/fps:.1f}s)")
    print(f"모델: {weights_path}")
    print(f"출력: {out_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        view, conf, area, mask = detect_frame(model, frame)

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

        if had_act and absent >= ABSENT_RESET:
            missing = tracker.missing_steps()
            if missing:
                warnings_log.append((completed + 1, missing))
            completed += 1
            tracker = StepTracker(fps=fps)
            had_act = False
            absent = 0

        canvas = render(frame, W, H, pw, tracker, view, conf, area, mask,
                        result, completed, warnings_log)
        writer.write(canvas)
        fi += 1
        if fi % 300 == 0:
            print(f"  {fi}/{total} ({100 * fi / total:.0f}%)")

    if had_act:
        missing = tracker.missing_steps()
        if missing:
            warnings_log.append((completed + 1, missing))
        completed += 1

    cap.release()
    writer.release()
    print(f"\n완료: {completed}개 부품 검수, 경고 {len(warnings_log)}건")
    for p, ms in warnings_log:
        print(f"  부품#{p}: {ms}")
    return out_path


if __name__ == "__main__":
    # ↓↓↓ 여기 경로만 직접 수정해서 실행 버튼으로 바로 돌리기 ↓↓↓
    VIDEO_PATH = "/Users/songminseo/Desktop/kkokkodack/data/media/cycle_mp4/crop017.mp4"
    WEIGHTS_PATH = "/Users/songminseo/Desktop/kkokkodack/research/weights/armrest_3class_best.pt"
    OUTPUT_PATH = None  # None이면 자동으로 crop017_result.mp4 로 저장됨

    run(VIDEO_PATH, WEIGHTS_PATH, OUTPUT_PATH)
