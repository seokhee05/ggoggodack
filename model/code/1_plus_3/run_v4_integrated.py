"""
========================================================================
Armrest 불량 검사 - 통합 파이프라인 v4 (덮개 + 1단계 + 3단계 + Pose)
========================================================================
담당 통합:
  - 덮개 열림/닫힘 (팀원 cover_check.CoverCheck)  → 사이클 시작/종료 트리거
  - 1단계 안쪽면/옆면 (내 담당, 3-class seg 모델)
  - 3단계 옆면 Hammer 타격 (팀원 hammer_check.HammerCheck)
  - MediaPipe Pose → 덮개 팔 동작 판정에 사용

핵심: 사이클 상태 머신으로 순서를 강제한다.
  WAITING_COVER → (덮개 열림) → INSPECTING → (덮개 닫힘) → 다음 부품
  INSPECTING 상태가 되기 전까지는 inside/side/hammer 를 카운트하지 않는다.
  (박스 안에 놓인 바구니/부품 오검출로 체크박스가 켜지는 문제 방지)

전제 파일 (같은 폴더):
  - cover_check.py, hammer_check.py  (팀원)
  - cover.pt, hammer.pt              (팀원 가중치)
  - armrest_3class_best.pt           (내 seg 모델)
  - fonts/NotoSansCJK-*.ttc
설치:
  pip install ultralytics opencv-python-headless pillow mediapipe tqdm torch
========================================================================
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from tqdm import tqdm

from cover_check import CoverCheck, arms_from_pose
from hammer_check import HammerCheck

# ======================================================================
# 경로 (로컬에 맞게 수정)
# ======================================================================
BASE = "/Users/songminseo/Desktop/kkokkodack"
SEG_MODEL_PATH = f"{BASE}/model/code/1_plus_3/armrest_3class_best.pt"
COVER_WEIGHTS = f"{BASE}/model/code/1_plus_3/cover.pt"
HAMMER_WEIGHTS = f"{BASE}/model/code/1_plus_3/hammer.pt"
FONT_REGULAR = f"{BASE}/fonts/NotoSansCJK-Regular.ttc"
FONT_BOLD = f"{BASE}/fonts/NotoSansCJK-Bold.ttc"
POSE_TASK_PATH = f"{BASE}/model/code/1_plus_3/pose_landmarker_lite.task"  # MediaPipe Tasks 모델

# ======================================================================
# 디바이스
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
# 모델 로드
# ======================================================================
seg_model = YOLO(SEG_MODEL_PATH)
seg_model.to(DEVICE)
SEG_NAMES = seg_model.names  # {0:'inside',1:'top',2:'side'}

cover = CoverCheck(COVER_WEIGHTS)
hammer = HammerCheck(HAMMER_WEIGHTS)

# ======================================================================
# MediaPipe Pose (Tasks API)
# ======================================================================
USE_POSE = True
pose_landmarker = None
if USE_POSE:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base_opts = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        print("[INFO] MediaPipe Pose 로드됨")
    except Exception as e:
        print(f"[WARN] MediaPipe Pose 로드 실패 ({e}) → 팔 동작 없이 면적 기준만 사용")
        pose_landmarker = None

# ======================================================================
# 임계값
# ======================================================================
CONF_TH = 0.4            # 체크리스트 반영 최소 확신도 (seg)
AREA_TH = 0.03
MERGE_GAP = 0.5
PARTS_PER_BOX = 3

DISPLAY_CONF_TH = {"inside": 0.7, "top": 0.7, "side": 0.8}

# ======================================================================
# 폰트
# ======================================================================
_fc = {}
def _font(sz, bold=False):
    k = (sz, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, sz)
    return _fc[k]

# ======================================================================
# 1단계 트래커 (안쪽면 → 옆면 순서 강제)
# ======================================================================
@dataclass
class Step1Tracker:
    acc_inside: float = 0.0
    acc_side: float = 0.0
    _last_t: dict = field(default_factory=lambda: {"inside": None, "side": None})

    MIN_DUR_INSIDE = 0.3
    MIN_DUR_SIDE = 0.3

    def _accum(self, view, t):
        last = self._last_t[view]
        delta = 0.0
        if last is not None and (t - last) <= MERGE_GAP:
            delta = t - last
        self._last_t[view] = t
        return delta

    def inside_done(self): return self.acc_inside >= self.MIN_DUR_INSIDE
    def side_done(self): return self.acc_side >= self.MIN_DUR_SIDE
    def done(self): return self.inside_done() and self.side_done()

    def update(self, view, t):
        if view == "inside":
            self.acc_inside += self._accum("inside", t)
            return "accepted"
        elif view == "side":
            if not self.inside_done():
                return "waiting_inside"
            self.acc_side += self._accum("side", t)
            return "accepted"
        return "ignored"

    def phase(self):
        if self.done(): return "STEP1_DONE"
        if self.inside_done(): return "WAIT_SIDE"
        if self.acc_inside > 0: return "WAIT_INSIDE"
        return "IDLE"

# ======================================================================
# 사이클 상태 머신
# ======================================================================
class Cycle:
    WAITING_COVER = "WAITING_COVER"   # 덮개 덮인 상태, 열리길 대기
    INSPECTING = "INSPECTING"          # 검사 진행 중

    def __init__(self):
        self.state = self.WAITING_COVER
        self.part_idx = 0              # 현재 부품 (0,1,2)
        self.completed = 0            # 완료된 부품 수
        self.step1 = Step1Tracker()
        self.step3_done = False       # 망치 2회 완료 여부
        self.warnings = []            # (부품번호, [누락단계])

    def reset_part(self):
        self.step1 = Step1Tracker()
        self.step3_done = False

# ======================================================================
# seg 인식
# ======================================================================
SEG_MASK_COLORS = {
    "inside": np.array([0, 220, 80]),
    "side": np.array([255, 180, 0]),
    "top": np.array([80, 180, 255]),
}

def merge_mask_fragments(mask_bin):
    """
    마스크가 여러 조각(blob)으로 쪼개져 보이는 문제 해결.
    조각들을 다 포함하는 가장 바깥 윤곽선(convex hull)을 구하고,
    그 안쪽을 전부 채워서 하나의 덩어리로 만든다.
    """
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_bin
    all_pts = np.vstack(cnts)
    hull = cv2.convexHull(all_pts)
    filled = np.zeros_like(mask_bin)
    cv2.drawContours(filled, [hull], -1, 1, -1)
    return filled

def detect_seg(frame):
    results = seg_model(frame, conf=0.35, verbose=False, device=DEVICE)[0]
    H, W = frame.shape[:2]
    if results.boxes is None or len(results.boxes) == 0:
        return "none", 0.0, 0.0, None
    idx = int(results.boxes.conf.argmax())
    cls_id = int(results.boxes.cls[idx])
    conf = float(results.boxes.conf[idx])
    view = SEG_NAMES.get(cls_id, "none")
    mask_overlay = None
    area_ratio = 0.0
    if results.masks is not None and len(results.masks) > idx:
        mask = results.masks.data[idx].cpu().numpy()
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = (mask > 0.5).astype(np.uint8)
        mask_overlay = merge_mask_fragments(raw_mask)
        area_ratio = mask_overlay.sum() / (W * H)
    return view, conf, area_ratio, mask_overlay

def passes_display(view, conf):
    return conf >= DISPLAY_CONF_TH.get(view, 0.7)

# ======================================================================
# MediaPipe pose → arms dict
# ======================================================================
def get_arms(frame_rgb, t_ms):
    if pose_landmarker is None:
        return None
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = pose_landmarker.detect_for_video(mp_image, int(t_ms))
    # cover_check.arms_from_pose 는 .pose_landmarks[0] 리스트를 기대
    return arms_from_pose(result)

# ======================================================================
# 렌더링
# ======================================================================
PHASE_DISPLAY = {
    "IDLE": ("대기 중", (120, 120, 120)),
    "WAIT_INSIDE": ("안쪽면 확인 중", (0, 220, 80)),
    "WAIT_SIDE": ("옆면 확인 중", (255, 180, 0)),
    "STEP1_DONE": ("1단계 완료 ✓", (0, 200, 100)),
}
STATE_LABEL = {
    "WAITING_COVER": ("덮개 열림 대기", (150, 150, 150)),
    "INSPECTING": ("검사 중", (0, 200, 120)),
}

def render(frame, W, H, pw, cyc, seg_view, seg_conf, seg_area, seg_mask,
           show_seg, cover_area, cover_base, hammer_strikes):
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)
    disp = frame.copy()

    # 덮개(cover) 마스크 오버레이 — 항상 표시
    if cover.last_mask is not None:
        H_f, W_f = frame.shape[:2]
        cmask = cv2.resize(cover.last_mask.astype(np.uint8), (W_f, H_f), interpolation=cv2.INTER_NEAREST)
        cover_color = np.array([200, 200, 0])
        disp[cmask > 0] = (disp[cmask > 0] * 0.55 + cover_color * 0.45).astype(np.uint8)
        cnts_c, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts_c, -1, (0, 220, 220), 2)

    # seg 마스크 오버레이 (검사 중 + 표시 기준 통과 시만)
    if (cyc.state == Cycle.INSPECTING and show_seg
            and seg_mask is not None and seg_view in SEG_MASK_COLORS):
        color = SEG_MASK_COLORS[seg_view]
        disp[seg_mask > 0] = (disp[seg_mask > 0] * 0.45 + color * 0.55).astype(np.uint8)
        cnts, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, tuple(int(x) for x in color), 3)

    # 망치 bbox 오버레이 — 검사 중 + 확신도 60% 이상일 때만
    HAMMER_DISPLAY_CONF_TH = 0.60
    if (cyc.state == Cycle.INSPECTING and hammer.last_box is not None
            and hammer.last_conf >= HAMMER_DISPLAY_CONF_TH):
        H_f, W_f = frame.shape[:2]
        cx_n, cy_n, bw_n, bh_n = hammer.last_box
        x1 = int((cx_n - bw_n / 2) * W_f)
        y1 = int((cy_n - bh_n / 2) * H_f)
        x2 = int((cx_n + bw_n / 2) * W_f)
        y2 = int((cy_n + bh_n / 2) * H_f)
        conf_h = hammer.last_conf
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 100, 255), 3)
        label_h = f"hammer {conf_h:.0%}"
        (tw, th), _ = cv2.getTextSize(label_h, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(disp, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 100, 255), -1)
        cv2.putText(disp, label_h, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    canvas[:, :W] = disp
    canvas[:, W:] = (32, 32, 32)

    # 중앙 배지: 상태에 따라
    if cyc.state == Cycle.WAITING_COVER:
        vlabel, vcolor = STATE_LABEL["WAITING_COVER"]
    else:
        vlabel, vcolor = PHASE_DISPLAY[cyc.step1.phase()]
    bw2, bh2 = 400, 60
    bx = (W - bw2) // 2
    cv2.rectangle(canvas, (bx, 18), (bx + bw2, 18 + bh2), vcolor, -1)
    cv2.rectangle(canvas, (bx, 18), (bx + bw2, 18 + bh2), (255, 255, 255), 2)
    cv2.rectangle(canvas, (W - 300, 20), (W - 20, 72), (0, 0, 0), -1)

    bar_h = 90
    cv2.rectangle(canvas, (0, H - bar_h), (W, H), (20, 20, 20), -1)

    # 우측 체크리스트 — 단계별 1줄씩 (요청: 1단계는 하나로 통합)
    checks = [
        ("1", "안쪽면 기스·파손 확인", cyc.step1.done()),
        ("3", "옆면 연결부 Hammer 확인", cyc.step3_done),
    ]
    y = 130
    for _, _, done in checks:
        y += 80
        col = (0, 200, 100) if done else (90, 90, 90)
        bx2 = W + 28
        cv2.rectangle(canvas, (bx2, y - 16), (bx2 + 34, y + 18), col, -1)
        cv2.rectangle(canvas, (bx2, y - 16), (bx2 + 34, y + 18), (200, 200, 200), 2)
        if done:
            cv2.line(canvas, (bx2 + 6, y + 1), (bx2 + 14, y + 11), (255, 255, 255), 3)
            cv2.line(canvas, (bx2 + 14, y + 11), (bx2 + 28, y - 9), (255, 255, 255), 3)

    warn = None
    if cyc.warnings:
        lp, miss = cyc.warnings[-1]
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
    txt(((W - tw) // 2, 28), vlabel, 26, (255, 255, 255), True)

    txt((W - 285, 30), f"검수 완료  {cyc.completed} / {PARTS_PER_BOX}", 26, (0, 255, 180), True)

    # 하단 디버그/상태
    by = H - bar_h + 10
    st_label, _ = STATE_LABEL.get(cyc.state, ("-", (200, 200, 200)))
    txt((20, by), f"상태: {st_label}   덮개: {cover_area*100:.1f}% / 기준 {cover_base*100:.1f}%", 18, (200, 220, 255), True)
    raw = seg_view if seg_view != "none" else "-"
    th_v = DISPLAY_CONF_TH.get(seg_view, 0.7)
    cc = (100, 255, 150) if seg_conf >= th_v else (180, 180, 180)
    txt((20, by + 30), f"[seg] {raw} {seg_conf:.0%}(기준{th_v:.0%})   [망치] 타격 {hammer_strikes}회", 17, cc)

    # 우측 체크리스트 텍스트
    txt((W + 18, 92), "검사 체크리스트", 22, (255, 255, 255), True)
    y = 130
    for code, label, done in checks:
        y += 80
        sc = (100, 255, 150) if done else (180, 180, 180)
        txt((W + 75, y - 18), f"{code}단계", 17, (200, 200, 200), True)
        txt((W + 75, y + 2), label, 15, (225, 225, 225))
        txt((W + 75, y + 24), "✓ 완료" if done else "미확인", 14, sc, True)

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
def run(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = 420
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + pw, H))

    cover.reset()
    hammer.reset()
    cyc = Cycle()

    fi = 0
    pbar = tqdm(total=total, desc=video_path.split("/")[-1], unit="f")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 공통: 덮개 + 팔 ---
        arms = get_arms(frame_rgb, t * 1000) if pose_landmarker else None
        cover.feed(frame, t, arms)

        # --- seg 인식 ---
        seg_view, seg_conf, seg_area, seg_mask = detect_seg(frame)
        show_seg = (seg_view in SEG_MASK_COLORS and passes_display(seg_view, seg_conf)
                    and seg_area >= AREA_TH)

        # --- 망치 인식 (상태 무관하게 feed, 카운트는 INSPECTING에서만 반영) ---
        hammer.feed(frame, t)

        # ============ 상태 머신 ============
        if cyc.state == Cycle.WAITING_COVER:
            opened, _ = cover.verdict_open()
            if opened:
                cyc.state = Cycle.INSPECTING
                cyc.reset_part()
                hammer.reset()

        elif cyc.state == Cycle.INSPECTING:
            # 1단계 (안쪽면/옆면)
            if seg_view in ("inside", "side") and seg_conf >= CONF_TH and seg_area >= AREA_TH:
                cyc.step1.update(seg_view, t)

            # 3단계 (망치 2회)
            if not cyc.step3_done:
                ok3, _ = hammer.verdict()
                if ok3:
                    cyc.step3_done = True
                    hammer.hold()

            # 덮개 닫힘 = 이 부품 검사 종료
            closed, _ = cover.verdict_close()
            if closed:
                missing = []
                if not cyc.step1.done():
                    missing.append("1단계: 안쪽면 기스·파손 확인")
                if not cyc.step3_done:
                    missing.append("3단계: 옆면 Hammer 확인")
                if missing:
                    cyc.warnings.append((cyc.completed + 1, missing))
                cyc.completed += 1
                # 다음 부품 준비: 덮개 상태 다시 측정하도록 리셋
                cover.reset()
                hammer.reset()
                cyc.reset_part()
                cyc.state = Cycle.WAITING_COVER

        # --- 렌더 ---
        canvas = render(frame, W, H, pw, cyc, seg_view, seg_conf, seg_area, seg_mask,
                        show_seg, cover.last_area, cover.base, hammer.strikes)
        writer.write(canvas)
        fi += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"완료: {cyc.completed}개 부품 검수, 경고 {len(cyc.warnings)}건")
    for p, ms in cyc.warnings:
        print(f"  부품#{p}: {ms}")
    return cyc.completed, cyc.warnings


if __name__ == "__main__":
    import sys, subprocess
    src = sys.argv[1] if len(sys.argv) > 1 else \
        f"{BASE}/data/media/cycle_mp4/crop017.mp4"
    out = f"{BASE}/model/result_v4.mp4"
    run(src, out)
    subprocess.run(["open", out])