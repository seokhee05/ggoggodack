"""
========================================================================
Armrest 불량 검사 - 통합 파이프라인 v6
========================================================================
v5 → v6 수정 (민서 피드백 7건):
  1. 마스크 조각 합치기(convex hull) 제거 → 원본 마스크 그대로 표시
  2. 팔(피부색) 오인식 방지 → seg 마스크 영역의 평균색이 피부색이면 제외
  3,4,5. 표시 기준 = 판정 기준 통일 (inside 0.70 / side 0.80)
         → "화면에 안 보이는데 체크됨/배지 안 맞음" 문제 해결
         → phase/배지/체크리스트가 전부 같은 기준으로 동작
  6. '검수 완료'를 제품(부품) 단위로 카운트.
     상단에 "제품 N개 완료 · 박스 M개" 분리 표시
  7. 미검수 알람 — 제품 하나라도 단계 누락이면 경고 표시

⚠️ 4단계(비닐 제거 후 윗면)는 아직 다른 팀원 미연동 → 체크리스트에서 제외.
   현재 판정 단계: 1단계(안쪽면/옆면) + 2단계(힌지) + 3단계(망치)
========================================================================
"""

import sys
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from tqdm import tqdm

# ======================================================================
# 경로
# ======================================================================
BASE = "/Users/songminseo/Desktop/kkokkodack"
DIR_1_3 = f"{BASE}/model/code/1_plus_3"
DIR_HINGE = f"{BASE}/model/code/2_hinge"

SEG_MODEL_PATH = f"{DIR_1_3}/armrest_3class_best.pt"
COVER_WEIGHTS = f"{DIR_1_3}/cover.pt"
HAMMER_WEIGHTS = f"{DIR_1_3}/hammer.pt"
FONT_REGULAR = f"{BASE}/fonts/NotoSansCJK-Regular.ttc"
FONT_BOLD = f"{BASE}/fonts/NotoSansCJK-Bold.ttc"
POSE_TASK_PATH = f"{DIR_1_3}/pose_landmarker_lite.task"

HINGE_WEIGHTS = f"{DIR_HINGE}/hinge_det_runs/round2/weights/best.pt"
ARMREST_SLOT_WEIGHTS = f"{DIR_HINGE}/armrest_box_runs/box2/weights/best.pt"

sys.path.insert(0, DIR_1_3)
sys.path.insert(0, DIR_HINGE)

from cover_check import CoverCheck, arms_from_pose
from hammer_check import HammerCheck
from hinge_check import (
    LiveJudge, PoseEstimator, StaticFilter, SlotTracker,
    detect_hinges, held_box, hinge_crop, ANALYZE_FPS as HINGE_ANALYZE_FPS,
)

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
SEG_NAMES = seg_model.names

cover = CoverCheck(COVER_WEIGHTS)
hammer = HammerCheck(HAMMER_WEIGHTS)

hinge_model = YOLO(HINGE_WEIGHTS)
armrest_slot_model = None
if os.path.exists(ARMREST_SLOT_WEIGHTS):
    armrest_slot_model = YOLO(ARMREST_SLOT_WEIGHTS)
    print("[INFO] 암레스트 자리(slot) 모델 로드됨")
else:
    print("[INFO] 암레스트 자리 모델 없음 → 힌지 순서 기반 추적")

hinge_pose = PoseEstimator()

# ======================================================================
# MediaPipe Pose (덮개 전용)
# ======================================================================
# ======================================================================
# MediaPipe Pose (덮개 전용)
# ======================================================================
pose_landmarker = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    
    # Delegate를 CPU로 명시적 지정 (Mac Metal API 충돌 방지)
    base_opts = mp_python.BaseOptions(
        model_asset_path=POSE_TASK_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    print("[INFO] MediaPipe Pose(덮개용) 로드됨 (CPU 모드)")
except Exception as e:
    print(f"[WARN] MediaPipe Pose 로드 실패 ({e}) → 면적 기준만 사용")
    pose_landmarker = None

# ======================================================================
# 임계값
# ======================================================================
# [수정 3,4,5] 표시 기준 = 판정 기준 통일. 이 값 하나로 "화면 표시"와
#             "체크리스트 반영"을 모두 결정 → 어긋남 없음.
CONF_TH = {"inside": 0.70, "side": 0.80, "top": 0.70}
AREA_TH = 0.03
MERGE_GAP = 0.5
PARTS_PER_BOX = 3
ABSENT_RESET_SEC = 2.5
HAMMER_DISPLAY_CONF_TH = 0.60

# [수정 2] 피부색 필터 — seg 마스크 안쪽 평균색이 피부색 범위면 팔로 보고 제외
SKIN_FILTER_ON = True
# HSV 기준 피부색 대략 범위 (H 0~25 또는 160~180, S 25~170, V 70~255)
def is_skin_region(frame_bgr, mask_bin, skin_ratio_th=0.55):
    """마스크 영역 픽셀 중 피부색 비율이 높으면 True (팔로 판단)."""
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 50:
        return False
    sample = frame_bgr[ys, xs]  # N x 3 BGR
    hsv = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    skin = (((h <= 25) | (h >= 160)) & (s >= 25) & (s <= 180) & (v >= 70))
    return (skin.mean() >= skin_ratio_th)

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
# 1단계 (안쪽면 → 옆면 순서 강제)
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
        # 여기 들어오는 view 는 "표시 기준(=판정 기준) 통과한" 것만.
        if view == "inside":
            self.acc_inside += self._accum("inside", t)
            return "accepted"
        elif view == "side":
            if not self.inside_done():
                return "waiting_inside"   # [수정 5] 안쪽면 전엔 옆면 누적 안 함
            self.acc_side += self._accum("side", t)
            return "accepted"
        return "ignored"

    def phase(self):
        # [수정 3,4] 실제 누적 상태만으로 배지 결정
        if self.done(): return "STEP1_DONE"
        if self.inside_done(): return "WAIT_SIDE"
        if self.acc_inside > 0: return "WAIT_INSIDE"
        return "IDLE"

# ======================================================================
# 부품 전환 추적
# ======================================================================
class PartTracker:
    def __init__(self, n_parts, fps):
        self.n = n_parts
        self.idx = 0
        self.fps = fps
        self.step1 = [Step1Tracker() for _ in range(n_parts)]
        self.step3_done = [False] * n_parts
        self.part_done_flag = [False] * n_parts   # [수정 6] 제품 완료 카운트용
        self._absent = 0
        self._had_activity = False
        self.absent_reset_frames = int(fps * ABSENT_RESET_SEC)

    def current(self):
        return self.idx if self.idx < self.n else None

    def on_presence(self, present):
        k = self.current()
        if k is None:
            return
        if present:
            self._had_activity = True
            self._absent = 0
        else:
            self._absent += 1
            if self._had_activity and self._absent >= self.absent_reset_frames:
                self._advance()

    def _advance(self):
        self.idx += 1
        self._absent = 0
        self._had_activity = False
        hammer.reset()

    def feed_step1(self, view, t):
        k = self.current()
        if k is None:
            return "ignored"
        return self.step1[k].update(view, t)

    def feed_hammer_pass(self):
        k = self.current()
        if k is None or self.step3_done[k]:
            return
        ok3, _ = hammer.verdict()
        if ok3:
            self.step3_done[k] = True

# ======================================================================
# 사이클 (박스 단위: 덮개 1번 열림~닫힘)
# ======================================================================
class Cycle:
    WAITING_COVER = "WAITING_COVER"
    INSPECTING = "INSPECTING"

    def __init__(self, fps):
        self.state = self.WAITING_COVER
        self.fps = fps
        self.parts = PartTracker(PARTS_PER_BOX, fps)
        self.hinge_judge = None
        self.boxes_completed = 0       # [수정 6] 박스 수
        self.parts_completed = 0       # [수정 6] 제품(부품) 수
        self.warnings = []             # [(박스, {부품: [누락]})]
        self.box_idx = 0

    def start_inspecting(self):
        self.state = self.INSPECTING
        self.parts = PartTracker(PARTS_PER_BOX, self.fps)
        self.hinge_judge = LiveJudge()
        if armrest_slot_model is not None:
            self.hinge_judge.slots = SlotTracker()
        hammer.reset()

    def part_is_complete(self, k, hinge_state):
        """제품 k 가 1·2·3단계 모두 완료인지"""
        return (self.parts.step1[k].done()
                and hinge_state == "완료"
                and self.parts.step3_done[k])

    def update_part_counts(self, hinge_states_live):
        """[수정 6,7] 매 프레임 제품 완료 여부 갱신 (한 번 완료되면 유지)"""
        for k in range(PARTS_PER_BOX):
            if self.parts.part_done_flag[k]:
                continue
            hs = hinge_states_live[k][0] if k < len(hinge_states_live) else "대기"
            if self.part_is_complete(k, hs):
                self.parts.part_done_flag[k] = True
                self.parts_completed += 1

    def finish_box(self):
        self.hinge_judge.finish()
        hinge_states = self.hinge_judge.product_states(finished=True)
        box_missing = {}
        for k in range(PARTS_PER_BOX):
            missing = []
            if not self.parts.step1[k].done():
                missing.append("1단계: 안쪽면/옆면 확인")
            if hinge_states[k][0] != "완료":
                missing.append("2단계: Hinge 작동 확인")
            if not self.parts.step3_done[k]:
                missing.append("3단계: 옆면 Hammer 확인")
            if missing:
                box_missing[k + 1] = missing
            else:
                # 완료인데 아직 카운트 안 됐으면 보정
                if not self.parts.part_done_flag[k]:
                    self.parts.part_done_flag[k] = True
                    self.parts_completed += 1
        self.box_idx += 1
        if box_missing:
            self.warnings.append((self.box_idx, box_missing))
        self.boxes_completed += 1
        self.state = self.WAITING_COVER
        return hinge_states

# ======================================================================
# seg 인식 (1단계)
# ======================================================================
SEG_MASK_COLORS = {
    "inside": np.array([0, 220, 80]),
    "side": np.array([255, 180, 0]),
    "top": np.array([80, 180, 255]),
}

def detect_seg(frame):
    """[수정 1] convex-hull 합치기 제거 — 원본 마스크 그대로."""
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
        mask_overlay = (mask > 0.5).astype(np.uint8)   # 원본 그대로
        area_ratio = mask_overlay.sum() / (W * H)

    # [수정 2] 피부색(팔) 필터
    if SKIN_FILTER_ON and mask_overlay is not None and view in ("inside", "side"):
        if is_skin_region(frame, mask_overlay):
            return "none", 0.0, 0.0, None

    return view, conf, area_ratio, mask_overlay

def passes(view, conf, area):
    """표시 = 판정 통일 기준."""
    return (view in CONF_TH and conf >= CONF_TH[view] and area >= AREA_TH)

def get_arms(frame_rgb, t_ms):
    if pose_landmarker is None:
        return None
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = pose_landmarker.detect_for_video(mp_image, int(t_ms))
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
           show_seg, hinge_boxes, hinge_hb, hinge_state_live):
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)
    disp = frame.copy()

    # 덮개 마스크 — 항상
    if cover.last_mask is not None:
        Hf, Wf = frame.shape[:2]
        cmask = cv2.resize(cover.last_mask.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
        disp[cmask > 0] = (disp[cmask > 0] * 0.55 + np.array([200, 200, 0]) * 0.45).astype(np.uint8)
        cnts_c, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts_c, -1, (0, 220, 220), 2)

    part_k = cyc.parts.current() if cyc.state == Cycle.INSPECTING else None

    # 1단계 seg 마스크 — 검사 중 + 통일 기준 통과시만
    if (cyc.state == Cycle.INSPECTING and show_seg
            and seg_mask is not None and seg_view in SEG_MASK_COLORS):
        color = SEG_MASK_COLORS[seg_view]
        disp[seg_mask > 0] = (disp[seg_mask > 0] * 0.45 + color * 0.55).astype(np.uint8)
        cnts, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, tuple(int(x) for x in color), 3)

    # 힌지 bbox
    if cyc.state == Cycle.INSPECTING:
        Hf, Wf = frame.shape[:2]
        for b in hinge_boxes:
            is_held = hinge_hb is not None and np.allclose(b, hinge_hb)
            c = (0, 0, 255) if is_held else (255, 0, 0)
            cv2.rectangle(disp, (int(b[0]*Wf), int(b[1]*Hf)), (int(b[2]*Wf), int(b[3]*Hf)), c, 3)

    # 망치 bbox (60%+)
    if (cyc.state == Cycle.INSPECTING and hammer.last_box is not None
            and hammer.last_conf >= HAMMER_DISPLAY_CONF_TH):
        Hf, Wf = frame.shape[:2]
        cx_n, cy_n, bw_n, bh_n = hammer.last_box
        x1, y1 = int((cx_n-bw_n/2)*Wf), int((cy_n-bh_n/2)*Hf)
        x2, y2 = int((cx_n+bw_n/2)*Wf), int((cy_n+bh_n/2)*Hf)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 100, 255), 3)
        lh = f"hammer {hammer.last_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lh, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(disp, (x1, y1-th-8), (x1+tw+6, y1), (0, 100, 255), -1)
        cv2.putText(disp, lh, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    canvas[:, :W] = disp
    canvas[:, W:] = (32, 32, 32)

    # 중앙 배지 (phase 기반)
    if cyc.state == Cycle.WAITING_COVER:
        vlabel, vcolor = STATE_LABEL["WAITING_COVER"]
    else:
        phase = cyc.parts.step1[part_k].phase() if part_k is not None else "IDLE"
        vlabel, vcolor = PHASE_DISPLAY[phase]
    bw2 = 420
    bx = (W - bw2) // 2
    cv2.rectangle(canvas, (bx, 18), (bx+bw2, 78), vcolor, -1)
    cv2.rectangle(canvas, (bx, 18), (bx+bw2, 78), (255,255,255), 2)
    cv2.rectangle(canvas, (W-340, 20), (W-20, 96), (0,0,0), -1)

    bar_h = 90
    cv2.rectangle(canvas, (0, H-bar_h), (W, H), (20,20,20), -1)

    # 체크리스트 (현재 부품)
    checks = [
        ("1", "안쪽면/옆면 확인", cyc.parts.step1[part_k].done() if part_k is not None else False),
        ("2", "Hinge 작동 확인", hinge_state_live == "완료"),
        ("3", "옆면 Hammer 확인", cyc.parts.step3_done[part_k] if part_k is not None else False),
    ]
    y = 150
    for _, _, done in checks:
        y += 72
        col = (0,200,100) if done else (90,90,90)
        bx2 = W + 28
        cv2.rectangle(canvas, (bx2, y-16), (bx2+34, y+18), col, -1)
        cv2.rectangle(canvas, (bx2, y-16), (bx2+34, y+18), (200,200,200), 2)
        if done:
            cv2.line(canvas, (bx2+6, y+1), (bx2+14, y+11), (255,255,255), 3)
            cv2.line(canvas, (bx2+14, y+11), (bx2+28, y-9), (255,255,255), 3)

    warn = None
    if cyc.warnings:
        bno, miss_dict = cyc.warnings[-1]
        y0 = H - 270
        cv2.rectangle(canvas, (W+12, y0), (W+pw-12, y0+230), (0,0,160), -1)
        cv2.rectangle(canvas, (W+12, y0), (W+pw-12, y0+230), (0,100,255), 2)
        warn = (bno, miss_dict, y0)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    def txt(xy, s, sz=20, c=(255,255,255), b=False):
        d.text(xy, s, font=_font(sz, b), fill=c)

    bbox = _font(26, True).getbbox(vlabel)
    txt(((W-(bbox[2]-bbox[0]))//2, 36), vlabel, 26, (255,255,255), True)

    # [수정 6] 제품 단위 카운트 + 박스 수
    txt((W-325, 28), f"검수 완료 제품 {cyc.parts_completed}개", 24, (0,255,180), True)
    part_label = f"현재 부품 {part_k+1}/{PARTS_PER_BOX}" if part_k is not None else "-"
    txt((W-325, 62), f"박스 {cyc.boxes_completed}개 · {part_label}", 18, (200,220,255))

    by = H - bar_h + 10
    st_label, _ = STATE_LABEL.get(cyc.state, ("-", (200,200,200)))
    txt((20, by), f"상태: {st_label}   덮개: {cover.last_area*100:.1f}% / 기준 {cover.base*100:.1f}%", 18, (200,220,255), True)
    raw = seg_view if seg_view != "none" else "-"
    th_v = int(CONF_TH.get(seg_view, 0.7)*100)
    passed = passes(seg_view, seg_conf, seg_area)
    cc = (100,255,150) if passed else (180,180,180)
    txt((20, by+30), f"[seg] {raw} {seg_conf:.0%}(기준{th_v}%){'✓' if passed else ''}  [힌지] {hinge_state_live}  [망치] {hammer.strikes}회", 16, cc)

    txt((W+18, 110), "검사 체크리스트 (현재 부품)", 20, (255,255,255), True)
    y = 150
    for code, label, done in checks:
        y += 72
        sc = (100,255,150) if done else (180,180,180)
        txt((W+75, y-18), f"{code}단계", 17, (200,200,200), True)
        txt((W+75, y+2), label, 15, (225,225,225))
        txt((W+75, y+24), "✓ 완료" if done else "미확인", 14, sc, True)

    if warn:
        bno, miss_dict, y0 = warn
        txt((W+25, y0+12), "⚠ 검사 누락 경고", 20, (0,255,255), True)
        txt((W+25, y0+44), f"박스 #{bno}:", 15, (255,255,255))
        yy = y0 + 70
        for pn, miss in miss_dict.items():
            txt((W+30, yy), f"부품 {pn}:", 14, (255,230,150), True)
            yy += 20
            for m in miss:
                txt((W+40, yy), f"· {m}", 13, (200,200,255))
                yy += 20

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ======================================================================
# 메인
# ======================================================================
def run(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = 460
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W+pw, H))

    cover.reset()
    hinge_pose.new_video()
    static_filter = StaticFilter()
    hinge_step = max(1, int(round(fps / HINGE_ANALYZE_FPS)))

    cyc = Cycle(fps)
    fi = 0
    pbar = tqdm(total=total, desc=video_path.split("/")[-1], unit="f")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        arms = get_arms(frame_rgb, t*1000) if pose_landmarker else None
        cover.feed(frame, t, arms)

        seg_view, seg_conf, seg_area, seg_mask = detect_seg(frame)
        # [수정 3,4,5] 표시=판정 통일: passes() 하나로 결정
        ok_seg = passes(seg_view, seg_conf, seg_area)
        show_seg = ok_seg and seg_view in ("inside", "side")

        hammer.feed(frame, t)

        hinge_boxes, hinge_hb = np.zeros((0,4)), None
        hinge_state_live = "대기"

        if cyc.state == Cycle.WAITING_COVER:
            opened, _ = cover.verdict_open()
            if opened:
                cyc.start_inspecting()

        elif cyc.state == Cycle.INSPECTING:
            # 부품 전환은 "통일 기준 통과한 seg 존재" 기준
            cyc.parts.on_presence(ok_seg and seg_view in ("inside", "side"))
            if ok_seg and seg_view in ("inside", "side"):
                cyc.parts.feed_step1(seg_view, t)
            cyc.parts.feed_hammer_pass()

            if fi % hinge_step == 0:
                hinge_boxes = detect_hinges(frame, hinge_model)
                kp, mp_lm, pose_src = hinge_pose(frame, t*1000)
                st = static_filter(hinge_boxes)
                hinge_hb = held_box(hinge_boxes, kp, st)
                if armrest_slot_model is not None and cyc.hinge_judge.slots is not None:
                    ar = armrest_slot_model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0]
                    abx = ar.boxes.xyxy.cpu().numpy() / [W, H, W, H] if len(ar.boxes) else np.zeros((0,4))
                    cyc.hinge_judge.slots.update(t, abx)
                cyc.hinge_judge.push(t, hinge_hb, kp, hinge_crop(frame, hinge_hb))

            # 현재 부품의 힌지 상태 (live)
            live_states = cyc.hinge_judge.product_states(finished=False)
            part_k = cyc.parts.current()
            if part_k is not None and part_k < len(live_states):
                hinge_state_live = live_states[part_k][0]

            # [수정 6,7] 제품 완료 카운트 실시간 갱신
            cyc.update_part_counts(live_states)

            closed, _ = cover.verdict_close()
            if closed:
                cyc.finish_box()
                cover.reset()

        canvas = render(frame, W, H, pw, cyc, seg_view, seg_conf, seg_area, seg_mask,
                        show_seg, hinge_boxes, hinge_hb, hinge_state_live)
        writer.write(canvas)
        fi += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"\n완료: 제품 {cyc.parts_completed}개 검수, 박스 {cyc.boxes_completed}개, 경고 {len(cyc.warnings)}건")
    for bno, miss_dict in cyc.warnings:
        print(f"  박스#{bno}:")
        for pn, miss in miss_dict.items():
            print(f"    부품{pn}: {miss}")
    return cyc.parts_completed, cyc.warnings


if __name__ == "__main__":
    import subprocess
    src = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/data/media/cycle_mp4/crop020.mp4"
    out = f"{BASE}/model/result_v6.mp4"
    run(src, out)
    subprocess.run(["open", out])