"""
========================================================================
Armrest 불량 검사 - 통합 파이프라인 v8
========================================================================
v7 → v8 수정:
  D. 4단계(비닐 제거 후 윗면 육안 확인) 통합
     - 장갑 검출 모델(glove_detector_v2/best.pt) 추가
     - Step4Checker: WAITING→ACTIVE→COMPLETE 상태 머신
       체크 3개: 양손 장갑 감지 / 팔-장갑 연결 / 윗면(top) 근거
     - mediapipe Pose는 기존 덮개용 pose_landmarker 재사용 불가
       → 4단계 전용으로 mp.solutions.pose.Pose() 사용
     - 4단계 체크리스트에 4단계 항목 추가

v6 → v7 수정 (4건):
  A. seg 마스크 최대 연결 컴포넌트만 남김 → 팔·배경에 튀는 작은 조각 제거
  B. 힌지 bbox: 1단계 완료 전에는 화면에 미표시
     1단계 완료 후에도 손목과 겹치는(held) 힌지만 표시
  C. 미검수 의심 경고 — 우하단 빨간박스
     검사 중 암레스트 미감지 5초 이상이면 실시간 경고

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

# 4단계 전용
DIR_STEP4 = f"{BASE}/model/code/4_top"
GLOVE_WEIGHTS = f"{DIR_STEP4}/glove_runs/glove_detector_v2/weights/best.pt"

sys.path.insert(0, DIR_1_3)
sys.path.insert(0, DIR_HINGE)
sys.path.insert(0, DIR_STEP4)

from cover_check import CoverCheck, arms_from_pose
from hammer_check import HammerCheck
from hinge_check import (
    LiveJudge, PoseEstimator, StaticFilter, SlotTracker,
    detect_hinges, held_box, hinge_crop, ANALYZE_FPS as HINGE_ANALYZE_FPS,
)

# 4단계 유틸
# segment_status는 step4_glove_pose_dashboard_v8_hybrid_top.py에 있음
# get_segments / choose_side / white_surface_review / find_pair 는 step4_glove_top_v3.py
try:
    from step4_glove_top_v3 import (
        get_segments, choose_side, white_surface_review, find_pair,
    )
    from step4_glove_pose_dashboard_v8_hybrid_top import segment_status as _seg_status4
    _STEP4_UTILS_OK = True
except ImportError as e:
    print(f"[WARN] 4단계 유틸 로드 실패 ({e}) → 4단계 비활성화")
    _STEP4_UTILS_OK = False
    _seg_status4 = None

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
# 4단계 글러브 모델
# ======================================================================
glove_model = None
if _STEP4_UTILS_OK and os.path.exists(GLOVE_WEIGHTS):
    glove_model = YOLO(GLOVE_WEIGHTS)
    print("[INFO] 4단계 글러브 모델 로드됨")
else:
    print("[WARN] 글러브 모델 없음 → 4단계 비활성화")

# 4단계 전용 MediaPipe Pose (mp.solutions.pose — 덮개용 PoseLandmarker와 별개)
_mp4_pose = None
try:
    import mediapipe as _mp4_mp
    _mp4_pose = _mp4_mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    print("[INFO] 4단계 MediaPipe Pose 로드됨")
except Exception as e:
    print(f"[WARN] 4단계 MediaPipe Pose 로드 실패 ({e})")
    _mp4_pose = None

# ======================================================================
# MediaPipe Pose (덮개 전용)
# ======================================================================
# ======================================================================
# MediaPipe Pose (덮개 전용)
# ======================================================================
pose_landmarker = None
try:
    # Mac Metal GPU 충돌 방지 — mediapipe가 Metal delegate를 시도하다 abort 발생하는 문제
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "false")
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
CONF_TH = {"inside": 0.70, "side": 0.70, "top": 0.70}  # [v9] side 0.80→0.70
AREA_TH = 0.03
MERGE_GAP = 0.5
PARTS_PER_BOX = 3
ABSENT_RESET_SEC = 1.0  # [v9패치2] 2.5→1.0초. 부품을 들고 내려놓는 자연스러운 미검출 구간을 더 민감하게 감지
HAMMER_DISPLAY_CONF_TH = 0.60

# [수정 2 / v9 강화] 피부색 필터
# 팔뚝이 마스크 위쪽에 걸릴 때 전체 평균으론 희석돼서 필터가 안 걸림.
# → 마스크를 "상단 30%" / "전체" 두 구역으로 나눠 각각 체크.
#   둘 중 하나라도 피부색 비율 임계 넘으면 팔뚝으로 판단.
SKIN_FILTER_ON = True

def is_skin_region(frame_bgr, mask_bin, skin_ratio_th=0.40):
    """
    마스크 영역 픽셀 중 피부색 비율 검사.
    전체 or 상단30% 중 하나라도 임계 초과 → True(팔뚝으로 판단).
    skin_ratio_th: 기본 0.40 (기존 0.55보다 낮춰 팔뚝 필터 민감도 상승)
    """
    H_f = frame_bgr.shape[0]

    def _skin_ratio(ys, xs):
        if len(xs) < 30:
            return 0.0
        sample = frame_bgr[ys, xs]
        hsv = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        # S 범위를 20~200으로 확장 — 장갑 경계 부분 피부도 잡힘
        skin = (((h <= 25) | (h >= 160)) & (s >= 20) & (s <= 200) & (v >= 60))
        return float(skin.mean())

    # 전체 마스크
    ys_all, xs_all = np.where(mask_bin > 0)
    if len(xs_all) < 50:
        return False
    ratio_all = _skin_ratio(ys_all, xs_all)

    # 상단 30% 마스크만 (팔뚝이 걸리는 구역)
    top_y = int(H_f * 0.30)
    mask_top = mask_bin.copy()
    mask_top[top_y:, :] = 0
    ys_top, xs_top = np.where(mask_top > 0)
    ratio_top = _skin_ratio(ys_top, xs_top)

    # 둘 중 하나라도 임계 초과 → 팔뚝
    return ratio_all >= skin_ratio_th or ratio_top >= (skin_ratio_th - 0.10)

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
    SIDE_GATE_SEC = 0.15   # [v9패치2] side 카운트 시작을 위한 inside 최소 누적 시간

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
    def _inside_gate_ok(self): return self.acc_inside >= self.SIDE_GATE_SEC

    def update(self, view, t):
        # 여기 들어오는 view 는 "표시 기준(=판정 기준) 통과한" 것만.
        if view == "inside":
            self.acc_inside += self._accum("inside", t)
            return "accepted"
        elif view == "side":
            # [v9패치2] inside 완전완료(0.3초) 대신 0.15초 이상 누적되면 side 카운트 시작.
            # 최종 완료(done)는 여전히 둘 다 0.3초 채워야 하므로 판정 기준 자체는 그대로.
            if not self._inside_gate_ok():
                return "waiting_inside"   # 순서(inside가 먼저)는 여전히 강제
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
        self.step4 = [Step4Checker() for _ in range(n_parts)]   # [v8] 4단계
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
        """제품 k 가 1·2·3·4단계 모두 완료인지 (4단계 모듈 없으면 3단계까지)"""
        base = (self.parts.step1[k].done()
                and hinge_state == "완료"
                and self.parts.step3_done[k])
        if glove_model is None or not _STEP4_UTILS_OK:
            return base
        return base and self.parts.step4[k].done()

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
            if (glove_model is not None and _STEP4_UTILS_OK
                    and not self.parts.step4[k].done()):
                missing.append("4단계: 윗면 육안 확인")
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

def _largest_component(mask_bin):
    """마스크에서 가장 큰 연결 컴포넌트만 남기고 반환. 전체의 30% 미만인 작은 조각은 제거."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask_bin
    # 배경(0) 제외, 넓이 기준 정렬
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    out = np.zeros_like(mask_bin)
    for i, area in enumerate(areas, start=1):
        if area >= max_area * 0.30:   # 최대 조각의 30% 이상만 남김
            out[labels == i] = 1
    return out

def detect_seg(frame):
    """[수정 1] 원본 마스크 그대로 + [v7] 최대 연결 컴포넌트만 남겨 조각 튀는 현상 방지."""
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
        mask_bin = (mask > 0.5).astype(np.uint8)
        # [v7] 최대 연결 컴포넌트만 남김 — 멀리 떨어진 작은 조각 제거
        mask_overlay = _largest_component(mask_bin)
        area_ratio = mask_overlay.sum() / (W * H)

    # [수정 2] 피부색(팔) 필터
    if SKIN_FILTER_ON and mask_overlay is not None and view in ("inside", "side"):
        if is_skin_region(frame, mask_overlay):
            return "none", 0.0, 0.0, None

    return view, conf, area_ratio, mask_overlay

# ======================================================================
# 4단계 체커
# ======================================================================
STEP4_PAIR_HOLD   = 0.20   # 양손 장갑 감지 유지 최소 초
STEP4_POSE_HOLD   = 0.20   # 팔-장갑 연결 유지 최소 초
STEP4_TOP_DIRECT  = 0.20   # direct top 누적 초
STEP4_TOP_REVIEW  = 0.40   # review top 누적 초 (장갑/팔 근거 필요)
STEP4_RESET_GAP   = 2.50   # 비활성 리셋 간격 초

def _pose_arm4(pose_result, w, h, arm):
    """4단계용 MediaPipe Pose → 팔 관절 좌표 추출."""
    import math as _math
    if pose_result is None or not pose_result.pose_landmarks:
        return None
    ids = (11, 13, 15) if arm == 'L' else (12, 14, 16)
    pts = []
    for idx in ids:
        lm = pose_result.pose_landmarks.landmark[idx]
        th = 0.35 if idx in (13, 14) else 0.45
        inside = -0.08 <= lm.x <= 1.08 and -0.08 <= lm.y <= 1.08
        pt = (float(lm.x * w), float(lm.y * h)) if lm.visibility >= th and inside else None
        pts.append((pt, float(lm.visibility)))
    return pts

def _arm_glove_score4(arm_pose, glove, diag):
    import math as _math
    import numpy as _np4
    if arm_pose is None or arm_pose[1][0] is None:
        return None
    elbow = arm_pose[1][0]
    shoulder = arm_pose[0][0]
    wrist_pt = arm_pose[2][0]
    g = glove['center']
    d = _math.dist(elbow, g) / diag
    if not (0.025 <= d <= 0.38):
        return None
    score = d
    if wrist_pt is not None:
        wd = _math.dist(wrist_pt, g) / diag
        score -= max(0.0, 0.055 - wd) * 0.65
    if shoulder is not None:
        a = _np4.array(elbow) - _np4.array(shoulder)
        b = _np4.array(g)     - _np4.array(elbow)
        na, nb = float(_np4.linalg.norm(a)), float(_np4.linalg.norm(b))
        if na > 1e-6 and nb > 1e-6:
            cos = float(_np4.dot(a, b) / (na * nb))
            score += 0.10 if cos < -0.55 else (0.04 if cos < -0.10 else 0.0)
    score -= min(0.03, glove['conf'] * 0.015)
    return score

def _fuse_pose_gloves4(lp, rp, gloves, pair, w, h):
    import math as _math
    diag = _math.hypot(w, h)
    cands = list(pair[1:3]) if pair is not None else sorted(gloves, key=lambda d: d['conf'], reverse=True)[:4]
    out = {a: {'glove': None, 'proxy': None} for a in ('L', 'R')}
    options = []
    for arm, pose in [('L', lp), ('R', rp)]:
        for gi, g in enumerate(cands):
            s = _arm_glove_score4(pose, g, diag)
            if s is not None:
                options.append((s, arm, gi))
    used_a, used_g = set(), set()
    for _, arm, gi in sorted(options):
        if arm in used_a or gi in used_g:
            continue
        out[arm]['glove'] = cands[gi]
        out[arm]['proxy'] = cands[gi]['center']
        used_a.add(arm); used_g.add(gi)
    return out

class Step4Checker:
    """4단계(비닐 제거 후 윗면 육안 확인) 판정기. 부품 1개 단위로 리셋."""
    WAITING  = "WAITING"
    ACTIVE   = "ACTIVE"
    COMPLETE = "COMPLETE"

    def __init__(self):
        self.reset()

    def reset(self):
        self.state          = self.WAITING
        self.ev_pair        = False
        self.ev_pose        = False
        self.ev_top         = False
        self.top_mode       = "NONE"   # NONE / DIRECT / REVIEW
        self.pair_streak    = 0
        self.pose_streak    = 0
        self.top_direct_f   = 0
        self.top_review_f   = 0
        self.last_relevant  = None     # 마지막 활성 프레임 t

    def done(self):
        return self.state == self.COMPLETE

    def feed(self, t, fps, status, has_pair, two_arm_links, direct, review_supported):
        """매 프레임 호출. status: step4_glove_top_v3의 segment_status 반환값."""
        top_candidate  = status in ("TOP_DIRECT", "TOP_REVIEW")
        hand_activity  = has_pair or (two_arm_links)
        relevant_now   = top_candidate or hand_activity or status == "PLASTIC_RISK"

        if relevant_now:
            self.last_relevant = t

        # WAITING → ACTIVE
        if self.state == self.WAITING and top_candidate and hand_activity:
            self.state = self.ACTIVE
            self.ev_pair = self.ev_pose = self.ev_top = False
            self.top_mode = "NONE"
            self.pair_streak = self.pose_streak = self.top_direct_f = self.top_review_f = 0

        if self.state in (self.ACTIVE, self.COMPLETE):
            self.pair_streak = self.pair_streak + 1 if has_pair else 0
            self.pose_streak = self.pose_streak + 1 if two_arm_links else 0
            if direct is not None:
                self.top_direct_f += 1
            if review_supported:
                self.top_review_f += 1

            if not self.ev_pair and self.pair_streak / fps >= STEP4_PAIR_HOLD:
                self.ev_pair = True
            if self.ev_pair and not self.ev_pose and self.pose_streak / fps >= STEP4_POSE_HOLD:
                self.ev_pose = True
            if self.ev_pair and self.ev_pose and not self.ev_top:
                if self.top_direct_f / fps >= STEP4_TOP_DIRECT:
                    self.ev_top = True; self.top_mode = "DIRECT"
                elif self.top_review_f / fps >= STEP4_TOP_REVIEW:
                    self.ev_top = True; self.top_mode = "REVIEW"

            if self.ev_pair and self.ev_pose and self.ev_top:
                self.state = self.COMPLETE

            # 비활성 리셋
            if self.last_relevant is not None and (t - self.last_relevant) > STEP4_RESET_GAP:
                if self.state != self.COMPLETE:
                    self.reset()

    def phase(self):
        if self.state == self.COMPLETE:
            return "STEP4_DONE"
        if self.state == self.ACTIVE:
            if self.ev_pair and self.ev_pose:
                return "WAIT_TOP"
            if self.ev_pair:
                return "WAIT_POSE"
            return "WAIT_PAIR"
        return "IDLE"

def passes(view, conf, area):
    """표시 = 판정 통일 기준."""
    return (view in CONF_TH and conf >= CONF_TH[view] and area >= AREA_TH)

# [v7] 미검수 의심 경고용 상태 — 부품당 "seg 미감지 누적 시간" 추적
_suspicious_absent: dict = {}   # {part_idx: absent_sec}
_suspicious_last_t: dict = {}   # {part_idx: last_frame_t}
SUSPICIOUS_ABSENT_TH = 5.0     # 이 초 이상 암레스트 미감지면 경고

def _draw_suspicious_warning(draw, canvas, cyc, W, H, pw, seg_view, passes_fn, seg_conf, seg_area, txt_fn):
    """[v7] 검사 중인데 암레스트가 오랫동안 감지 안 되면 우하단에 빨간 경고."""
    global _suspicious_absent, _suspicious_last_t

    if cyc.state != Cycle.INSPECTING:
        _suspicious_absent.clear()
        _suspicious_last_t.clear()
        return

    part_k = cyc.parts.current()
    if part_k is None:
        return

    # 이미 1단계 완료된 부품은 경고 대상 아님
    if cyc.parts.step1[part_k].done():
        _suspicious_absent[part_k] = 0.0
        return

    import time as _time_mod
    now_real = _time_mod.time()
    last = _suspicious_last_t.get(part_k, None)

    seg_visible = passes_fn(seg_view, seg_conf, seg_area) and seg_view in ("inside", "side")

    if last is not None:
        delta = min(now_real - last, 0.5)  # 최대 0.5초씩만 누적 (pause 대비)
        if not seg_visible:
            _suspicious_absent[part_k] = _suspicious_absent.get(part_k, 0.0) + delta
        else:
            _suspicious_absent[part_k] = 0.0  # 감지되면 리셋
    _suspicious_last_t[part_k] = now_real

    absent_sec = _suspicious_absent.get(part_k, 0.0)
    if absent_sec < SUSPICIOUS_ABSENT_TH:
        return

    # 우하단 빨간 경고 박스
    bx1, by1 = W + 12, H - 130
    bx2, by2 = W + pw - 12, H - 10
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
    txt_fn((bx1 + 10, by1 + 10), "⚠ 미검수 의심", 18, (0, 200, 255), True)
    txt_fn((bx1 + 10, by1 + 38), f"부품 {part_k+1}: 암레스트", 14, (255, 200, 200))
    txt_fn((bx1 + 10, by1 + 58), f"미감지 {absent_sec:.0f}초 이상", 14, (255, 200, 200))
    txt_fn((bx1 + 10, by1 + 82), "카메라 각도 확인 필요", 13, (200, 200, 255))

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
           show_seg, hinge_boxes, hinge_hb, hinge_state_live,
           step4_status="NO_TOP_EVIDENCE", step4_has_pair=False,
           display_part_k=None, display_hinge_state=None):
    canvas = np.zeros((H, W + pw, 3), dtype=np.uint8)
    disp = frame.copy()

    # 덮개 마스크 — 항상
    if cover.last_mask is not None:
        Hf, Wf = frame.shape[:2]
        cmask = cv2.resize(cover.last_mask.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
        disp[cmask > 0] = (disp[cmask > 0] * 0.55 + np.array([200, 200, 0]) * 0.45).astype(np.uint8)
        cnts_c, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts_c, -1, (0, 220, 220), 2)

    # [v9] 체크리스트 표시용 part_k — 부품 전환 중에도 마지막 검사 부품 유지
    # display_part_k 가 None이면 현재 활성 부품으로 폴백
    part_k = display_part_k if display_part_k is not None else (
        cyc.parts.current() if cyc.state == Cycle.INSPECTING else None
    )
    # 힌지 상태도 마찬가지로 유지된 값 우선 사용
    if display_hinge_state is not None:
        hinge_state_live = display_hinge_state

    # 1단계 seg 마스크 — 검사 중 + 통일 기준 통과시만
    if (cyc.state == Cycle.INSPECTING and show_seg
            and seg_mask is not None and seg_view in SEG_MASK_COLORS):
        color = SEG_MASK_COLORS[seg_view]
        disp[seg_mask > 0] = (disp[seg_mask > 0] * 0.45 + color * 0.55).astype(np.uint8)
        cnts, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, tuple(int(x) for x in color), 3)

    # [v7] 힌지 bbox — 1단계 완료 전엔 표시 안 함.
    #       1단계 완료 후에도 "손목이 힌지와 겹치는(held)" 경우만 표시.
    part_k_for_hinge = cyc.parts.current() if cyc.state == Cycle.INSPECTING else None
    step1_done = (part_k_for_hinge is not None and cyc.parts.step1[part_k_for_hinge].done())
    if cyc.state == Cycle.INSPECTING and step1_done:
        Hf, Wf = frame.shape[:2]
        for b in hinge_boxes:
            is_held = hinge_hb is not None and np.allclose(b, hinge_hb)
            if not is_held:
                continue   # 손목이 안 닿은 힌지는 표시 생략
            cv2.rectangle(disp, (int(b[0]*Wf), int(b[1]*Hf)), (int(b[2]*Wf), int(b[3]*Hf)), (0, 0, 255), 3)

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

    # 체크리스트 (현재 부품) — 4단계는 모듈 유무 상관없이 항상 표시
    _s4_done = (part_k is not None and glove_model is not None
                and _STEP4_UTILS_OK and cyc.parts.step4[part_k].done())
    checks = [
        ("1", "안쪽면/옆면 확인", cyc.parts.step1[part_k].done() if part_k is not None else False),
        ("2", "Hinge 작동 확인", hinge_state_live == "완료"),
        ("3", "옆면 Hammer 확인", cyc.parts.step3_done[part_k] if part_k is not None else False),
        ("4", "윗면 육안 확인",   _s4_done),
    ]
    _cv2_gap = 62   # 4단계 항목 항상 있으므로 간격 고정
    y = 148
    for _, _, done in checks:
        y += _cv2_gap
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
    _step4_active = glove_model is not None and _STEP4_UTILS_OK
    y = 148
    for code, label, done in checks:
        y += 62
        sc = (100,255,150) if done else (180,180,180)
        txt((W+75, y-18), f"{code}단계", 17, (200,200,200), True)
        txt((W+75, y+2), label, 15, (225,225,225))
        if code == "4":
            if not _step4_active:
                # 모델 미로드여도 항목은 보임 — 회색으로 "대기"
                txt((W+75, y+24), "대기(모델 미로드)", 13, (130,130,130))
            elif done:
                txt((W+75, y+24), "✓ 완료", 14, sc, True)
            elif part_k is not None:
                phase4 = cyc.parts.step4[part_k].phase()
                phase4_label = {
                    "IDLE": "대기", "WAIT_PAIR": "장갑 감지 중",
                    "WAIT_POSE": "팔 연결 확인 중", "WAIT_TOP": "윗면 확인 중"
                }.get(phase4, "대기")
                txt((W+75, y+24), phase4_label, 14, (200,200,100))
            else:
                txt((W+75, y+24), "미확인", 14, sc, True)
        else:
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

    # [v7] 미검수 의심 실시간 경고 — 우하단 빨간박스
    # 조건: INSPECTING 상태이고 현재 부품에서 1단계 미완료 + seg 장시간 미감지
    _draw_suspicious_warning(d, canvas, cyc, W, H, pw, seg_view, passes, seg_conf, seg_area, txt)
    # (passes 는 모듈 레벨 함수이므로 직접 참조 가능)

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
    # [v9] 체크리스트 유지용 — 마지막으로 검사 중이던 부품 상태 기억
    _display_part_k    = None   # 체크리스트에 표시할 부품 인덱스
    _display_hinge_st  = "대기"  # 체크리스트에 표시할 힌지 상태

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

        # [v8] 4단계 처리 변수 초기화
        step4_status = "NO_TOP_EVIDENCE"
        step4_direct = None
        step4_review_sup = False
        step4_has_pair = False
        step4_two_arm  = False

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

            # [v8] 4단계 처리
            if glove_model is not None and _STEP4_UTILS_OK and _mp4_pose is not None:
                part_k4 = cyc.parts.current()
                if part_k4 is not None:
                    # seg 상태(top_id / side_id 재활용)
                    top_id4  = next((k for k, v in SEG_NAMES.items() if v == "top"),  None)
                    side_id4 = next((k for k, v in SEG_NAMES.items() if v == "side"), None)
                    _s4_result = seg_model(frame, conf=0.20, verbose=False, device=DEVICE)[0]
                    # 4단계 전용 anchor (부품 단위로 리셋하지 않고 Checker 내부에서 관리)
                    _s4_anchor = getattr(cyc, "_step4_anchor", None)
                    _s4_seed   = getattr(cyc, "_step4_seed", None)
                    step4_status, step4_direct, _s4_white, _s4_side, _s4_anchor, _s4_seed = \
                        _seg_status4(frame, _s4_result, top_id4, side_id4, _s4_anchor, _s4_seed, fi, fps, W, H)
                    cyc._step4_anchor = _s4_anchor
                    cyc._step4_seed   = _s4_seed

                    # 장갑 검출
                    _gr4 = glove_model.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
                    _s4_elapsed = (fi - getattr(cyc, "_step4_last_glove", -999)) / fps
                    _s4_prior   = getattr(cyc, "_step4_glove_prior", None) if _s4_elapsed <= 0.32 else None
                    _s4_all_g, _s4_pair = find_pair(_gr4.boxes, W, H, _s4_prior, _s4_elapsed)
                    step4_has_pair = _s4_pair is not None
                    if step4_has_pair:
                        cyc._step4_glove_prior = (_s4_pair[1]['center'], _s4_pair[2]['center'])
                        cyc._step4_last_glove  = fi
                    else:
                        if _s4_elapsed > 0.32:
                            cyc._step4_glove_prior = None

                    # MediaPipe Pose
                    _rgb4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _rgb4.flags.writeable = False
                    _pr4 = _mp4_pose.process(_rgb4)
                    _lp4 = _pose_arm4(_pr4, W, H, 'L')
                    _rp4 = _pose_arm4(_pr4, W, H, 'R')
                    _fused4 = _fuse_pose_gloves4(_lp4, _rp4, _s4_all_g, _s4_pair, W, H)
                    step4_two_arm = (_fused4['L']['proxy'] is not None and _fused4['R']['proxy'] is not None)
                    step4_review_sup = (step4_status == "TOP_REVIEW") and (step4_has_pair or step4_two_arm)

                    cyc.parts.step4[part_k4].feed(
                        t, fps, step4_status, step4_has_pair, step4_two_arm,
                        step4_direct, step4_review_sup,
                    )

            # [수정 6,7] 제품 완료 카운트 실시간 갱신
            cyc.update_part_counts(live_states)
            # [v9패치2] 완료 즉시 강제 전환은 롤백함 — 4단계 등 일부 단계가
            # 누락된 채로 완료 처리될 위험이 있음. 대신 아래 ABSENT_RESET_SEC을
            # 단축해서 "부품을 들고 제자리에 내려놓는" 자연스러운 미검출 구간을
            # 더 민감하게 잡도록 함 (on_presence 로직, PartTracker 클래스 참고).

            closed, _ = cover.verdict_close()
            if closed:
                cyc.finish_box()
                cover.reset()

        # [v9] 체크리스트 유지: 검사 중이면 현재 값으로 갱신, 아니면 마지막 값 유지
        if cyc.state == Cycle.INSPECTING:
            _cur = cyc.parts.current()
            if _cur is not None:
                _display_part_k   = _cur
                _display_hinge_st = hinge_state_live

        canvas = render(frame, W, H, pw, cyc, seg_view, seg_conf, seg_area, seg_mask,
                        show_seg, hinge_boxes, hinge_hb, hinge_state_live,
                        step4_status=step4_status, step4_has_pair=step4_has_pair,
                        display_part_k=_display_part_k,
                        display_hinge_state=_display_hinge_st)
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
    VIDEO_DIR = f"{BASE}/data/media/cycle_mp4"
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # 파일명만 줬으면 기본 영상 폴더에서 찾기
        if not os.path.isabs(arg) and not os.path.exists(arg):
            arg = f"{VIDEO_DIR}/{arg}"
        src = arg
    else:
        src = f"{VIDEO_DIR}/crop020.mp4"
    # 출력 파일명은 입력 영상명 기반으로 자동 생성
    stem = os.path.splitext(os.path.basename(src))[0]
    out = f"{BASE}/model/result_{stem}_v9.mp4"
    run(src, out)
    subprocess.run(["open", out])