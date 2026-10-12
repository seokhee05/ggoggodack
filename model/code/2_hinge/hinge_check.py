"""
힌지 검사 판정 코드 (hinge_check.py)

판정 방식
  1) 학습한 힌지 검출 모델(best.pt)로 매 프레임 힌지 위치(네모)를 찾음
  2) YOLO-pose 로 손목 위치를 찾고 (POSE_BACKEND="mediapipe" 로 바꾸면 미디어파이프 사용), 손목이 힌지 네모 근처에 있으면 "힌지를 쥐고 있음"
  3) 쥐고 있는 동안 약 1.5초 창에서
       - 힌지 네모의 모양(세로/가로 비율)이 크게 바뀌고   -> 힌지가 꺾임
       - 힌지 네모가 위아래로 여러 번 방향을 바꾸면        -> 왔다갔다 구부림
     두 조건을 모두 만족하면 "구부림 중"
  4) 두 손 조건: 두 손목이 같은 방향으로 함께 움직이면(재포장·옮기기) 구부림에서 제외
  5) 자리 추적: 박스 안 암레스트 모델로 자리 1·2·3 (왼쪽부터)을 등록하고, 비는 자리를 보고
     어느 제품을 꺼내 검사했는지 구분 (꺼냈는데 생략 / 안 꺼냄도 자리 번호로 표시).
     암레스트 모델이 없거나 자리가 3개 안 잡히면 예전처럼 순서대로 1, 2, 3 을 셈
  6) (참고) 영상 하나 = 박스 하나(암레스트 3개) 기준. 암레스트 1개만 찍힌 짧은 클립은 FAIL 로 나옴
  7) 박스 판정: 구부림이 확인된 제품이 3개면 PASS, 부족하면 FAIL,
     힌지가 거의 안 보였으면 REVIEW(사람 확인 필요)

사용법
  python hinge_check.py                 -> VIDEO_FOLDER 안의 영상 전부 판정
  python hinge_check.py crop020.mp4     -> 영상 하나만

결과
  hinge_check_results.csv   영상별 판정 요약
  hinge_check_events.csv    찾은 구부림 구간 목록
  (SAVE_VIDEO=True 이면) hinge_check_out/영상이름_result.mp4  네모와 판정이 그려진 영상
  정답표(힌지_정답표.csv)가 있으면 정답과 자동 비교해서 화면에 출력
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# ====== 환경에 맞게 수정 ======
HINGE_MODEL = r"hinge_det_runs\round2\weights\best.pt"   # 학습한 힌지 모델 (없으면 round1 자동 사용)
HINGE_MODEL_FALLBACK = r"hinge_det_runs\round1\weights\best.pt"
POSE_BACKEND = "yolo"                                     # "mediapipe" 또는 "yolo"
MP_POSE_MODEL = "pose_landmarker_heavy.task"              # 미디어파이프 모델 (라벨링 폴더에 있음. 느리면 pose_landmarker.task)
YOLO_FALLBACK = True                                      # 미디어파이프가 사람을 못 찾은 프레임은 YOLO-pose 로 보충
POSE_MODEL = "yolov8n-pose.pt"
VIDEO_FOLDER = r"hinge_clips"                             # 판정할 영상 폴더
GT_CSV = r"힌지_정답표.csv"                                # 정답표 (없으면 비교 생략)
SAVE_VIDEO = True                                         # 결과 영상 자동 저장 (False 면 저장 안 함)
SIMPLE_UI = True          # True: 화면 체크리스트를 '대기 / 완료'(끝나면 '누락')만 보여줌. 내부 판정(잡음·분석)은 그대로
SHOW_PREVIEW = True                                       # True 면 처리 화면 보기 (q = 이 영상 건너뛰기, d = 전체 종료)

ANALYZE_FPS = 10          # 초당 몇 프레임 분석할지 (CPU 부담 조절)
HINGE_CONF = 0.3          # 힌지 검출 확신도 기준 (0.2 는 휴대폰 등을 힌지로 잘못 잡음)
KPT_CONF = 0.3            # 손목 키포인트 확신도 기준
HOLD_MARGIN = 0.3         # 손목이 힌지 네모에서 (네모 크기 x 이 값) 이내면 "쥐고 있음" (예전 0.6 은 너무 느슨)
STATIC_SEC = 3.0          # 이 시간 동안 거의 같은 자리에 있던 힌지 = 박스 안에 놓인 힌지 -> "쥐고 있음"에서 제외
STATIC_RATIO = 0.7        #   (최근 STATIC_SEC 중 이 비율 이상 같은 자리(IoU>=STATIC_IOU)에 보였으면 놓인 것으로 봄)
STATIC_IOU = 0.7
TRACK_HELD = False         # True: 한 번 잡은 힌지를 계속 따라감 (영상 13개 실험에서 오히려 나빠져서 꺼 둠)
TRACK_KEEP_SEC = 0.5       # 한 번 잡은 힌지를 따라감. 그 힌지가 이 시간 이상 안 보일 때만 다른 힌지로 바꿈

WIN_SEC = 0.7             # 앞뒤 이 시간(초)만큼을 한 창으로 봄 (총 약 1.5초)
MIN_OBS = 5               # 창 안에 "쥐고 있음" 프레임이 이 개수 이상일 때만 판정
AR_STD_TH = 0.10          # 힌지 네모 모양 변화 기준 (클수록 엄격)
ZC_TH = 2                 # 위아래 방향 전환 횟수 기준
ZC_AMP = 0.03             # 방향 전환으로 칠 최소 움직임 (힌지 크기 대비)

EVENT_MERGE_SEC = 0.5     # 구부림 구간 사이가 이 시간 이내면 하나로 합침
EVENT_MIN_SEC = 0.3       # 이보다 짧은 구부림은 무시
BEND_MIN_RANGE = 0.5      # 구부림 구간 동안 힌지가 위아래로 움직인 폭(힌지 크기 대비) 최소값. 살짝 흔들린 건 제외
WRIST_ZC_TH = 2           # [구부림 후보] 쥔 손목이 위아래 방향을 바꾼 횟수 최소값 (1.5초 창)
WRIST_ZC_AMP = 0.1        #   방향 전환으로 칠 최소 움직임 (힌지 크기 대비)
WRIST_MIN_RANGE = 0.3     #   쥔 손목 위아래 폭 최소값 (힌지 크기 대비). 잡고만 있으면 구부림 아님
SHAPE_SIM_MAX = 0.1       # [힌지 꺾임 확인] 구부림 동안 힌지 사진이 처음 모습과 이 값 이하로 달라져야 함(유사도 -1~1)
                          #   재포장처럼 쥐고 흔들기만 하면 힌지 모습이 그대로라 유사도가 높음(0.15 이상)
AR_STD_MAX = 0.30         # 힌지 모양 변화가 이보다 크면 검출이 튀는 것(다른 물체/다른 힌지로 바뀜)으로 보고 제외
PRODUCT_MIN_BEND_SEC = 0.8  # 제품 1개를 '완료'로 치려면 구부림 시간 합이 이 이상이어야 함 (짧은 가짜 구부림 방지)
ARMREST_GAP_SEC = 3.0     # 앞 구부림이 끝나고 이 시간 안에 다시 구부리면 같은 제품으로 봄
GROUP_MAX_SEC = 10.0      # 한 제품 구부림 묶음의 최대 길이(초). 넘으면 다음 제품
ARMRESTS_PER_BOX = 3      # 한 박스당 암레스트 개수
MIN_HINGE_SEEN = 0.10     # 힌지가 보인 프레임 비율이 이보다 낮으면 REVIEW

# 두 손 조건 (재포장·옮기기 오탐 줄이기)
#  진짜 구부림: 한 손은 제품을 받치고 다른 손이 힌지를 위아래로 비틂 -> 두 손 움직임이 서로 다름
#  재포장/옮기기: 두 손이 같은 방향으로 함께 움직임 -> 이 경우는 구부림으로 안 침
TWO_HAND_CHECK = True
TWO_HAND_CORR_MAX = 0.45  # 두 손목 움직임 방향 일치도(-1~1). 이보다 크면 "두 손이 같이 움직임" = 구부림 아님

# 자리(칸) 추적: 박스 안 암레스트 위치로 어느 제품을 검사했는지 구분
ARMREST_MODEL = r"armrest_box_runs\box2\weights\best.pt"   # 박스 안 암레스트 모델 (없으면 순서대로 셈)
SLOT_EVERY = 5            # 분석 프레임 몇 장마다 암레스트 검사 (5 = 초당 2번)
SLOT_CONF = 0.3
SLOT_EMPTY_SEC = 3.0      # 다른 자리는 보이는데 이 자리만 이 시간 이상 비면 "꺼냄"
SLOT_SLACK = 0.5          # 구부림이 자리가 비기 이 시간 전에 시작해도 그 자리로 인정
SLOT_CONTINUE = False      # True: 앞 구부림 바로 뒤 구부림은 같은 자리로
SLOT_OVERLAP = 0.4        # 자리와 네모가 가로로 이만큼 겹치면 "있음"
# ============================

LEFT_WRIST, RIGHT_WRIST = 9, 10   # COCO 키포인트 번호


# 미디어파이프 33개 랜드마크 -> 판정에 쓰는 COCO 번호로 변환 (어깨/팔꿈치/손목/골반)
MP_TO_COCO = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24}


class PoseEstimator:
    """미디어파이프 PoseLandmarker (영상 모드). 못 찾은 프레임은 YOLO-pose 로 보충"""

    def __init__(self):
        self.yolo = YOLO(POSE_MODEL) if (POSE_BACKEND == "yolo" or YOLO_FALLBACK) else None
        self.mp = None
        if POSE_BACKEND == "mediapipe":
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._mp = mp
            self._opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MP_POSE_MODEL),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self._vision = vision
            self.stats = {"mediapipe": 0, "yolo": 0, "none": 0}

    def new_video(self):
        """영상마다 새로 만들어야 함 (미디어파이프 영상 모드는 시간이 계속 증가해야 해서)"""
        if POSE_BACKEND == "mediapipe":
            if self.mp is not None:
                self.mp.close()
            self.mp = self._vision.PoseLandmarker.create_from_options(self._opts)
            self.stats = {"mediapipe": 0, "yolo": 0, "none": 0}

    def __call__(self, frame, t_ms):
        """반환: (kp COCO 17x3 [x,y,확신도], mp_landmarks 33x3 또는 None, 사용한 방식)"""
        h, w = frame.shape[:2]
        if POSE_BACKEND == "mediapipe":
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = self.mp.detect_for_video(img, int(t_ms))
            if res.pose_landmarks:
                lm = np.array([[p.x, p.y, p.visibility] for p in res.pose_landmarks[0]])
                kp = np.zeros((17, 3))
                for c, m in MP_TO_COCO.items():
                    kp[c] = lm[m]
                self.stats["mediapipe"] += 1
                return kp, lm, "mediapipe"
            if not YOLO_FALLBACK:
                self.stats["none"] += 1
                return None, None, "none"
        p = self.yolo.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
        if p.keypoints is not None and len(p.keypoints.xy):
            i = int((p.boxes.xywh[:, 2] * p.boxes.xywh[:, 3]).argmax())   # 가장 큰(가까운) 사람
            xy = p.keypoints.xy[i].cpu().numpy() / [w, h]
            cf = p.keypoints.conf[i].cpu().numpy()[:, None]
            if POSE_BACKEND == "mediapipe":
                self.stats["yolo"] += 1
            return np.concatenate([xy, cf], 1), None, "yolo"
        if POSE_BACKEND == "mediapipe":
            self.stats["none"] += 1
        return None, None, "none"


def detect_hinges(frame, hinge_model):
    """한 프레임에서 힌지 네모들을 찾음 (좌표는 0~1 비율)"""
    h, w = frame.shape[:2]
    r = hinge_model.predict(frame, conf=HINGE_CONF, imgsz=640, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy() / [w, h, w, h] if len(r.boxes) else np.zeros((0, 4))


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StaticFilter:
    """박스 안에 가만히 놓인 힌지를 찾음. 사람이 쥐고 있는 힌지는 움직이니까 같은 자리에 오래 안 있음"""

    def __init__(self):
        self.n = int(STATIC_SEC * ANALYZE_FPS)
        self.hist = []

    def __call__(self, boxes):
        flags = []
        for b in boxes:
            same = sum(1 for bs in self.hist if any(_iou(b, x) >= STATIC_IOU for x in bs))
            flags.append(len(self.hist) >= self.n // 2 and same >= STATIC_RATIO * len(self.hist))
        self.hist = (self.hist + [list(boxes)])[-self.n:]
        return flags


class HeldTracker:
    """한 번 잡은 힌지를 계속 따라감. 손이 옆 힌지에 가까워져도 바로 옆 힌지로 바꾸지 않음"""

    def __init__(self):
        self.prev = None
        self.prev_t = -99.0

    def __call__(self, t, boxes, kp, static=None):
        if static is None:
            static = [False] * len(boxes)
        cand = [b for b, st in zip(boxes, static) if not st]
        if self.prev is not None and t - self.prev_t <= TRACK_KEEP_SEC:
            p = self.prev
            size = max(p[2] - p[0], p[3] - p[1])
            pc = np.array([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
            best = None
            for b in cand:       # 전에 잡던 힌지와 가장 가까운 네모 (힌지 크기 이내)
                d = np.hypot(*(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - pc))
                if d <= 1.5 * size and (best is None or d < best[0]):
                    best = (d, b)
            if best is not None and held_box([best[1]], kp) is not None:
                self.prev, self.prev_t = best[1], t
                return best[1]
            return None      # 잠깐 놓친 것 -> 다른 힌지로 바꾸지 않고 기다림
        hb = held_box(cand, kp)
        if hb is not None:
            self.prev, self.prev_t = hb, t
        return hb


def held_box(boxes, kp, static=None):
    """손목과 가장 가까운(쥐고 있는) 힌지 네모 하나. 없으면 None.
    static: 박스 안에 놓인 힌지 표시 (True 인 힌지는 제외)"""
    if kp is None or len(boxes) == 0:
        return None
    if static is None:
        static = [False] * len(boxes)
    best = None
    for wi in (LEFT_WRIST, RIGHT_WRIST):
        if kp[wi, 2] < KPT_CONF:
            continue
        x, y = kp[wi, :2]
        for b, st in zip(boxes, static):
            if st:
                continue
            m = HOLD_MARGIN * max(b[2] - b[0], b[3] - b[1])
            if b[0] - m <= x <= b[2] + m and b[1] - m <= y <= b[3] + m:
                d = np.hypot(x - (b[0] + b[2]) / 2, y - (b[1] + b[3]) / 2)
                if best is None or d < best[0]:
                    best = (d, b)
    return None if best is None else best[1]


def direction_changes(x, amp):
    d = np.diff(x)
    d = d[np.abs(d) > amp]
    return int(np.sum(np.diff(np.sign(d)) != 0)) if len(d) > 1 else 0


def bend_flags(times, held):
    """프레임마다 '구부림 중'인지 계산"""
    half = max(1, int(round(WIN_SEC * ANALYZE_FPS)))
    feats = []
    for b in held:
        if b is None:
            feats.append(None)
        else:
            feats.append(((b[1] + b[3]) / 2,
                          np.log((b[3] - b[1]) / max(b[2] - b[0], 1e-3)),
                          max(b[2] - b[0], b[3] - b[1])))
    flags = np.zeros(len(times), bool)
    for i in range(len(times)):
        win = [feats[j] for j in range(max(0, i - half), min(len(times), i + half + 1)) if feats[j] is not None]
        if len(win) < MIN_OBS:
            continue
        cy = np.array([f[0] for f in win])
        ar = np.array([f[1] for f in win])
        size = np.median([f[2] for f in win])
        if np.std(ar) > AR_STD_TH and direction_changes(cy / size, ZC_AMP) >= ZC_TH:
            flags[i] = True
    return flags


def two_hand_check(times, held, kps, e):
    """구부림 구간 e 동안 두 손 움직임 검사. (통과여부, 일치도, 거리변화) 반환.
    두 손목이 다 보이는 프레임이 5개 미만이면 판단 불가 -> 통과로 처리"""
    g, o, size = [], [], []
    for t, b, k in zip(times, held, kps):
        if not (e[0] - 0.3 <= t <= e[1] + 0.3) or b is None or k is None:
            continue
        if k[LEFT_WRIST, 2] < KPT_CONF or k[RIGHT_WRIST, 2] < KPT_CONF:
            continue
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        dl = np.hypot(*(k[LEFT_WRIST, :2] - c))
        dr = np.hypot(*(k[RIGHT_WRIST, :2] - c))
        gi, oi = (LEFT_WRIST, RIGHT_WRIST) if dl < dr else (RIGHT_WRIST, LEFT_WRIST)
        g.append(k[gi, :2]); o.append(k[oi, :2]); size.append(max(b[2] - b[0], b[3] - b[1]))
    if len(g) < 5:
        return True, None, None
    g, o, s = np.array(g), np.array(o), float(np.median(size))
    vg, vo = np.diff(g, axis=0), np.diff(o, axis=0)
    corr = float(np.sum(vg * vo) / (np.sqrt(np.sum(vg ** 2) * np.sum(vo ** 2)) + 1e-9))
    dstd = float(np.hypot(*(g - o).T).std() / s)
    ok = corr <= TWO_HAND_CORR_MAX
    return ok, corr, dstd


def to_events(times, flags):
    ev, start = [], None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        if not f and start is not None:
            ev.append([start, t])
            start = None
    if start is not None:
        ev.append([start, times[-1]])
    merged = []
    for e in ev:
        if merged and e[0] - merged[-1][1] <= EVENT_MERGE_SEC:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    return [e for e in merged if e[1] - e[0] >= EVENT_MIN_SEC]


def group_armrests(events):
    """구부림 이벤트를 암레스트 단위로 셈.
    앞 구부림이 끝난 뒤 ARMREST_GAP_SEC 이내에 시작한 구부림은 같은 암레스트로 묶음.
    (이전 방식은 묶음이 사슬처럼 계속 이어져 암레스트 2개가 1개로 합쳐지는 문제가 있었음)"""
    groups = []
    for e in events:
        if groups and e[0] - groups[-1][0][1] < ARMREST_GAP_SEC:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups



def _ncc(a, b):
    """두 흑백 사진의 유사도 (-1~1, 1 = 똑같음)"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def hinge_crop(frame, hb):
    """쥔 힌지 부분을 잘라 흑백 48x48 로 (앞뒤 10% 여유)"""
    if hb is None:
        return None
    H, W = frame.shape[:2]
    bw, bh = hb[2] - hb[0], hb[3] - hb[1]
    x1, y1 = int(max(0, (hb[0] - 0.1 * bw) * W)), int(max(0, (hb[1] - 0.1 * bh) * H))
    x2, y2 = int(min(W, (hb[2] + 0.1 * bw) * W)), int(min(H, (hb[3] + 0.1 * bh) * H))
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (48, 48)) if g.size else None


def _ov1(a, b):
    """가로 구간 a, b 겹침 비율 (짧은 쪽 기준)"""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


class SlotTracker:
    """박스 안 암레스트 자리(칸) 추적.
    - 처음에 암레스트가 가장 많이 보인 순간의 위치를 왼쪽부터 자리 1, 2, 3 으로 등록
    - 다른 자리는 보이는데 한 자리만 SLOT_EMPTY_SEC 이상 비면 그 자리 제품을 "꺼냄"
    - 모든 자리가 동시에 안 보이면 손/비닐에 가려진 것으로 보고 판단 보류
    - 구부림은 그 순간 꺼내져 있는 자리 중 가장 최근에 꺼낸 자리 제품으로 배정"""

    def __init__(self):
        self.slots = []        # 자리별 가로 구간 [x1, x2] (0~1)
        self.reg = []          # 자리 등록에 쓴 프레임들
        self.frozen = False    # 한 번이라도 꺼냄이 확정되면 자리 위치 고정
        self.out = []          # 자리별 비어 있던 구간 [시작, 끝(None=지금도 비어 있음)]
        self.boxes = np.zeros((0, 4))

    @staticmethod
    def _merge(xs):
        out = []
        for a in sorted(xs):
            if out and _ov1(out[-1], a) > 0.5:          # 같은 암레스트를 2개로 잡은 경우 합치기
                out[-1] = (min(out[-1][0], a[0]), max(out[-1][1], a[1]))
            else:
                out.append(tuple(a))
        return out

    def ready(self):
        return len(self.slots) >= ARMRESTS_PER_BOX

    def update(self, t, boxes):
        self.boxes = boxes
        xs = self._merge([(b[0], b[2]) for b in boxes])
        if not self.frozen and 2 <= len(xs) <= ARMRESTS_PER_BOX and len(xs) >= len(self.slots):
            if len(xs) > len(self.slots):
                self.reg = []
            self.reg.append(xs)
            self.slots = [list(v) for v in np.median(np.array(self.reg), axis=0)]
            self.out = [[] for _ in self.slots]
        if not self.slots:
            return
        seen = [any(_ov1(a, s) > SLOT_OVERLAP for a in xs) for s in self.slots]
        if not any(seen):
            return                                       # 전부 가려짐 -> 판단 보류
        for k, sk in enumerate(seen):
            iv = self.out[k][-1] if self.out[k] else None
            if sk:
                if iv is not None and iv[1] is None:
                    if t - iv[0] < SLOT_EMPTY_SEC:
                        self.out[k].pop()                # 잠깐 가려진 것 -> 없던 일로
                    else:
                        iv[1] = t                        # 제품을 다시 넣음
            else:
                if iv is None or iv[1] is not None:
                    self.out[k].append([t, None])        # 비기 시작
                elif t - iv[0] >= SLOT_EMPTY_SEC:
                    self.frozen = True

    def taken(self, k, now=None):
        """자리 k 제품을 꺼낸 적이 있는지 (SLOT_EMPTY_SEC 이상 비었던 적)"""
        for s, e in self.out[k]:
            end = e if e is not None else (now if now is not None else s)
            if end - s >= SLOT_EMPTY_SEC:
                return True
        return False

    def _real(self, k, i, now=None):
        """자리 k 의 i번째 빈 구간이 진짜 '꺼냄'인지.
        SLOT_EMPTY_SEC 이상 비었고, 다른 자리와 거의 동시에(70% 이상 겹쳐) 빈 게 아니어야 함
        (두 자리가 같이 비면 팔/비닐에 가려진 것으로 봄)"""
        s, en = self.out[k][i]
        en = en if en is not None else (now if now is not None else s + SLOT_EMPTY_SEC)
        if en - s < SLOT_EMPTY_SEC:
            return False
        for k2 in range(len(self.slots)):
            if k2 == k:
                continue
            for s2, en2 in self.out[k2]:
                en2 = en2 if en2 is not None else max(en, s2)
                inter = max(0.0, min(en, en2) - max(s, s2))
                if inter > 0.7 * (en - s):
                    return False
        return True

    def assign(self, e, prev=None):
        """구부림 e 를 어느 자리 제품으로 볼지. 없으면 None.
        prev = (앞 구부림, 그 자리): 앞 구부림 바로 뒤에 이어지면 같은 자리로 봄"""
        if SLOT_CONTINUE and prev is not None and prev[1] is not None and e[0] - prev[0][1] < ARMREST_GAP_SEC:
            k = prev[1]
            if any(s <= e[0] + 2.5 and (en is None or en >= e[0]) for s, en in self.out[k]):
                return k
        best = None
        for k in range(len(self.slots)):
            for i, (s, en) in enumerate(self.out[k]):
                if s <= e[0] + SLOT_SLACK and (en is None or en >= e[0]):
                    # 우선순위: 이 자리를 처음 꺼낸 것 > 구부림 전에 꺼낸 것 > 가장 최근에 꺼낸 것
                    first = all(en2 is not None and en2 - s2 < SLOT_EMPTY_SEC for s2, en2 in self.out[k][:i])
                    key = (first, s)
                    if best is None or key > best[0]:
                        best = (key, k)
        if best is not None:
            return best[1]
        for k in range(len(self.slots)):          # 그 순간 빈 자리가 없으면 가장 최근에 꺼낸 자리
            for s, en in self.out[k]:
                if s <= e[0] and (best is None or s > best[0]):
                    best = (s, k)
        return None if best is None else best[1]

    def split(self, e, step=0.1):
        """구부림 e 를 시간 조각으로 나눠 조각마다 자리 배정. 반환: {자리: 구부림 시간}"""
        out = {}
        t = e[0]
        while t < e[1] - 1e-9:
            k = self.assign([t, t])
            if k is not None:
                out[k] = out.get(k, 0.0) + min(step, e[1] - t)
            t += step
        return out


class LiveJudge:
    """영상을 한 프레임씩 받으면서 실시간으로 제품 1~3의 힌지 검사 상태를 계산.
    영상이 끝난 뒤 계산하는 방식과 같은 규칙(앞뒤 WIN_SEC 창)을 쓰기 때문에 WIN_SEC 만큼 늦게 반영됨."""

    def __init__(self):
        self.half = max(1, int(round(WIN_SEC * ANALYZE_FPS)))
        self.times, self.feats = [], []
        self.holds, self.kps, self.crops = [], [], []
        self.rejected = []           # 두 손 조건으로 제외된 구간 (재포장/옮기기로 봄)
        self.slots = None            # SlotTracker (자리 추적, 없으면 순서대로 셈)
        self.event_slot = []         # 구부림별 배정된 자리
        self.next_idx = 0            # 아직 판정 안 한 프레임 번호
        self.cur = None              # 진행 중인 구부림 [시작, 마지막 구부림 시각]
        self.events = []             # 끝난 구부림 이벤트
        self.groups = []             # 암레스트(제품)별 구부림 묶음
        self.last_flag = False

    def push(self, t, hb, kp=None, crop=None):
        """crop: 쥔 힌지 부분 흑백 사진 (48x48). 힌지가 꺾였는지 모습 변화로 확인할 때 씀"""
        self.times.append(t)
        self.holds.append(hb)
        self.kps.append(kp)
        self.crops.append(crop)
        f = None
        if hb is not None and kp is not None:
            c = np.array([(hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2])
            j = min((LEFT_WRIST, RIGHT_WRIST), key=lambda j: np.hypot(*(kp[j, :2] - c)))
            f = (c[1], np.log((hb[3] - hb[1]) / max(hb[2] - hb[0], 1e-3)), max(hb[2] - hb[0], hb[3] - hb[1]), kp[j, 1])
        self.feats.append(f)
        while self.next_idx + self.half < len(self.times):
            self._judge(self.next_idx)
            self.next_idx += 1

    def finish(self):
        while self.next_idx < len(self.times):
            self._judge(self.next_idx)
            self.next_idx += 1
        if self.cur is not None:
            self._close(self.cur)
            self.cur = None

    def _flag_at(self, i):
        win = [f for f in self.feats[max(0, i - self.half): i + self.half + 1] if f is not None]
        if len(win) < MIN_OBS:
            return False
        size = np.median([f[2] for f in win])
        wy = np.array([f[3] for f in win])     # 힌지를 쥔 손목의 세로 위치
        return bool(direction_changes(wy / size, WRIST_ZC_AMP) >= WRIST_ZC_TH and np.ptp(wy) / size >= WRIST_MIN_RANGE)

    def _judge(self, i):
        t = self.times[i]
        f = self._flag_at(i)
        self.last_flag = f
        if f:
            if self.cur is None:
                self.cur = [t, t]
            else:
                self.cur[1] = t
        elif self.cur is not None and t - self.cur[1] > EVENT_MERGE_SEC:
            self._close(self.cur)
            self.cur = None

    def _close(self, e):
        e = [e[0], e[1] + 1.0 / ANALYZE_FPS]
        if e[1] - e[0] < EVENT_MIN_SEC:
            return
        if not self._two_hand_ok(e) or not self._range_ok(e):
            self.rejected.append(e)
            return
        self.events.append(e)
        self.event_slot.append(self.slots.split(e) if self.slot_mode() else None)
        if self.groups and self._same_product(self.groups[-1], e):
            self.groups[-1].append(e)
        else:
            self.groups.append([e])

    @staticmethod
    def _same_product(g, e):
        """구부림 e 가 묶음 g(같은 제품)에 이어지는지.
        앞 구부림이 끝난 뒤 ARMREST_GAP_SEC 안에 다시 시작하면 같은 제품. 단, 묶음 전체가 GROUP_MAX_SEC 를 넘으면 새 제품"""
        return e[0] - g[-1][1] < ARMREST_GAP_SEC and e[1] - g[0][0] <= GROUP_MAX_SEC

    def _range_ok(self, e):
        """구부림 후보 e 가 진짜 힌지 꺾임인지:
          - 힌지가 위아래로 충분히 움직임 (BEND_MIN_RANGE)
          - 검출이 튀지 않음 (AR_STD_MAX)
          - 힌지 모습이 처음과 달라짐 = 꺾임 (SHAPE_SIM_MAX)"""
        idx = [i for i, t in enumerate(self.times) if e[0] <= t <= e[1] and self.holds[i] is not None]
        if len(idx) < 3:
            return False
        bx = np.array([self.holds[i] for i in idx])
        size = np.median(np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]))
        cy = (bx[:, 1] + bx[:, 3]) / 2
        ar = np.log((bx[:, 3] - bx[:, 1]) / np.maximum(bx[:, 2] - bx[:, 0], 1e-3))
        if float(np.ptp(cy) / size) < BEND_MIN_RANGE or float(np.std(ar)) > AR_STD_MAX:
            return False
        cr = [self.crops[i] for i in idx if self.crops[i] is not None]
        if len(cr) >= 4:
            if min(_ncc(cr[0], c) for c in cr) > SHAPE_SIM_MAX:
                return False
        return True

    def _two_hand_ok(self, e):
        if not TWO_HAND_CHECK:
            return True
        return two_hand_check(self.times, self.holds, self.kps, e)[0]

    @staticmethod
    def _dur(g):
        return sum(b - a for a, b in g)

    def confirmed_groups(self):
        """구부림 시간 합이 PRODUCT_MIN_BEND_SEC 이상인 묶음 = 힌지 검사한 제품"""
        return [g for g in self.groups if self._dur(g) >= PRODUCT_MIN_BEND_SEC]

    def slot_mode(self):
        return self.slots is not None and self.slots.ready()

    def slot_states(self, finished=False):
        """자리 추적 모드: 자리 1~3 (왼쪽부터) 상태"""
        now = self.times[self.next_idx - 1] if self.next_idx > 0 else 0.0
        dur = [0.0] * ARMRESTS_PER_BOX
        span = [None] * ARMRESTS_PER_BOX
        for e, part in zip(self.events, self.event_slot):
            for k, d in (part or {}).items():
                if k < ARMRESTS_PER_BOX:
                    dur[k] += d
                    span[k] = (e[0] if span[k] is None else span[k][0], e[1])
        cur_k, cur_dur, ok_now = None, 0.0, True
        if self.cur is not None and not finished:
            cur_dur = self.cur[1] - self.cur[0] + 1.0 / ANALYZE_FPS
            ok_now = cur_dur < 0.5 or (self._two_hand_ok(self.cur) and self._range_ok(self.cur))
            cur_k = self.slots.assign([self.cur[1], self.cur[1]])   # 지금 구부리는 자리
        holding = any(h is not None for h in self.holds[-int(0.5 * ANALYZE_FPS):])
        states = []
        for k in range(ARMRESTS_PER_BOX):
            d = dur[k] + (cur_dur if (k == cur_k and ok_now) else 0.0)
            out_now = bool(self.slots.out[k]) and self.slots.out[k][-1][1] is None
            if d >= PRODUCT_MIN_BEND_SEC:
                a, b = span[k] if span[k] else (self.cur[0], None)
                states.append(("완료", 1.0, f"구부림 {a:.1f}초~" + (f"{b:.1f}초" if b else " (진행 중)") + f" · 자리 {k + 1}"))
            elif finished:
                if self.slots.taken(k, now):
                    states.append(("누락", 0.0, f"자리 {k + 1}: 꺼냈는데 힌지 구부림 없음"))
                else:
                    states.append(("누락", 0.0, f"자리 {k + 1}: 제품을 꺼내지 않음"))
            elif out_now and d > 0 and ok_now:
                states.append(("분석", min(1.0, d / PRODUCT_MIN_BEND_SEC), f"구부림 {d:.1f}s / {PRODUCT_MIN_BEND_SEC:.1f}s"))
            elif out_now and holding:
                states.append(("잡음", 0.0, f"자리 {k + 1} 제품 꺼냄 · 힌지 쥠"))
            else:
                states.append(("대기", 0.0, f"자리 {k + 1}"))
        return self._by_inspect_order(states, span, cur_k)

    def _by_inspect_order(self, states, span, cur_k):
        """화면·결과를 '검사한 순서'로 정렬: 먼저 완료된 자리가 제품 1.
        한 번 순서가 정해지면 바뀌지 않음. 아직 완료 안 된 자리는 그 뒤에 (지금 검사 중인 자리 먼저)."""
        order = getattr(self, "_slot_order", [])
        new = [k for k, st in enumerate(states) if st[0] == "완료" and k not in order]
        new.sort(key=lambda k: span[k][0] if span[k] else 1e9)
        order = order + new
        self._slot_order = order
        rest = [k for k in range(len(states)) if k not in order]
        rest.sort(key=lambda k: (k != cur_k, k))
        return [states[k] for k in order + rest]

    def product_states(self, finished=False):
        if self.slot_mode():
            return self.slot_states(finished)
        return self._seq_states(finished)

    def _seq_states(self, finished=False):
        """제품 1~3 상태 (상태, 진행률 0~1, 설명). 상태 기준:
          대기 : 이 제품 차례인데 힌지를 안 쥐고 있음
          잡음 : 힌지를 쥐고 있음 (아직 구부림은 없음)
          분석 : 구부림 동작 감지 중. 진행률 = 구부림 시간 합 / PRODUCT_MIN_BEND_SEC
          완료 : 구부림 시간 합이 PRODUCT_MIN_BEND_SEC 이상 (두 손 조건·움직임 폭 조건 통과)
          누락 : 영상이 끝날 때까지 구부림 없음"""
        done = self.confirmed_groups()
        now = self.times[self.next_idx - 1] if self.next_idx > 0 else 0.0
        last = self.groups[-1] if self.groups else None
        # 지금 진행 중인 제품의 구부림 시간 합 (확정 안 된 묶음 + 진행 중 구부림)
        pend, ok_now = 0.0, True
        if not finished:
            cur_dur = 0.0
            if self.cur is not None:
                cur_dur = self.cur[1] - self.cur[0] + 1.0 / ANALYZE_FPS
                # 시작 0.5초 동안은 판단할 자료가 부족하니 통과로 봄
                ok_now = cur_dur < 0.5 or (self._two_hand_ok(self.cur) and self._range_ok(self.cur))
            last_open = last is not None and self._dur(last) < PRODUCT_MIN_BEND_SEC and now - last[-1][1] < ARMREST_GAP_SEC
            cur_joins_done = (self.cur is not None and last is not None and self._dur(last) >= PRODUCT_MIN_BEND_SEC
                              and self._same_product(last, self.cur))
            if not cur_joins_done:
                pend = cur_dur + (self._dur(last) if last_open else 0.0)
        holding = any(h is not None for h in self.holds[-int(0.5 * ANALYZE_FPS):])   # 최근 0.5초 안에 쥐었으면 "잡음"
        states = []
        for k in range(ARMRESTS_PER_BOX):
            if k < len(done):
                g = done[k]
                states.append(("완료", 1.0, f"구부림 {g[0][0]:.1f}~{g[-1][1]:.1f}초 (합 {self._dur(g):.1f}s)"))
            elif finished:
                states.append(("누락", 0.0, "힌지 구부림 없음"))
            elif k == len(done):
                if (pend >= PRODUCT_MIN_BEND_SEC and ok_now) or getattr(self, "_live_done", -1) == k:
                    self._live_done = k          # 한 번 완료로 보여준 제품은 다시 되돌리지 않음 (깜빡임 방지)
                    states.append(("완료", 1.0, f"구부림 합 {PRODUCT_MIN_BEND_SEC:.1f}s 이상 확인" if pend < PRODUCT_MIN_BEND_SEC else f"구부림 합 {pend:.1f}s (진행 중)"))
                elif pend > 0 and ok_now:
                    states.append(("분석", min(1.0, pend / PRODUCT_MIN_BEND_SEC), f"구부림 {pend:.1f}s / {PRODUCT_MIN_BEND_SEC:.1f}s"))
                elif pend > 0:
                    states.append(("잡음", 0.0, "힌지 쥐고 있음 · 구부림 아님(움직임 작음/두 손 같이 움직임)"))
                elif holding:
                    states.append(("잡음", 0.0, f"힌지 쥐고 있음 · 구부림 0.0s / {PRODUCT_MIN_BEND_SEC:.1f}s"))
                else:
                    states.append(("대기", 0.0, f"힌지 안 쥠 · 0.0s / {PRODUCT_MIN_BEND_SEC:.1f}s"))
            else:
                states.append(("대기", 0.0, f"0.0s / {PRODUCT_MIN_BEND_SEC:.1f}s"))
        return states


# 화면에 그릴 관절 연결선 (COCO 17 키포인트: 어깨 5,6 / 팔꿈치 7,8 / 손목 9,10 / 골반 11,12)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12)]


# 미디어파이프 상체 연결선 (얼굴 제외: 어깨 11,12 / 팔꿈치 13,14 / 손목 15,16 / 손가락 17~22 / 골반 23,24)
MP_UPPER = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (15, 21), (17, 19),
            (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24)]


def draw_overlay(frame, boxes, hb, kp, t, live_bend, mp_lm=None, pose_src=""):
    """힌지 네모 + 팔/몸통 관절 + 상태 글자를 프레임 위에 그림"""
    fh, fw = frame.shape[:2]
    if mp_lm is not None:
        # 미디어파이프 랜드마크 그대로 그림
        pts = {i: (int(mp_lm[i, 0] * fw), int(mp_lm[i, 1] * fh)) for i in range(11, 25) if mp_lm[i, 2] >= KPT_CONF}
        for a, b in MP_UPPER:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 3)
        for i, p in pts.items():
            cv2.circle(frame, p, 8 if i in (15, 16) else 5, (0, 255, 0) if i in (15, 16) else (0, 0, 255), -1)
    elif kp is not None:
        pts = {i: (int(kp[i, 0] * fw), int(kp[i, 1] * fh)) for i in range(len(kp)) if kp[i, 2] >= KPT_CONF}
        for a, b in SKELETON:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 200, 255), 3)
        for i, p in pts.items():
            if i >= 5:
                cv2.circle(frame, p, 7 if i in (LEFT_WRIST, RIGHT_WRIST) else 5,
                           (0, 255, 0) if i in (LEFT_WRIST, RIGHT_WRIST) else (0, 200, 255), -1)
    # 힌지 네모: 파랑 = 보이기만 함, 빨강 = 손으로 쥐고 있음
    for b in boxes:
        is_held = hb is not None and np.allclose(b, hb)
        color = (0, 0, 255) if is_held else (255, 0, 0)
        p1, p2 = (int(b[0] * fw), int(b[1] * fh)), (int(b[2] * fw), int(b[3] * fh))
        cv2.rectangle(frame, p1, p2, color, 3)
        cv2.putText(frame, "hinge (held)" if is_held else "hinge", (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if live_bend:
        cv2.putText(frame, "HINGE BENDING", (20, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)



# ====== 대시보드 화면 (해머 검사 화면과 같은 구성) ======
FONT_PATHS = [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
_FONT_CACHE = {}


def _font(size):
    if size not in _FONT_CACHE:
        from PIL import ImageFont
        f = None
        for p in FONT_PATHS:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, size)
                    break
                except OSError:
                    pass
        _FONT_CACHE[size] = f or ImageFont.load_default()
    return _FONT_CACHE[size]


BADGE = {"대기": ((110, 110, 110), (230, 230, 230)), "잡음": ((60, 130, 230), (255, 255, 255)),
         "분석": ((255, 190, 30), (30, 30, 30)),
         "완료": ((40, 190, 90), (255, 255, 255)), "누락": ((220, 60, 60), (255, 255, 255))}


def render_dashboard(frame, states, t, total_sec, fps_proc, hb, bending, verdict=None, name=""):
    """왼쪽: 영상(힌지 네모/관절) / 오른쪽: 제품별 체크리스트 / 아래: 판단 기준"""
    from PIL import Image, ImageDraw
    W, H, VW, VH = 1600, 900, 1120, 630
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    vid = cv2.resize(frame, (VW, VH))
    canvas.paste(Image.fromarray(cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)), (0, 90))
    d = ImageDraw.Draw(canvas)
    if name:
        d.text((24, 26), f"영상: {name}", font=_font(32), fill=(240, 240, 240))

    n_done = sum(1 for st, _, _ in states if st == "완료")
    d.rectangle([20, 110, 470, 170], fill=(10, 10, 10))
    d.text((36, 122), "힌지 검사 (담당 단계만)", font=_font(28), fill=(235, 235, 235))
    d.text((360, 118), f"{n_done} / {len(states)}", font=_font(34), fill=(90, 220, 120))

    # 오른쪽 체크리스트
    x0 = VW + 30
    d.text((x0, 110), "체크리스트", font=_font(34), fill=(240, 240, 240))
    step_y = 120 if len(states) <= 3 else 90
    if SIMPLE_UI:   # 잡음·분석은 화면에 안 보이게 '대기'로 표시 (판정 알고리즘은 그대로)
        states = [("대기", 0.0, "검사 대기") if st in ("잡음", "분석", "대기") else (st, prog, desc) for st, prog, desc in states]
    for k, (st, prog, desc) in enumerate(states):
        y = 185 + k * step_y
        bg, fg = BADGE[st]
        d.rounded_rectangle([x0, y, x0 + 70, y + 36], radius=18, fill=bg)
        d.text((x0 + 14, y + 4), st, font=_font(22), fill=fg)
        d.text((x0 + 85, y + 2), f"제품 {k + 1}. 힌지 구부림 검사", font=_font(26),
               fill=(240, 240, 240) if st != "대기" else (150, 150, 150))
        d.rectangle([x0, y + 50, W - 30, y + 62], fill=(60, 60, 60))
        if prog > 0:
            d.rectangle([x0, y + 50, x0 + int((W - 30 - x0) * prog), y + 62], fill=bg)
        d.text((x0, y + 70), desc, font=_font(20), fill=(170, 170, 170))

    if verdict is not None:
        vc = {"PASS": (40, 190, 90), "FAIL": (220, 60, 60)}.get(verdict, (255, 190, 30))
        d.rounded_rectangle([x0, 560, W - 30, 640], radius=12, fill=vc)
        d.text((x0 + 20, 578), f"박스 판정: {verdict}", font=_font(34), fill=(255, 255, 255))

    # 아래 판단 기준
    d.text((24, 735), "판단 기준", font=_font(20), fill=(150, 150, 150))
    if not SIMPLE_UI:   # 쥐고 있음/구부림 감지 상태는 SIMPLE_UI 에서 숨김
        d.text((24, 765), f"힌지 쥐고 있음: {'O' if hb is not None else 'X'}   /   "
                          f"구부림: {'감지 중' if bending else '-'}", font=_font(26), fill=(240, 240, 240))
    cur = next((k for k, (st, _, _) in enumerate(states) if st in ("분석", "대기", "잡음")), None)
    msg = f"현재 감지: 제품 {cur + 1}. 힌지 구부림 검사" if cur is not None else "모든 제품 검사 끝"
    d.text((24, 805), msg, font=_font(26), fill=(90, 220, 120))
    legend = (f"대기: 아직 힌지 검사 확인 안 됨  →  완료: 힌지 구부림 확인" if SIMPLE_UI else
              f"대기: 힌지 안 쥠  →  잡음: 힌지 쥠  →  분석: 구부림 감지(합 {PRODUCT_MIN_BEND_SEC:.1f}초까지)  →  완료: 구부림 합 {PRODUCT_MIN_BEND_SEC:.1f}초 이상")
    d.text((24, 848), legend,
           font=_font(18), fill=(170, 170, 170))
    d.text((x0, 860), f"{t:.0f}초 / 전체 {total_sec:.0f}초 · 처리 {fps_proc:.0f}fps", font=_font(20), fill=(230, 140, 60))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def process_video(path, hinge_model, pose, armrest_model=None):
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / ANALYZE_FPS)))

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    total_sec = total_frames / fps if total_frames > 0 else 0.0
    writer = None
    if SAVE_VIDEO:
        os.makedirs("hinge_check_out", exist_ok=True)
        writer = cv2.VideoWriter(f"hinge_check_out/{name}_result.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), ANALYZE_FPS, (1600, 900))
    judge = LiveJudge()
    if armrest_model is not None:
        judge.slots = SlotTracker()
    import time as _time
    t_start = _time.time()

    pose.new_video()
    static_filter = StaticFilter()
    tracker = HeldTracker()
    times, held, seen = [], [], []
    stop_all = False
    skipped = False
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_last = frame
            boxes = detect_hinges(frame, hinge_model)
            kp, mp_lm, pose_src = pose(frame, idx / fps * 1000)
            st = static_filter(boxes)
            hb = tracker(idx / fps, boxes, kp, st) if TRACK_HELD else held_box(boxes, kp, st)
            times.append(idx / fps)
            held.append(hb)
            seen.append(len(boxes) > 0)
            if armrest_model is not None and len(times) % SLOT_EVERY == 1:
                ar = armrest_model.predict(frame, conf=SLOT_CONF, imgsz=640, verbose=False)[0]
                H_, W_ = frame.shape[:2]
                abx = ar.boxes.xyxy.cpu().numpy() / [W_, H_, W_, H_] if len(ar.boxes) else np.zeros((0, 4))
                judge.slots.update(idx / fps, abx)
            judge.push(idx / fps, hb, kp, hinge_crop(frame, hb))
            if judge.slots is not None:   # 박스 안 암레스트 (하늘색)
                H_, W_ = frame.shape[:2]
                for b in judge.slots.boxes:
                    cv2.rectangle(frame, (int(b[0] * W_), int(b[1] * H_)), (int(b[2] * W_), int(b[3] * H_)), (255, 200, 0), 2)
            if writer is not None or SHOW_PREVIEW:
                draw_overlay(frame, boxes, hb, kp, idx / fps, judge.last_flag, mp_lm, pose_src)
                fps_proc = len(times) / max(1e-3, _time.time() - t_start)
                screen = render_dashboard(frame, judge.product_states(), idx / fps, total_sec, fps_proc, hb, judge.last_flag, name=name)
                if writer is not None:
                    writer.write(screen)
                if SHOW_PREVIEW:
                    cv2.imshow("hinge check", screen)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):          # 이 영상만 종료하고 다음 영상으로
                        print(f"[{name}] q: 이 영상 건너뜀")
                        skipped = True
                        break
                    if key == ord("d"):          # 전체 종료
                        print(f"[{name}] d: 전체 종료")
                        stop_all = True
                        skipped = True
                        break
        idx += 1
    cap.release()

    judge.finish()
    events = judge.events
    groups = judge.confirmed_groups()
    seen_rate = float(np.mean(seen)) if seen else 0.0

    if skipped:
        verdict = "SKIPPED"
        reason = f"q 로 중간에 멈춤 ({times[-1]:.1f}초까지만 봄) - 판정하지 않음"
    elif seen_rate < MIN_HINGE_SEEN:
        verdict = "REVIEW"
        reason = f"힌지가 거의 안 보임 (보인 비율 {seen_rate:.0%})"
    else:
        final = judge.product_states(finished=True)
        n_ok = sum(1 for st, _, _ in final if st == "완료")
        missing = [str(k + 1) for k, (st, _, _) in enumerate(final) if st == "누락"]
        if n_ok >= ARMRESTS_PER_BOX:
            verdict = "PASS"
            reason = f"제품 {n_ok}개에서 구부림 확인" + (" (자리 추적)" if judge.slot_mode() else " (순서대로 셈)")
        else:
            verdict = "FAIL"
            reason = f"힌지 구부림 확인 {n_ok}개 / 필요 {ARMRESTS_PER_BOX}개 (누락 {len(missing)}개)"
            if judge.slot_mode():
                reason += " - " + ", ".join(desc for st, _, desc in final if st == "누락")

    print(f"[{name}] {verdict} - {reason}")
    if judge.slot_mode() and verdict != "SKIPPED":
        for k, (st, _, desc) in enumerate(judge.product_states(True)):
            print(f"    제품 {k + 1}: {st} ({desc})")
    missing_list = ""
    if verdict in ("PASS", "FAIL"):
        missing_list = str(sum(1 for st, _, _ in judge.product_states(True) if st == "누락"))
    if (SHOW_PREVIEW or writer is not None) and times and not stop_all:
        final_states = judge.product_states(finished=(verdict != "SKIPPED"))
        screen = render_dashboard(frame_last, final_states, times[-1], total_sec, 0, None, False, verdict, name=name)
        if writer is not None:
            for _ in range(ANALYZE_FPS * 2):
                writer.write(screen)
        if SHOW_PREVIEW:
            cv2.imshow("hinge check", screen)
            key = cv2.waitKey(1500) & 0xFF
            if key == ord("d"):
                stop_all = True
    if writer is not None:
        writer.release()
        print(f"    결과 영상 저장: hinge_check_out/{name}_result.mp4")
    for k, g in enumerate(groups, 1):
        print(f"    구부림 묶음 {k}: " + ", ".join(f"{a:.1f}~{b:.1f}초" for a, b in g))
    if judge.rejected:
        print("    구부림에서 제외(힌지 안 꺾임 / 두 손이 같이 움직임 / 검출 튐): " + ", ".join(f"{a:.1f}~{b:.1f}초" for a, b in judge.rejected))
    if POSE_BACKEND == "mediapipe":
        st = pose.stats
        tot = max(1, sum(st.values()))
        print(f"    관절 인식: 미디어파이프 {st['mediapipe'] / tot:.0%} / YOLO 보충 {st['yolo'] / tot:.0%} / 못 찾음 {st['none'] / tot:.0%}")
    return dict(video=name, verdict=verdict, reason=reason, armrest_groups=len(groups), missing=missing_list,
                hinge_seen_rate=round(seen_rate, 2), events=events, stop_all=stop_all)


def load_gt():
    if not os.path.exists(GT_CSV):
        return {}
    gt = {}
    for r in csv.DictReader(open(GT_CSV, encoding="utf-8-sig")):
        key = r["영상"].replace("crop0", "crop").replace("crop", "")
        ivs = []
        for a, b in (("구부림_시작(초)", "구부림_끝(초)"), ("구부림2_시작(초)", "구부림2_끝(초)")):
            if r.get(a):
                ivs.append((float(r[a]), float(r[b])))
        gt.setdefault(key, []).append((r["암레스트_순서"], r["힌지_정답"], ivs))
    return gt


def compare_with_gt(result, gt, tol=1.5):
    key = result["video"].replace("crop0", "crop").replace("crop", "")
    if key not in gt or result["verdict"] == "SKIPPED":
        return
    rows = gt[key]
    expect = "PASS" if all(lab == "PASS" for _, lab, _ in rows) else "FAIL"
    bends = [iv for _, lab, ivs in rows if lab == "PASS" for iv in ivs]
    hit = set()
    fp = 0
    for e in result["events"]:
        m = [i for i, (s, t) in enumerate(bends) if e[0] <= t + tol and e[1] >= s - tol]
        hit.update(m)
        fp += not m
    mark = "맞음" if expect == result["verdict"] else "틀림"
    print(f"    정답 {expect} / 판정 {result['verdict']} -> {mark} | 정답 구부림 {len(bends)}개 중 {len(hit)}개 찾음, 오탐 {fp}개")
    gt_miss = sum(1 for _, lab, _ in rows if lab != "PASS")
    if gt_miss or result.get("missing") not in (None, "", "0"):
        print(f"    누락 개수: 정답 {gt_miss} / 판정 {result.get('missing') or 0}")


def find_model():
    """best.pt 자동 찾기. ultralytics 는 버전에 따라 runs\\detect\\ 아래에 저장하기도 함"""
    home = os.path.expanduser("~")
    cands = []
    for rnd in ("round2", "round1"):
        for base in ("hinge_det_runs",
                     os.path.join("runs", "detect", "hinge_det_runs"),
                     os.path.join(home, "runs", "detect", "hinge_det_runs")):
            cands.append(os.path.join(base, rnd, "weights", "best.pt"))
    for c in [HINGE_MODEL, HINGE_MODEL_FALLBACK] + cands:
        if c and os.path.exists(c):
            return c
    return None


def main():
    if len(sys.argv) > 1:
        videos = sys.argv[1:]
    else:
        videos = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4")) + glob.glob(os.path.join(VIDEO_FOLDER, "*.mov")))
    if not videos:
        print(f"영상이 없습니다: {VIDEO_FOLDER}")
        return
    model_path = find_model()
    if model_path is None:
        print("힌지 모델(best.pt)을 찾지 못했습니다. train_hinge_det.py 를 먼저 실행하거나,")
        print("best.pt 위치를 코드 위쪽 HINGE_MODEL 에 직접 적어 주세요.")
        return
    print(f"힌지 모델: {model_path}")

    hinge_model = YOLO(model_path)
    armrest_model = None
    if ARMREST_MODEL and os.path.exists(ARMREST_MODEL):
        armrest_model = YOLO(ARMREST_MODEL)
        print(f"암레스트 모델: {ARMREST_MODEL} (자리 추적 사용)")
    else:
        print(f"[안내] 암레스트 모델({ARMREST_MODEL})이 없어 제품을 순서대로 셉니다.")
    pose = PoseEstimator()
    gt = load_gt()

    results = []
    for v in videos:
        r = process_video(v, hinge_model, pose, armrest_model)
        compare_with_gt(r, gt)
        results.append(r)
        if r["stop_all"]:
            break

    with open("hinge_check_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["영상", "판정", "이유", "누락_개수", "구부림_묶음_수", "힌지_보인_비율"])
        for r in results:
            w.writerow([r["video"], r["verdict"], r["reason"], r["missing"], r["armrest_groups"], r["hinge_seen_rate"]])
    with open("hinge_check_events.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["영상", "구부림_시작(초)", "구부림_끝(초)"])
        for r in results:
            for a, b in r["events"]:
                w.writerow([r["video"], round(a, 1), round(b, 1)])
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    if pose.mp is not None:
        pose.mp.close()
        pose.mp = None
    print("\n결과 저장: hinge_check_results.csv, hinge_check_events.csv")


if __name__ == "__main__":
    main()
