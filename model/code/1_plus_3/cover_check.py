"""덮개(보호망) 여닫기 검사. 파일 하나로 끝난다.

쓰는 법 (합치는 쪽에서):

    from cover_check import CoverCheck, arms_from_pose

    check = CoverCheck("cover.pt")
    for t, frame in 영상:
        arms = arms_from_pose(pose_result)      # MediaPipe 결과, 없으면 None
        check.feed(frame, t, arms)
    열림, 근거 = check.verdict_open()
    ...제품 3개 검사...
    닫힘, 근거 = check.verdict_close()

판정하는 것: 덮개를 열었는가 / 다시 닫았는가.

**위치가 아니라 면적을 본다.** 덮개가 걷히면 그물이 접히면서 화면에서
차지하는 면적이 줄어든다. 그래서 세그멘테이션(bbox 가 아니라 마스크)을
쓴다. bbox 는 접힌 그물도 펼친 그물도 비슷한 크기로 잡는다.

**절대 면적으로는 못 잰다.** 카메라마다 덮힌 상태의 그물 면적이 화면의
10.7%~32.3% 로 다르다. 그래서 사이클 시작에 "덮힌 상태"를 스스로 재서
기준으로 삼고, 그 기준의 절반을 오르내리는 것을 본다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- 검출 ---------------------------------------------------------------
IMAGE_SIZE = 512
CONF = .35

# ---- 덮힘 기준 잡기 -----------------------------------------------------
# 사이클이 시작될 때 덮개는 덮여 있다. 처음 이만큼 동안 본 면적으로
# 기준을 잡는다.
BASE_SECONDS = 2.0
# 중앙값이 아니라 80 분위를 쓴다. crop009 에서 기준 구간 2초 동안 손이
# 그물을 가려 중앙값이 7.7% 로 잡혔다(실제 덮힘은 그보다 훨씬 넓다).
# 최댓값으로 바꾸면 crop011 이 28.7% 로 튀어 반대로 깨진다.
BASE_PERCENTILE = 80
# 기준이 이보다 낮으면 판정하지 않는다. 사이클이 이미 열린 채로 시작했거나,
# 모델이 이 카메라의 그물을 못 보는 것이다. 둘 다 "모르겠다"가 맞는 답이다.
BASE_FLOOR = .05

# ---- 여닫기 ------------------------------------------------------------
# 기준의 이 비율 아래로 내려가면 열린 것으로 본다.
SHARE = .50
# 이만큼 버텨야 인정한다. 스쳐 지나가는 가림은 통과시키지 않는다.
HOLD_SECONDS = 1.0

# ---- 팔 동작 -----------------------------------------------------------
# 여는 것은 팔 동작까지 같이 본다. 실측 5편 모두 아크가 0.84~0.92 였다.
ARM_ARC = .40
ARM_DIP = .75
# 팔이 이만큼도 안 잡히면 팔 조건을 아예 면제한다. 못 보는 것을 조건으로
# 걸면 진짜 여는 것을 영영 못 잡는다. crop19 는 팔이 10% 프레임만 잡힌다.
ARMS_NEEDED = .20
# 아크를 재는 창. 여는 동작 한 벌이 이 안에 들어간다.
WINDOW_SECONDS = 3.2

# 어깨-팔꿈치-손목. MediaPipe Pose 번호.
SHOULDER_ELBOW_WRIST = {"left": (11, 13, 15), "right": (12, 14, 16)}


def arm_trusted(landmarks, side: str, min_visibility: float = .4) -> bool:
    """이 팔의 어깨-팔꿈치-손목을 믿을 수 있는가.

    MediaPipe 는 안 보이는 관절을 화면 밖으로 늘려 찍으면서 visibility 는
    높게 준다. 세로가 짧게 잘린 영상에서 상반신이 화면 밖이면 팔이 통째로
    엉뚱한 데 찍힌다. 그 값으로 각도를 재면 판정이 틀린다.

    각도를 재는 쪽과 화면에 그리는 쪽이 **같은 기준을 써야 한다.** 따로
    두었더니 계산에서는 버린 팔이 화면에는 그대로 그려졌다. 보는 사람에게는
    "미디어파이프가 엉뚱한 데 가 있다"로만 보인다.
    """
    marks = [landmarks[i] for i in SHOULDER_ELBOW_WRIST[side]]
    if min(getattr(p, "visibility", 1.) for p in marks) < min_visibility:
        return False
    if any(not (-.05 <= p.x <= 1.05 and -.05 <= p.y <= 1.05) for p in marks):
        return False
    # 팔이 화면 가장자리에 납작하게 붙은 경우. 어깨와 손목이 둘 다 가장자리
    # 5% 안에 있으면 사람 몸이 아니라 늘려 찍은 것이다.
    edge = lambda p: p.x <= .05 or p.x >= .95 or p.y <= .05 or p.y >= .95
    return not (edge(marks[0]) and edge(marks[2]))


def arms_from_pose(pose_result, min_visibility: float = .4) -> Dict[str, float]:
    """MediaPipe Pose 결과에서 팔마다 얼마나 펴졌는지 꺼낸다.

    1.0 에 가까울수록 곧게 펴진 상태다. 믿을 수 없는 팔은 아예 넣지 않는다.
    `pose_result` 는 Tasks API 의 PoseLandmarkerResult 다.
    """
    if not pose_result or not getattr(pose_result, "pose_landmarks", None):
        return {}
    landmarks = pose_result.pose_landmarks[0]
    out: Dict[str, float] = {}
    for side, (shoulder, elbow, wrist) in SHOULDER_ELBOW_WRIST.items():
        if not arm_trusted(landmarks, side, min_visibility):
            continue
        a, b, c = landmarks[shoulder], landmarks[elbow], landmarks[wrist]
        upper = math.dist((a.x, a.y), (b.x, b.y))
        forearm = math.dist((b.x, b.y), (c.x, c.y))
        if upper + forearm < 1e-6:
            continue
        out[side] = math.dist((a.x, a.y), (c.x, c.y)) / (upper + forearm)
    return out


def one_arm_arc(values: List[float]) -> Tuple[float, float]:
    """폈다 굽혔다 다시 폈는가. (아크, 최저점) 을 돌려준다.

    덮개를 여닫는 동작의 모양이다. 실측(cover.mp4, 여는 장면):

        1.1~2.4초  0.93~1.00   팔 뻗어 그물망을 잡음
        2.5초      0.18        당기면서 확 굽힘
        3.1~3.5초  0.35~0.68   굽힌 채로 당김
        3.7~4.1초  0.66~1.00   다 당기고 다시 펴짐

    그냥 "팔이 펴졌다"로 보면 물건을 집는 동작과 구분이 안 된다. 폈다가
    굽혔다가 다시 펴지는 한 벌을 봐야 한다.
    """
    if len(values) < 3:
        return 0., 1.
    best = 0.
    dip = 1.
    for middle in range(1, len(values) - 1):
        before = max(values[:middle])
        after = max(values[middle + 1:])
        low = values[middle]
        arc = min(before, after) - low
        if arc > best:
            best, dip = arc, low
    return best, dip


class CoverCheck:
    """프레임을 넣으면 그물 면적을 재고, 열었는지 닫았는지 답한다."""

    def __init__(self, weights: str = "cover.pt") -> None:
        self.weights = Path(weights)
        self._model = None
        self.reason_off = ""
        self.last_mask = None       # 화면에 그리고 싶을 때 쓴다
        self.last_area = 0.
        self.reset()

    def reset(self) -> None:
        self.base = 0.
        self._first: List[Tuple[float, float]] = []
        self._low_since: Optional[float] = None
        self._high_since: Optional[float] = None
        self.opened = False
        self._now = 0.
        self._arms: List[Tuple[float, Dict[str, float]]] = []
        self._arm_frames = 0
        self._frames = 0

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if not self.weights.exists():
            self.reason_off = f"가중치가 없습니다: {self.weights}"
            return False
        return True

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights), task="segment")
        return self._model

    # -- 한 프레임 ---------------------------------------------------------
    def look(self, frame) -> float:
        """이 프레임에서 그물이 화면의 몇 할을 덮고 있나. 0.0 ~ 1.0."""
        self.last_mask = None
        if not self.available:
            return 0.
        try:
            result = self._load().predict(frame, conf=CONF, imgsz=IMAGE_SIZE,
                                          verbose=False)[0]
        except Exception as exc:
            self.reason_off = f"검출 중 오류: {exc}"
            return 0.
        if result.masks is None or len(result.masks.data) == 0:
            return 0.
        # **재는 것은 가장 큰 조각, 그리는 것은 전부.** 덮개가 걷히면 그물이
        # 여러 조각으로 잡히는데, 재는 값에 조각을 다 더하면 접힌 그물이
        # 펼친 그물보다 넓어 보일 수 있다. 반대로 그릴 때 가장 큰 것만
        # 그리면 화면에 절반만 칠해져 고장난 것처럼 보인다.
        best = 0.
        whole = None
        for one in result.masks.data:
            small = one.cpu().numpy()
            share = float(small.sum()) / small.size
            best = max(best, share)
            whole = small if whole is None else np.maximum(whole, small)
        self.last_mask = whole
        return best

    def feed(self, frame, t: float,
             arms: Optional[Dict[str, float]] = None,
             area: Optional[float] = None) -> float:
        """프레임 하나를 넣는다. 이미 재 둔 면적을 넣어도 된다."""
        if area is None:
            area = self.look(frame)
        self.last_area = area
        self._now = t
        self._frames += 1
        if arms:
            self._arm_frames += 1
        self._arms.append((t, arms or {}))
        while self._arms and t - self._arms[0][0] > WINDOW_SECONDS:
            self._arms.pop(0)

        if self.base <= 0.:
            self._first.append((t, area))
            if t - self._first[0][0] >= BASE_SECONDS:
                seen = sorted(a for _, a in self._first)
                at = min(len(seen) - 1,
                         int(len(seen) * BASE_PERCENTILE / 100))
                self.base = seen[at]
            return area

        line = self.base * SHARE
        if area < line:
            self._high_since = None
            if self._low_since is None:
                self._low_since = t
            elif t - self._low_since >= HOLD_SECONDS:
                self.opened = True
        else:
            self._low_since = None
            if self._high_since is None:
                self._high_since = t
        return area

    # -- 팔 --------------------------------------------------------------
    @property
    def arms_seen(self) -> float:
        return (self._arm_frames / self._frames) if self._frames else 0.

    def arm_arc(self) -> Tuple[float, float]:
        """팔마다 따로 이어 보고 더 큰 쪽을 쓴다.

        **덮개는 한 손으로도 열고 두 손으로도 연다.** 두 팔을 합쳐서 더 편
        쪽만 보면, 한 손으로 열 때 노는 팔이 곧게 펴져 있어(1.0 근처)
        일하는 팔이 굽는 것을 덮어 버린다. crop003 에서 합친 값은 0.00,
        팔마다 보면 0.80 이었다.
        """
        best, dip = 0., 1.
        for side in SHOULDER_ELBOW_WRIST:
            values = [a[side] for _, a in self._arms if side in a]
            arc, low = one_arm_arc(values)
            if arc > best:
                best, dip = arc, low
        return best, dip

    @property
    def shut(self) -> bool:
        """열렸다가 다시 덮인 채로 충분히 버텼는가."""
        return (self.opened and self._high_since is not None
                and self._now - self._high_since >= HOLD_SECONDS)

    # -- 답 --------------------------------------------------------------
    def _common(self) -> Optional[str]:
        if not self.available:
            return f"[cannot judge] {self.reason_off}"
        if self.base <= 0.:
            return "[waiting] still measuring what covered looks like"
        if self.base < BASE_FLOOR:
            return (f"[cannot judge] covered baseline too low "
                    f"({self.base * 100:.1f}%) - cycle started open, or the "
                    f"model cannot see this camera's mesh")
        return None

    def _where(self) -> str:
        return (f"mesh {self.last_area * 100:.1f}% of {self.base * 100:.1f}%")

    def verdict_open(self) -> Tuple[bool, str]:
        stop = self._common()
        if stop:
            return False, stop
        arc, dip = self.arm_arc()
        blind = self.arms_seen < ARMS_NEEDED
        pulled = blind or (arc >= ARM_ARC and dip <= ARM_DIP)
        extra = f", arms seen {self.arms_seen * 100:.0f}%" if blind else ""
        if self.opened and pulled:
            return True, f"[cover moved] {self._where()}, arm arc {arc:.2f}{extra}"
        if not self.opened:
            return False, f"[waiting] {self._where()} - still covered"
        return False, (f"[waiting] {self._where()}, arm arc {arc:.2f} "
                       f"- no pulling motion{extra}")

    def verdict_close(self) -> Tuple[bool, str]:
        """**닫는 것은 면적만 본다.** 팔 동작을 요구하면 5편 중 3편만 통과한다.

        crop003 은 팔을 편 채 쓸어 덮어서 아크가 0.02 이고, crop19 는 팔이
        10% 프레임만 잡힌다. 면적만으로도 실측 5편에서 가짜 상승 구간은
        0개였다.
        """
        stop = self._common()
        if stop:
            return False, stop
        if self.shut:
            return True, f"[cover moved] {self._where()}"
        if not self.opened:
            return False, f"[waiting] {self._where()} - never opened"
        return False, f"[waiting] {self._where()} - not back in place yet"
