"""공구·비닐·그립 세그멘테이션.

학습된 모델 없이 색상·형태·밝기 분포만으로 대상을 분리한다. 각 함수는
판정에 쓸 수치와 화면에 표시할 영역을 함께 돌려주어, 어떤 근거로 통과했는지
영상에서 눈으로 확인할 수 있게 한다.

- segment_tool   : 망치·Driver 같은 주황/황동 공구를 연결요소로 분리한다.
- segment_vinyl  : 빛반사가 심한 비닐 덩어리를 국소 밝기차로 분리한다.
- grip_strength  : 손가락이 막대를 감싸 쥐었는지 손 랜드마크로 판단한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

# 분석 해상도. 원본 크기와 무관하게 같은 임계값을 쓰기 위해 고정한다.
SEG_WIDTH = 320
SEG_HEIGHT = 180
SEG_AREA = float(SEG_WIDTH * SEG_HEIGHT)

# 화면 상단(작업자 얼굴·배경 선반)은 공구 탐색에서 제외한다.
TOP_CUT = 48

# 주황/황동 계열 공구 색.
TOOL_LOWER = np.array([5, 105, 95])
TOOL_UPPER = np.array([28, 255, 255])

# 공구로 인정할 연결요소 크기. 이보다 작으면 노이즈로 본다.
TOOL_MIN_AREA = SEG_AREA * .0010
TOOL_MAX_AREA = SEG_AREA * .1200
# 공구가 "잘 보인다"고 볼 최소 크기. 노션 기준의 "잘보이는 경우 한해서 진행".
TOOL_CLEAR_AREA = SEG_AREA * .0025

# 비닐 판정용. 빛반사 픽셀은 밝고(V 높음) 색이 거의 없다(S 낮음).
VINYL_MIN_VALUE = 205
VINYL_MAX_SAT = 60
# 국소 표준편차가 이보다 크면 "밝기차가 뚜렷한" 픽셀로 본다.
VINYL_MIN_STD = 26.0
VINYL_MIN_AREA = SEG_AREA * .0015
# 반사점 하나의 최대 크기. 이보다 넓게 이어진 밝은 면은 비닐이 아니라
# 흰 장갑이나 조명 받은 평면으로 본다.
VINYL_SPECKLE_MAX = SEG_AREA * .0060
# 비닐로 인정할 최소 반사점 개수. 낱개 반사는 비닐이 아니다.
VINYL_MIN_SPECKLES = 6


@dataclass
class ToolSegment:
    """분리된 공구 하나."""

    found: bool = False
    clear: bool = False          # 가려지지 않고 충분히 보이는지
    center: Optional[Point] = None
    head: Optional[Point] = None  # 손에서 먼 쪽 끝 = 망치 머리 추정
    area_ratio: float = 0.
    box: Optional[BBox] = None


@dataclass
class VinylSegment:
    """빛반사로 분리한 비닐 덩어리."""

    found: bool = False
    area_ratio: float = 0.
    center: Optional[Point] = None
    box: Optional[BBox] = None
    blobs: int = 0               # 덩어리 개수. 벗겨내면 본체와 분리되어 늘어난다.
    spread: float = 0.           # 덩어리들이 흩어진 정도


def _prepare(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (SEG_WIDTH, SEG_HEIGHT))


def segment_tool(frame: np.ndarray,
                 hands: Sequence[Point] = ()) -> ToolSegment:
    """주황/황동 공구를 연결요소 단위로 분리한다.

    색상 마스크 전체의 무게중심을 쓰면 공구가 없을 때 흩어진 노이즈의 평균이
    잡혀 엉뚱한 좌표가 나온다. 연결요소로 나눈 뒤 크기·형태 조건을 만족하는
    덩어리만 공구로 인정한다.
    """
    small = _prepare(frame)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TOOL_LOWER, TOOL_UPPER)
    mask[:TOP_CUT] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_index = -1
    best_area = 0.
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < TOOL_MIN_AREA or area > TOOL_MAX_AREA:
            continue
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        if width < 4 or height < 4:
            continue
        # 공구는 자루를 포함해 길쭉하거나 머리처럼 뭉툭하다. 극단적으로 얇게
        # 번진 반사광은 제외한다.
        fill = area / max(1., width * height)
        if fill < .22:
            continue
        if area > best_area:
            best_area = area
            best_index = index

    segment = ToolSegment(area_ratio=cv2.countNonZero(mask) / SEG_AREA)
    if best_index < 0:
        return segment

    x = float(stats[best_index, cv2.CC_STAT_LEFT])
    y = float(stats[best_index, cv2.CC_STAT_TOP])
    width = float(stats[best_index, cv2.CC_STAT_WIDTH])
    height = float(stats[best_index, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[best_index]

    segment.found = True
    segment.area_ratio = best_area / SEG_AREA
    segment.clear = best_area >= TOOL_CLEAR_AREA
    segment.center = (cx / SEG_WIDTH, cy / SEG_HEIGHT)
    segment.box = (x / SEG_WIDTH, y / SEG_HEIGHT,
                   width / SEG_WIDTH, height / SEG_HEIGHT)
    segment.head = _tool_head(labels == best_index, hands)
    return segment


def _tool_head(component: np.ndarray,
               hands: Sequence[Point]) -> Optional[Point]:
    """공구 픽셀 중 손에서 가장 먼 점을 머리로 본다.

    두드릴 때 크게 움직이는 쪽은 자루를 쥔 손이 아니라 머리 쪽이므로, 머리
    좌표를 추적해야 두드림 진폭이 선명하게 나온다.
    """
    ys, xs = np.nonzero(component)
    if not len(xs):
        return None
    points = np.stack([xs / float(SEG_WIDTH), ys / float(SEG_HEIGHT)], axis=1)
    if not hands:
        # 손을 모르면 세로로 가장 위쪽 끝을 머리로 둔다.
        top = int(np.argmin(points[:, 1]))
        return float(points[top][0]), float(points[top][1])
    anchor = np.asarray(hands, dtype=np.float32)
    distances = np.linalg.norm(points[:, None, :] - anchor[None, :, :], axis=2)
    farthest = int(np.argmax(distances.min(axis=1)))
    return float(points[farthest][0]), float(points[farthest][1])


def segment_vinyl(frame: np.ndarray) -> VinylSegment:
    """빛반사가 심한 비닐 덩어리를 국소 밝기차로 분리한다.

    비닐은 투명해서 색으로는 잡히지 않는다. 대신 표면에서 빛이 강하게 반사돼
    아주 밝은 점과 어두운 부분이 좁은 범위 안에 섞이므로, 국소 표준편차가 큰
    영역을 덩어리로 묶어 비닐 후보로 삼는다.
    """
    small = _prepare(frame)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 국소 표준편차 = sqrt(E[x^2] - E[x]^2)
    mean = cv2.boxFilter(gray, -1, (7, 7))
    mean_square = cv2.boxFilter(gray * gray, -1, (7, 7))
    std = cv2.sqrt(np.maximum(mean_square - mean * mean, 0))

    specular = ((hsv[:, :, 2] >= VINYL_MIN_VALUE)
                & (hsv[:, :, 1] <= VINYL_MAX_SAT)).astype(np.uint8)
    specular[:TOP_CUT] = 0

    # 비닐의 특징은 "밝다"가 아니라 "작은 반사점이 흩뿌려져 있다"이다.
    # 넓게 이어진 밝은 면(흰 장갑, 조명 받은 평면)은 가장자리에서만 밝기차가
    # 크므로 반사점 크기로 걸러내야 오검출이 없다.
    spots, labels, stats, _ = cv2.connectedComponentsWithStats(specular, 8)
    speckle = np.zeros_like(specular)
    speckles = 0
    for index in range(1, spots):
        if stats[index, cv2.CC_STAT_AREA] <= VINYL_SPECKLE_MAX:
            speckle[labels == index] = 1
            speckles += 1
    if speckles < VINYL_MIN_SPECKLES:
        return VinylSegment()

    # 반사점 주변의 밝기차가 실제로 뚜렷한 곳만 남긴다.
    contrast = (std >= VINYL_MIN_STD).astype(np.uint8)
    near = cv2.dilate(speckle, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_and(near, contrast) * 255
    # 흩어진 반사점을 하나의 비닐 덩어리로 묶는다.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    kept = [index for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= VINYL_MIN_AREA]
    segment = VinylSegment(blobs=len(kept))
    if not kept:
        return segment

    total = float(sum(stats[i, cv2.CC_STAT_AREA] for i in kept))
    biggest = max(kept, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    cx, cy = centroids[biggest]
    x = float(stats[biggest, cv2.CC_STAT_LEFT])
    y = float(stats[biggest, cv2.CC_STAT_TOP])
    width = float(stats[biggest, cv2.CC_STAT_WIDTH])
    height = float(stats[biggest, cv2.CC_STAT_HEIGHT])

    segment.found = True
    segment.area_ratio = total / SEG_AREA
    segment.center = (cx / SEG_WIDTH, cy / SEG_HEIGHT)
    segment.box = (x / SEG_WIDTH, y / SEG_HEIGHT,
                   width / SEG_WIDTH, height / SEG_HEIGHT)
    if len(kept) > 1:
        points = np.asarray([centroids[i] for i in kept], np.float32)
        points[:, 0] /= SEG_WIDTH
        points[:, 1] /= SEG_HEIGHT
        segment.spread = float(np.linalg.norm(points - points.mean(axis=0),
                                              axis=1).max())
    return segment


# 검지~새끼 손가락의 (끝, 중간마디) 랜드마크 쌍
_FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))


def grip_strength(landmarks) -> float:
    """손가락이 감겨 막대를 쥐고 있는 정도를 0~1로 돌려준다.

    손목(0)에서 손가락 끝까지의 거리가 중간마디까지의 거리보다 짧으면 그
    손가락은 접힌 것으로 본다. 네 손가락 중 접힌 비율을 그립 강도로 쓴다.
    """
    if not landmarks:
        return 0.
    wrist = landmarks[0]
    curled = 0
    for tip_index, middle_index in _FINGERS:
        tip = landmarks[tip_index]
        middle = landmarks[middle_index]
        tip_distance = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
        middle_distance = np.hypot(middle.x - wrist.x, middle.y - wrist.y)
        if tip_distance < middle_distance:
            curled += 1
    return curled / float(len(_FINGERS))


def best_grip(hand_result) -> float:
    """검출된 손 중 가장 강하게 쥔 손의 그립 강도."""
    if not hand_result or not hand_result.hand_landmarks:
        return 0.
    return max((grip_strength(lm) for lm in hand_result.hand_landmarks),
               default=0.)


def draw_box(frame: np.ndarray, box: Optional[BBox], label: str,
             color: Tuple[int, int, int]) -> None:
    """세그멘테이션 결과를 화면에 표시해 판정 근거를 눈으로 확인하게 한다."""
    if box is None:
        return
    height, width = frame.shape[:2]
    x, y, box_width, box_height = box
    p1 = (int(x * width), int(y * height))
    p2 = (int((x + box_width) * width), int((y + box_height) * height))
    cv2.rectangle(frame, p1, p2, color, 2)
    cv2.putText(frame, label, (p1[0], max(16, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)


def draw_point(frame: np.ndarray, point: Optional[Point], label: str,
               color: Tuple[int, int, int]) -> None:
    if point is None:
        return
    height, width = frame.shape[:2]
    center = (int(point[0] * width), int(point[1] * height))
    cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (center[0] + 10, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
