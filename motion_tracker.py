"""작업자 손 중심 추적.

기존 버전의 프레임 차분 fallback은 비닐, 제품, 옷의 움직임까지 손으로
오인했으므로 기본적으로 사용하지 않는다. MediaPipe가 실제로 찾은 손만
짧게 평활화해 반환한다.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

# MediaPipe Hand Landmarker의 21개 관절 연결 구조
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class WorkerMotionTracker:
    def __init__(self, smoothing: int = 5) -> None:
        self.smoothing = max(1, smoothing)
        self.history: Deque[List[Point]] = deque(maxlen=self.smoothing)

    def reset(self) -> None:
        self.history.clear()

    def detect(self, frame: np.ndarray, landmark_centers: Sequence[Point]) -> List[Point]:
        del frame  # API 호환용. 배경 움직임은 더 이상 손으로 사용하지 않는다.
        current = sorted(
            [(float(x), float(y)) for x, y in landmark_centers if 0 <= x <= 1 and 0 <= y <= 1],
            key=lambda p: p[0],
        )[:2]
        if not current:
            self.history.clear()
            return []

        self.history.append(current)
        # 손 개수가 같은 최근 프레임끼리만 평균내어 좌우 손이 섞이지 않게 한다.
        same = [points for points in self.history if len(points) == len(current)]
        output: List[Point] = []
        for index in range(len(current)):
            output.append((
                float(np.median([points[index][0] for points in same])),
                float(np.median([points[index][1] for points in same])),
            ))
        return output


def draw_motion_points(frame: np.ndarray, points: Sequence[Point]) -> None:
    height, width = frame.shape[:2]
    for x, y in points:
        cv2.circle(frame, (int(x * width), int(y * height)), 11, (0, 255, 255), 3)
        cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 80, 255), -1)


def draw_hand_skeleton(frame: np.ndarray, hand_result) -> None:
    """손가락 21개 관절과 연결선을 손 위에 표시한다."""
    height, width = frame.shape[:2]
    for landmarks in hand_result.hand_landmarks or []:
        pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pixels[a], pixels[b], (0, 220, 255), 3, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            color = (0, 80, 255) if index in (0, 4, 8, 12, 16, 20) else (255, 255, 255)
            radius = 5 if index in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def pose_wrist_centers(pose_result, min_visibility: float = 0.45) -> List[Point]:
    """Pose의 좌우 손목(15, 16)을 손 검출 보완 좌표로 반환한다."""
    if not pose_result or not pose_result.pose_landmarks:
        return []
    landmarks = pose_result.pose_landmarks[0]
    output = []
    for index in (15, 16):
        lm = landmarks[index]
        if getattr(lm, "visibility", 1.0) >= min_visibility and 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
            output.append((float(lm.x), float(lm.y)))
    return sorted(output, key=lambda p: p[0])


def draw_pose_arms(frame: np.ndarray, pose_result) -> None:
    """예시 이미지처럼 어깨-팔꿈치-손목 연결선을 표시한다."""
    if not pose_result or not pose_result.pose_landmarks:
        return
    height, width = frame.shape[:2]
    landmarks = pose_result.pose_landmarks[0]
    # 좌/우: 어깨-팔꿈치-손목, 어깨 사이
    connections = ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12))
    pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for a, b in connections:
        if getattr(landmarks[a], "visibility", 1.0) < .4 or getattr(landmarks[b], "visibility", 1.0) < .4:
            continue
        cv2.line(frame, pixels[a], pixels[b], (255, 120, 0), 4, cv2.LINE_AA)
    for index in (11, 12, 13, 14, 15, 16):
        if getattr(landmarks[index], "visibility", 1.0) >= .4:
            cv2.circle(frame, pixels[index], 6, (0, 255, 255), -1, cv2.LINE_AA)
