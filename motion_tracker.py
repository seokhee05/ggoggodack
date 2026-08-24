"""작업자 손 중심 추적.

기존 버전의 프레임 차분 fallback은 비닐, 제품, 옷의 움직임까지 손으로
오인했으므로 기본적으로 사용하지 않는다. MediaPipe가 실제로 찾은 손만
좌우 정체성을 유지한 채 평활화해 반환한다.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

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

PALM_IDS = (0, 5, 9, 13, 17)
SHOULDER_ELBOW_WRIST = {"left": (11, 13, 15), "right": (12, 14, 16)}


class _HandTrack:
    """한쪽 손의 최근 좌표 이력."""

    __slots__ = ("points", "missing")

    def __init__(self, point: Point, smoothing: int) -> None:
        self.points: Deque[Point] = deque([point], maxlen=smoothing)
        self.missing = 0

    def smoothed(self) -> Point:
        return (
            float(np.median([p[0] for p in self.points])),
            float(np.median([p[1] for p in self.points])),
        )


class WorkerMotionTracker:
    """검출된 손을 트랙에 배정해 좌우가 섞이거나 좌표가 튀는 것을 막는다."""

    def __init__(self, smoothing: int = 9, max_jump: float = 0.18,
                 max_missing: int = 4) -> None:
        self.smoothing = max(1, smoothing)
        self.max_jump = max_jump
        self.max_missing = max_missing
        self.tracks: List[_HandTrack] = []

    def reset(self) -> None:
        self.tracks.clear()

    def detect(self, frame: np.ndarray, landmark_centers: Sequence[Point]) -> List[Point]:
        del frame  # API 호환용. 배경 움직임은 더 이상 손으로 사용하지 않는다.
        detections = [
            (float(x), float(y)) for x, y in landmark_centers
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        ][:2]

        unassigned = list(range(len(detections)))
        for track in self.tracks:
            anchor = track.smoothed()
            best, best_distance = None, self.max_jump
            for index in unassigned:
                distance = math.dist(anchor, detections[index])
                if distance < best_distance:
                    best, best_distance = index, distance
            if best is None:
                track.missing += 1
            else:
                track.points.append(detections[best])
                track.missing = 0
                unassigned.remove(best)

        self.tracks = [t for t in self.tracks if t.missing <= self.max_missing]
        for index in unassigned:
            if len(self.tracks) >= 2:
                break
            self.tracks.append(_HandTrack(detections[index], self.smoothing))

        # 트랙 생성 순서를 유지해야 다음 프레임에서도 같은 손이 같은 자리에 온다.
        return [t.smoothed() for t in self.tracks if t.missing == 0]


def hand_centers(hand_result) -> List[Point]:
    """손바닥 랜드마크 평균을 손 중심 좌표로 반환한다."""
    centers = []
    for landmarks in hand_result.hand_landmarks or []:
        centers.append((
            sum(landmarks[i].x for i in PALM_IDS) / len(PALM_IDS),
            sum(landmarks[i].y for i in PALM_IDS) / len(PALM_IDS),
        ))
    return centers


def merge_hand_sources(hand_result, pose_result, min_separation: float = 0.10) -> List[Point]:
    """손 검출이 부족하면 Pose 손목으로 보완한다."""
    centers = hand_centers(hand_result)
    if len(centers) >= 2:
        return centers
    wrists = pose_wrist_centers(pose_result)
    if not centers:
        return wrists
    centers += [
        p for p in wrists
        if min(math.dist(p, q) for q in centers) > min_separation
    ]
    return centers


def arm_metrics(pose_result, min_visibility: float = 0.4) -> Dict[str, Dict[str, float]]:
    """어깨-팔꿈치-손목으로 팔이 얼마나 펴졌는지와 뻗은 방향을 계산한다.

    extension은 1.0에 가까울수록 팔이 곧게 펴진 상태, reach는 어깨 대비 손목의
    수평 방향(음수=화면 왼쪽)이다.
    """
    if not pose_result or not pose_result.pose_landmarks:
        return {}
    landmarks = pose_result.pose_landmarks[0]
    output: Dict[str, Dict[str, float]] = {}
    for side, (shoulder, elbow, wrist) in SHOULDER_ELBOW_WRIST.items():
        a, b, c = landmarks[shoulder], landmarks[elbow], landmarks[wrist]
        if min(getattr(p, "visibility", 1.0) for p in (a, b, c)) < min_visibility:
            continue
        upper = math.dist((a.x, a.y), (b.x, b.y))
        forearm = math.dist((b.x, b.y), (c.x, c.y))
        if upper + forearm < 1e-6:
            continue
        output[side] = {
            "extension": math.dist((a.x, a.y), (c.x, c.y)) / (upper + forearm),
            "reach": float(c.x - a.x),
        }
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
