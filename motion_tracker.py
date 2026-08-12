"""장갑을 포함한 작업자 손/팔의 움직임을 추적한다.

MediaPipe가 손을 찾으면 그 좌표를 우선 사용하고, 찾지 못하면 프레임 간
움직임 윤곽을 사용한다. 고정 카메라 환경을 전제로 한다.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


class WorkerMotionTracker:
    def __init__(self) -> None:
        self.previous_gray = None

    def reset(self) -> None:
        self.previous_gray = None

    def detect(self, frame: np.ndarray, landmark_centers: Sequence[Point]) -> List[Point]:
        """정규화된 좌표를 최대 두 개 반환한다."""
        if landmark_centers:
            return list(landmark_centers[:2])

        height, width = frame.shape[:2]
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        if self.previous_gray is None:
            self.previous_gray = gray
            return []

        delta = cv2.absdiff(self.previous_gray, gray)
        self.previous_gray = gray
        _, mask = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, None, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 280 or area > 18000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # 화면 가장자리의 흔들림/자막 변화는 작업 동작에서 제외한다.
            cx, cy = x + w / 2, y + h / 2
            if cy < 35 or cy > 345:
                continue
            candidates.append((area, (cx / 640.0, cy / 360.0)))

        candidates.sort(reverse=True, key=lambda item: item[0])
        points = sorted((point for _, point in candidates[:2]), key=lambda p: p[0])
        # 한 물체의 윤곽이 둘로 갈라진 경우를 양손으로 오인하지 않는다.
        if len(points) == 2:
            dx = points[0][0] - points[1][0]
            dy = points[0][1] - points[1][1]
            if (dx * dx + dy * dy) ** .5 < .12:
                points = points[:1]
        return points


def draw_motion_points(frame: np.ndarray, points: Sequence[Point]) -> None:
    height, width = frame.shape[:2]
    for x, y in points:
        cv2.circle(frame, (int(x * width), int(y * height)), 12, (0, 255, 255), 3)
