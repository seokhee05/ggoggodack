"""고정 카메라 영상에서 암레스트 후보를 자동 탐지·추적한다.

별도 ROI 입력 없이 작업대 하단의 큰 회색/검정 제품 윤곽을 찾는다. 학습된
부품 검출기가 없는 프로토타입이므로 제품이 불분명하면 억지로 PASS시키지 않고
미검출 상태로 둔다.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]


class ArmrestTracker:
    def __init__(self) -> None:
        self.history: Deque[BBox] = deque(maxlen=7)
        self.seen_streak = 0

    def reset(self) -> None:
        self.history.clear(); self.seen_streak = 0

    def detect(self, frame: np.ndarray) -> Tuple[Optional[BBox], float]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, 360))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # 암레스트의 검정/회색 표면 후보. 밝은 피부·장갑·반사 비닐은 제외한다.
        dark = cv2.inRange(hsv, np.array([0, 0, 18]), np.array([179, 145, 195]))
        dark[:70, :] = 0
        dark[:, :18] = 0; dark[:, -18:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None; best_score = 0.0
        image_area = 640.0*360.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area*.012 or area > image_area*.38:
                continue
            x,y,bw,bh = cv2.boundingRect(contour)
            if bw < 95 or bh < 35 or y+bh/2 < 155:
                continue
            rectangularity = area/max(1.0,bw*bh)
            aspect = bw/max(1.0,bh)
            if not (.55 <= aspect <= 6.5) or rectangularity < .30:
                continue
            # 작업대 중앙에 있고 가로로 긴 큰 윤곽을 우선한다.
            center_bonus = 1.0-abs((x+bw/2)/640.0-.52)
            score = (area/image_area)*2.5 + min(aspect,3.0)*.08 + rectangularity*.35 + center_bonus*.12
            if score > best_score:
                best_score=score; best=(x/640.0,y/360.0,bw/640.0,bh/360.0)

        if best is None:
            self.seen_streak=max(0,self.seen_streak-1)
            if self.seen_streak==0: self.history.clear()
            return (self._median_box() if self.history else None), 0.0
        self.history.append(best); self.seen_streak=min(30,self.seen_streak+1)
        confidence=min(1.0,best_score/1.05)*min(1.0,self.seen_streak/4.0)
        return self._median_box(), confidence

    def _median_box(self) -> Optional[BBox]:
        if not self.history: return None
        a=np.asarray(self.history,dtype=np.float32)
        return tuple(float(v) for v in np.median(a,axis=0))


def draw_armrest_box(frame: np.ndarray, box: Optional[BBox], confidence: float) -> None:
    if box is None: return
    h,w=frame.shape[:2]; x,y,bw,bh=box
    p1=(int(x*w),int(y*h)); p2=(int((x+bw)*w),int((y+bh)*h))
    color=(40,220,90) if confidence>=.45 else (0,180,255)
    cv2.rectangle(frame,p1,p2,color,3)
    cv2.putText(frame,f"ARMREST {confidence:.2f}",(p1[0],max(22,p1[1]-7)),
                cv2.FONT_HERSHEY_SIMPLEX,.62,color,2,cv2.LINE_AA)
