"""암레스트 검사용 실시간 행동 판정 엔진.

수동 ROI를 사용하지 않는다. MediaPipe가 찾은 손 중심의 시간 변화와
Driver의 주황색 영역을 조합해 공식 7개 체크리스트의 행동 신호를 판정한다.
이 엔진은 제품 불량 자체가 아니라 작업자가 검사 행동을 수행했는지 확인한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class StepDefinition:
    id: str
    label: str
    detector: str
    required_hits: int = 1


OFFICIAL_STEPS: List[StepDefinition] = [
    StepDefinition("knob", "1. Knob 작동 이상 확인", "one_hand_operate"),
    StepDefinition("hinge", "2. Hinge 작동 이상 확인", "two_hand_open_close"),
    StepDefinition("screw", "3. Screw 외관·누락 확인 (2Point)", "driver", 2),
    StepDefinition("damper", "4. Damper 누락 확인 (2Point)", "push", 2),
    StepDefinition("rib", "5. Rib 파손 조각 확인 (흔들기)", "shake"),
    StepDefinition("operation", "6. Knob·Hinge 굽힘/펴기 확인", "bend_unbend"),
    StepDefinition("gap", "7. Lining–Skin Gap 확인", "edge_trace"),
]


@dataclass
class MotionSample:
    t: float
    hands: Tuple[Point, ...]
    driver_ratio: float


@dataclass
class StepResult:
    id: str
    label: str
    status: str = "WAIT"
    hits: int = 0
    required_hits: int = 1
    completed_at_seconds: Optional[float] = None


@dataclass
class InspectionResult:
    barcode: str
    started_at: str
    ended_at: str
    duration_seconds: float
    final_status: str
    checklist: List[Dict]
    warnings: List[Dict]
    video: str
    prototype_notice: str = (
        "손 동작과 Driver 색상 신호로 검사 행동 수행 여부를 판정합니다. "
        "제품의 물리적 양불량 자체를 판정하지는 않습니다."
    )


class ActionRecognizer:
    """최근 3초의 손 궤적으로 검사 동작을 판정한다."""

    def __init__(self) -> None:
        self.samples: Deque[MotionSample] = deque(maxlen=180)
        self.last_hit_time: Dict[str, float] = {}
        self.signal_latched: Dict[str, bool] = {}

    def reset(self) -> None:
        self.samples.clear()
        self.last_hit_time.clear()
        self.signal_latched.clear()

    @staticmethod
    def _driver_ratio(frame: np.ndarray) -> float:
        """영상에서 주황색 Driver가 차지하는 비율을 계산한다."""
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 90, 90], dtype=np.uint8)
        upper = np.array([28, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return float(cv2.countNonZero(mask)) / float(mask.size)

    def update(self, frame: np.ndarray, hands: Sequence[Point], t: float) -> None:
        normalized = tuple(sorted(hands, key=lambda p: p[0])[:2])
        self.samples.append(MotionSample(t, normalized, self._driver_ratio(frame)))
        while self.samples and t - self.samples[0].t > 3.2:
            self.samples.popleft()

    def _window(self, seconds: float) -> List[MotionSample]:
        if not self.samples:
            return []
        end = self.samples[-1].t
        return [s for s in self.samples if end - s.t <= seconds]

    @staticmethod
    def _primary_path(samples: Sequence[MotionSample]) -> np.ndarray:
        points = [s.hands[0] for s in samples if s.hands]
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _pair_distances(samples: Sequence[MotionSample]) -> np.ndarray:
        values = []
        for sample in samples:
            if len(sample.hands) == 2:
                a, b = sample.hands
                values.append(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _reversals(values: np.ndarray, minimum_delta: float = 0.012) -> int:
        if values.size < 5:
            return 0
        velocity = np.diff(values)
        velocity[np.abs(velocity) < minimum_delta] = 0
        signs = np.sign(velocity)
        signs = signs[signs != 0]
        return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0

    def _cooldown_ready(self, detector: str, now: float, seconds: float = 1.0) -> bool:
        return now - self.last_hit_time.get(detector, -999.0) >= seconds

    def detect(self, detector: str) -> Tuple[bool, float, str]:
        """(감지 여부, 신뢰 점수, 근거 문구)를 반환한다."""
        if not self.samples:
            return False, 0.0, "손을 찾는 중"
        now = self.samples[-1].t
        short = self._window(1.5)
        medium = self._window(2.8)
        path = self._primary_path(medium)
        pair = self._pair_distances(medium)
        driver_peak = max((s.driver_ratio for s in short), default=0.0)

        if detector == "driver":
            score = min(1.0, driver_peak / 0.006)
            detected = driver_peak >= 0.004
            reason = f"Driver 색상 {driver_peak * 100:.2f}%"
        elif detector in {"two_hand_open_close", "bend_unbend"}:
            if pair.size < 12:
                return False, 0.0, "양손을 찾는 중"
            spread = float(np.percentile(pair, 90) - np.percentile(pair, 10))
            reversals = self._reversals(pair, 0.01)
            needed = 0.16 if detector == "bend_unbend" else 0.11
            score = min(1.0, spread / needed)
            detected = spread >= needed and reversals >= 1
            reason = f"양손 거리 변화 {spread:.2f}, 방향전환 {reversals}회"
        elif detector == "shake":
            if path.shape[0] < 15:
                return False, 0.0, "손 움직임을 찾는 중"
            x_rev = self._reversals(path[:, 0], 0.012)
            y_rev = self._reversals(path[:, 1], 0.012)
            rev = x_rev + y_rev
            score = min(1.0, rev / 7.0)
            detected = rev >= 6
            reason = f"반복 흔들림 {rev}회"
        elif detector == "push":
            if path.shape[0] < 12:
                return False, 0.0, "누름 동작을 찾는 중"
            y = path[:, 1]
            amplitude = float(np.percentile(y, 90) - np.percentile(y, 10))
            reversals = self._reversals(y, 0.008)
            score = min(1.0, amplitude / 0.10)
            detected = amplitude >= 0.07 and reversals >= 1
            reason = f"누름·복귀 이동 {amplitude:.2f}"
        elif detector == "one_hand_operate":
            if path.shape[0] < 15:
                return False, 0.0, "작동 동작을 찾는 중"
            x_amp = float(np.ptp(path[:, 0]))
            y_amp = float(np.ptp(path[:, 1]))
            reversals = self._reversals(path[:, 0], 0.008) + self._reversals(path[:, 1], 0.008)
            score = min(1.0, (x_amp + y_amp) / 0.22)
            detected = x_amp + y_amp >= 0.16 and reversals >= 2
            reason = f"손 작동 범위 {x_amp + y_amp:.2f}"
        elif detector == "edge_trace":
            if path.shape[0] < 20:
                return False, 0.0, "Gap 확인 동작을 찾는 중"
            displacement = float(np.linalg.norm(path[-1] - path[0]))
            travel = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            smoothness = displacement / max(travel, 1e-6)
            score = min(1.0, travel / 0.7) * min(1.0, smoothness / 0.35)
            detected = travel >= 0.45 and smoothness >= 0.22
            reason = f"가장자리 추적 거리 {travel:.2f}"
        else:
            return False, 0.0, "알 수 없는 판정기"

        # 하나의 길게 이어진 행동을 2Point 두 번으로 중복 계산하지 않는다.
        # 신호가 한 번 사라진 뒤 다시 나타나야 새로운 hit로 인정한다.
        raw_detected = detected
        if not raw_detected:
            self.signal_latched[detector] = False
        detected = (
            raw_detected
            and not self.signal_latched.get(detector, False)
            and self._cooldown_ready(detector, now, 0.7)
        )
        if detected:
            self.signal_latched[detector] = True
            self.last_hit_time[detector] = now
        return detected, score, reason

    def strong_wrong_action(self, expected_detector: str) -> Optional[str]:
        """현재 단계와 명백히 다른 강한 행동만 경고한다."""
        if not self.samples:
            return None
        driver_peak = max((s.driver_ratio for s in self._window(0.8)), default=0.0)
        if expected_detector != "driver" and driver_peak >= 0.008:
            return "현재 순서가 아닌데 Driver 사용이 감지되었습니다."
        return None


class InspectionEngine:
    def __init__(self, nominal_seconds: float = 70.0, warning_seconds: float = 18.0) -> None:
        self.nominal_seconds = nominal_seconds
        self.warning_seconds = warning_seconds
        self.recognizer = ActionRecognizer()
        self.active = False
        self.barcode = ""
        self.started_monotonic = 0.0
        self.started_iso = ""
        self.step_index = 0
        self.steps: List[StepResult] = []
        self.warnings: List[Dict] = []
        self.last_progress_time = 0.0
        self.last_warning_time = -999.0

    def start(self, barcode: str, now: float) -> None:
        self.active = True
        self.barcode = barcode
        self.started_monotonic = now
        self.started_iso = datetime.now().isoformat(timespec="seconds")
        self.step_index = 0
        self.steps = [
            StepResult(s.id, s.label, required_hits=s.required_hits)
            for s in OFFICIAL_STEPS
        ]
        self.warnings = []
        self.last_progress_time = now
        self.last_warning_time = -999.0
        self.recognizer.reset()

    @property
    def current_definition(self) -> Optional[StepDefinition]:
        return OFFICIAL_STEPS[self.step_index] if self.step_index < len(OFFICIAL_STEPS) else None

    @property
    def current_result(self) -> Optional[StepResult]:
        return self.steps[self.step_index] if self.step_index < len(self.steps) else None

    def update(self, frame: np.ndarray, hands: Sequence[Point], now: float) -> Dict:
        if not self.active:
            return {"event": "idle", "message": "바코드를 스캔해 주세요."}
        self.recognizer.update(frame, hands, now)
        definition = self.current_definition
        result = self.current_result
        if definition is None or result is None:
            return {"event": "complete", "message": "모든 검사가 완료되었습니다."}

        detected, score, reason = self.recognizer.detect(definition.detector)
        if detected:
            result.hits += 1
            self.last_progress_time = now
            if result.hits >= result.required_hits:
                result.status = "PASS"
                result.completed_at_seconds = round(now - self.started_monotonic, 2)
                self.step_index += 1
                if self.step_index >= len(self.steps):
                    return {"event": "complete", "message": "7개 검사가 모두 완료되었습니다."}
                return {
                    "event": "step_complete",
                    "message": f"완료: {result.label}",
                    "next": self.current_definition.label,
                }
            return {
                "event": "hit",
                "message": f"{result.label}: {result.hits}/{result.required_hits} Point 확인",
            }

        wrong = self.recognizer.strong_wrong_action(definition.detector)
        if wrong and now - self.last_warning_time >= 3.0:
            self.last_warning_time = now
            warning = {
                "time_seconds": round(now - self.started_monotonic, 2),
                "type": "WRONG_ORDER",
                "message": wrong,
                "expected": definition.label,
            }
            self.warnings.append(warning)
            return {"event": "warning", "message": wrong}

        waiting = now - self.last_progress_time
        if waiting >= self.warning_seconds and now - self.last_warning_time >= 8.0:
            self.last_warning_time = now
            warning = {
                "time_seconds": round(now - self.started_monotonic, 2),
                "type": "POSSIBLE_MISSING",
                "message": f"검사 지연 또는 누락 가능성: {definition.label}",
                "expected": definition.label,
            }
            self.warnings.append(warning)
            return {"event": "warning", "message": warning["message"]}

        return {
            "event": "progress",
            "message": definition.label,
            "score": round(score, 3),
            "reason": reason,
            "elapsed": round(now - self.started_monotonic, 1),
        }

    def finish(self, video_path: str, now: float, forced: bool = False) -> InspectionResult:
        for result in self.steps:
            if result.status != "PASS":
                result.status = "MISSING"
        passed = all(s.status == "PASS" for s in self.steps)
        final_status = "PASS" if passed and not forced else "NG"
        output = InspectionResult(
            barcode=self.barcode,
            started_at=self.started_iso,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            duration_seconds=round(now - self.started_monotonic, 2),
            final_status=final_status,
            checklist=[asdict(s) for s in self.steps],
            warnings=list(self.warnings),
            video=video_path,
        )
        self.active = False
        return output
