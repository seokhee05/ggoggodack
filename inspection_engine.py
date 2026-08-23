"""직원 기준영상(check.mp4)의 반복 작업 순서를 따르는 검사 엔진."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

WINDOW_SECONDS = 3.2
WINDOW_MAX_SAMPLES = 180
MIN_SAMPLES = 15
# 한 프레임에 손이 이만큼 넘게 튀면 추적이 바뀐 것으로 보고 이동량에서 제외한다.
MAX_STEP = 0.18

# 현장 튜닝 지점. 각 detector의 통과 조건과 점수 정규화 기준을 모아 둔다.
THRESHOLDS: Dict[str, Dict[str, float]] = {
    "cover_move": {"extension": .93, "extension_travel": .25,
                   "coverage": .35, "travel": .48, "span": .18},
    "large_move": {"coverage": .35, "travel": .48, "span": .18},
    "lift_visual": {"coverage": .40, "travel": .52, "span": .20},
    "edge_check": {"coverage": .45, "travel": .62, "span": .24},
    "two_hand_operate": {"two": .28, "pair_span": .09, "travel": .34},
    "peel": {"two": .30, "pair_peak": .45, "pair_gain": .14,
             "travel": .70, "span": .28},
    # tap_ratio: 노이즈 블롭이 두드림으로 오인되지 않도록 공구가 실제로
    # 보일 때만 두드림을 인정한다.
    "driver": {"coverage": .25, "tool_ratio": .0012, "tool_travel": .035,
               "tap_swings": 4, "tap_amplitude": .02, "tap_ratio": .0008},
}


def _why(primary: bool, primary_name: str,
         fallback: bool, fallback_name: str) -> str:
    """어느 조건으로 통과했는지 근거 앞에 표시해 임계값 튜닝을 돕는다."""
    if primary:
        return primary_name
    if fallback:
        return fallback_name
    return "미충족"


@dataclass(frozen=True)
class StepDefinition:
    id: str
    label: str
    detector: str
    required_hits: int = 1
    minimum_seconds: float = .8


def _build_official_steps() -> List[StepDefinition]:
    steps = [StepDefinition("box_open", "준비. 박스 덮개·보호망 열기", "cover_move", 1, 1.5)]
    for n in range(1, 4):
        steps += [
            StepDefinition(f"item{n}_visual", f"제품 {n}. 들어 올려 외관·흔들림 확인", "lift_visual", 1, 1.5),
            StepDefinition(f"item{n}_unwrap", f"제품 {n}. 비닐 벗기기", "peel", 1, 1.),
            StepDefinition(f"item{n}_driver", f"제품 {n}. Driver Screw 2Point 확인", "driver", 2, 1.),
            StepDefinition(f"item{n}_operate", f"제품 {n}. 뒤집어 Knob·Hinge 작동 확인", "two_hand_operate", 1, 1.5),
            StepDefinition(f"item{n}_under", f"제품 {n}. Damper·Rib·이음부·Gap 확인", "edge_check", 1, 1.5),
            StepDefinition(f"item{n}_return", f"제품 {n}. 포장 후 박스에 반환", "large_move", 1, 1.2),
        ]
    steps.append(StepDefinition("box_close", "마무리. 보호망·덮개 닫기", "cover_move", 1, 1.5))
    return steps


OFFICIAL_STEPS: List[StepDefinition] = _build_official_steps()


@dataclass
class MotionSample:
    t: float
    hands: Tuple[Point, ...]
    tool_ratio: float
    tool_center: Optional[Point]
    extension: float
    reach: float


@dataclass
class StepResult:
    id: str
    label: str
    status: str = "WAIT"
    hits: int = 0
    required_hits: int = 1
    completed_at_seconds: Optional[float] = None
    evidence: str = ""
    confidence: float = 0.


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
    reference_workflow: str = "직원 기준영상 기반: 박스 열기 + 제품 3개 반복 검사 + 박스 닫기"
    prototype_notice: str = "검사 행동과 순서를 확인하는 프로토타입이며 실제 부품 양불량은 REVIEW 대상입니다."


@dataclass
class WindowMetrics:
    """최근 관찰 창에서 뽑아낸 판정용 지표."""

    coverage: float = 0.
    two: float = 0.
    travel: float = 0.
    span: float = 0.
    pair_span: float = 0.
    pair_peak: float = 0.
    pair_gain: float = 0.
    tool_travel: float = 0.
    tool_peak: float = 0.
    tap_swings: int = 0
    tap_amplitude: float = 0.
    extension: float = 0.
    reach: float = 0.


class ActionRecognizer:
    def __init__(self) -> None:
        self.samples: Deque[MotionSample] = deque(maxlen=WINDOW_MAX_SAMPLES)

    def reset(self) -> None:
        self.samples.clear()

    def clear_window(self) -> None:
        self.samples.clear()

    @staticmethod
    def _tool_signal(frame) -> Tuple[float, Optional[Point]]:
        """주황색 계열 공구(Driver·황동 망치) 픽셀 비율과 중심을 구한다."""
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5, 105, 95]), np.array([28, 255, 255]))
        mask[:48] = 0
        count = cv2.countNonZero(mask)
        ratio = count / float(mask.size)
        if count < 10:
            return ratio, None
        moments = cv2.moments(mask, True)
        if not moments["m00"]:
            return ratio, None
        return ratio, (moments["m10"] / moments["m00"] / 320.,
                       moments["m01"] / moments["m00"] / 180.)

    def update(self, frame, hands: Sequence[Point], t: float,
               arms: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        ratio, center = self._tool_signal(frame)
        best_arm = max(arms.values(), key=lambda a: a["extension"]) if arms else None
        self.samples.append(MotionSample(
            t=t,
            hands=tuple(hands[:2]),
            tool_ratio=ratio,
            tool_center=center,
            extension=best_arm["extension"] if best_arm else 0.,
            reach=best_arm["reach"] if best_arm else 0.,
        ))
        while self.samples and t - self.samples[0].t > WINDOW_SECONDS:
            self.samples.popleft()

    def _metrics(self, samples: List[MotionSample]) -> WindowMetrics:
        m = WindowMetrics()
        m.coverage = sum(bool(x.hands) for x in samples) / len(samples)
        m.two = sum(len(x.hands) >= 2 for x in samples) / len(samples)

        points = np.asarray([x.hands[0] for x in samples if x.hands], np.float32)
        if len(points) > 1:
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            # 추적이 뒤바뀌며 생기는 순간 점프는 이동량으로 세지 않는다.
            m.travel = float(steps[steps <= MAX_STEP].sum())
            m.span = float(np.ptp(points[:, 0]) + np.ptp(points[:, 1]))

        pair = np.asarray([math.dist(x.hands[0], x.hands[1])
                           for x in samples if len(x.hands) >= 2], np.float32)
        if len(pair) > 2:
            m.pair_span = float(np.ptp(pair))
            m.pair_peak = float(pair.max())
            m.pair_gain = float(pair.max() - np.median(pair))

        tool = np.asarray([x.tool_center for x in samples if x.tool_center], np.float32)
        if len(tool) > 1:
            m.tool_travel = float(np.linalg.norm(np.diff(tool, axis=0), axis=1).sum())
            # 두드림은 공구 머리의 y좌표가 위아래로 반복 반전하는 것으로 본다.
            dy = np.diff(tool[:, 1])
            significant = dy[np.abs(dy) > .004]
            if len(significant) > 1:
                m.tap_swings = int(np.count_nonzero(np.diff(np.sign(significant))))
                m.tap_amplitude = float(np.ptp(tool[:, 1]))
        m.tool_peak = max((x.tool_ratio for x in samples), default=0.)

        m.extension = max((x.extension for x in samples), default=0.)
        m.reach = next((x.reach for x in samples if x.extension == m.extension), 0.)
        return m

    def detect(self, detector: str) -> Tuple[bool, float, str]:
        samples = list(self.samples)
        if len(samples) < MIN_SAMPLES:
            return False, 0., "동작을 관찰하는 중"
        m = self._metrics(samples)
        t = THRESHOLDS.get(detector, THRESHOLDS["large_move"])

        if detector == "driver":
            screwing = (m.coverage >= t["coverage"] and m.tool_peak >= t["tool_ratio"]
                        and m.tool_travel >= t["tool_travel"])
            # 망치로 두드리는 동작도 같은 2Point 확인 행동으로 인정하되,
            # 공구가 실제로 보일 때만 센다. 색상이 거의 없으면 노이즈 블롭의
            # 중심점 떨림이 두드림으로 오인된다.
            tapping = (m.tap_swings >= t["tap_swings"]
                       and m.tap_amplitude >= t["tap_amplitude"]
                       and m.tool_peak >= t["tap_ratio"])
            raw = screwing or tapping
            score = max(min(1., m.tool_peak / .003) * min(1., m.tool_travel / .10),
                        min(1., m.tap_swings / 6.) * min(1., m.tap_amplitude / .06))
            reason = (f"[{_why(screwing, 'Screw', tapping, '두드림')}] "
                      f"공구 이동 {m.tool_travel:.2f}, 색상 {m.tool_peak * 100:.2f}%, "
                      f"두드림 {m.tap_swings}회")
        elif detector == "peel":
            # 비닐을 벗길 때는 양손 간격이 다른 동작보다 크게 벌어진다.
            spread = (m.two >= t["two"] and m.pair_peak >= t["pair_peak"]
                      and m.pair_gain >= t["pair_gain"])
            bulk = m.travel >= t["travel"] and m.span >= t["span"]
            raw = spread or bulk
            score = max(min(1., m.pair_peak / .75) * min(1., m.pair_gain / .30),
                        min(1., m.travel / .90))
            reason = (f"[{_why(spread, '양손벌림', bulk, '이동량')}] "
                      f"양손 최대 간격 {m.pair_peak:.2f}, 증가폭 {m.pair_gain:.2f}, "
                      f"이동 {m.travel:.2f}")
        elif detector == "cover_move":
            # 덮개가 어느 쪽으로 열리든 팔이 곧게 뻗는 동작으로 잡는다.
            reaching = m.extension >= t["extension"] and m.travel >= t["extension_travel"]
            bulk = (m.coverage >= t["coverage"]
                    and m.travel >= t["travel"] and m.span >= t["span"])
            raw = reaching or bulk
            score = max(min(1., m.extension / .98) * min(1., m.travel / .50),
                        min(1., m.travel / .80) * min(1., m.span / .32))
            side = "화면 왼쪽" if m.reach < 0 else "화면 오른쪽"
            reason = (f"[{_why(reaching, '팔뻗음', bulk, '이동량')}] "
                      f"팔 펴짐 {m.extension:.2f}({side} 방향), 이동 {m.travel:.2f}")
        elif detector == "two_hand_operate":
            raw = (m.two >= t["two"] and m.pair_span >= t["pair_span"]
                   and m.travel >= t["travel"])
            score = min(1., m.pair_span / .16) * min(1., m.travel / .65)
            reason = f"양손 작동 변화 {m.pair_span:.2f}, 이동 {m.travel:.2f}"
        elif detector == "lift_visual":
            raw = (m.coverage >= t["coverage"] and m.travel >= t["travel"]
                   and m.span >= t["span"])
            score = min(1., m.travel / .90) * min(1., m.span / .35)
            reason = f"들기·회전 이동 {m.travel:.2f}, 범위 {m.span:.2f}"
        elif detector == "edge_check":
            raw = (m.coverage >= t["coverage"] and m.travel >= t["travel"]
                   and m.span >= t["span"])
            score = min(1., m.travel / .95) * min(1., m.span / .38)
            reason = f"하부·가장자리 탐색 {m.travel:.2f}, 범위 {m.span:.2f}"
        else:
            raw = (m.coverage >= t["coverage"] and m.travel >= t["travel"]
                   and m.span >= t["span"])
            score = min(1., m.travel / .80) * min(1., m.span / .32)
            reason = f"제품·덮개 이동 {m.travel:.2f}, 범위 {m.span:.2f}"

        return raw, score, reason


class InspectionEngine:
    def __init__(self, nominal_seconds: float = 70., warning_seconds: float = 18.) -> None:
        self.nominal_seconds = nominal_seconds
        self.warning_seconds = warning_seconds
        self.recognizer = ActionRecognizer()
        self.active = False
        self.barcode = ""
        self.steps: List[StepResult] = []
        self.warnings: List[Dict] = []
        self.step_index = 0
        self.started_monotonic = 0.
        self.step_started_time = 0.
        self.started_iso = ""
        self.last_warning_time = -999.

    def start(self, barcode: str, now: float) -> None:
        self.active = True
        self.barcode = barcode
        self.started_monotonic = now
        self.step_started_time = now
        self.started_iso = datetime.now().isoformat(timespec="seconds")
        self.step_index = 0
        self.warnings = []
        self.last_warning_time = -999.
        self.steps = [StepResult(x.id, x.label, required_hits=x.required_hits)
                      for x in OFFICIAL_STEPS]
        self.recognizer.reset()

    @property
    def current_definition(self) -> Optional[StepDefinition]:
        return OFFICIAL_STEPS[self.step_index] if self.step_index < len(OFFICIAL_STEPS) else None

    @property
    def current_result(self) -> Optional[StepResult]:
        return self.steps[self.step_index] if self.step_index < len(self.steps) else None

    def update(self, frame, hands: Sequence[Point], now: float,
               arms: Optional[Dict[str, Dict[str, float]]] = None) -> Dict:
        if not self.active:
            return {"event": "idle", "message": "바코드를 스캔해 주세요.", "reason": ""}
        if self.step_index >= len(self.steps):
            return {"event": "complete", "message": "직원 기준 작업 순서가 모두 확인되었습니다.", "reason": ""}

        self.recognizer.update(frame, hands, now, arms)
        definition = self.current_definition
        result = self.current_result
        elapsed = now - self.step_started_time

        if elapsed < definition.minimum_seconds:
            return {"event": "progress", "message": definition.label,
                    "reason": f"현재 단계 관찰 {elapsed:.1f}초"}

        detected, score, reason = self.recognizer.detect(definition.detector)
        if detected:
            result.hits += 1
            result.evidence = reason
            result.confidence = round(score, 3)
            self.recognizer.clear_window()
            self.step_started_time = now
            if result.hits >= result.required_hits:
                result.status = "PASS"
                result.completed_at_seconds = round(now - self.started_monotonic, 2)
                self.step_index += 1
                if self.step_index >= len(self.steps):
                    return {"event": "complete", "message": "모든 제품의 기준 작업이 확인되었습니다.",
                            "reason": reason}
                return {"event": "step_complete", "message": "완료: " + result.label,
                        "reason": reason, "next": self.current_definition.label}
            return {"event": "hit", "message": f"{result.label}: {result.hits}/{result.required_hits} Point",
                    "reason": reason}

        if elapsed >= self.warning_seconds and now - self.last_warning_time >= 8:
            self.last_warning_time = now
            warning = {"time_seconds": round(now - self.started_monotonic, 2),
                       "type": "POSSIBLE_MISSING",
                       "message": "현재 작업이 확인되지 않습니다.",
                       "expected": definition.label}
            self.warnings.append(warning)
            return {"event": "warning", "message": warning["message"], "reason": reason}

        return {"event": "progress", "message": "확인 중: " + definition.label,
                "reason": reason, "score": round(score, 3)}

    def finish(self, video_path: str, now: float, forced: bool = False) -> InspectionResult:
        for step in self.steps:
            if step.status != "PASS":
                step.status = "MISSING"
        passed = all(step.status == "PASS" for step in self.steps)
        result = InspectionResult(
            barcode=self.barcode,
            started_at=self.started_iso,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            duration_seconds=round(now - self.started_monotonic, 2),
            final_status="PASS" if passed and not forced else "REVIEW",
            checklist=[asdict(step) for step in self.steps],
            warnings=list(self.warnings),
            video=video_path,
        )
        self.active = False
        return result
