"""검사 엔진 판정 로직 회귀 테스트."""

from __future__ import annotations

import numpy as np

from inspection_engine import (OFFICIAL_STEPS, THRESHOLDS, ActionRecognizer,
                               InspectionEngine, WindowMetrics)


def _blank(value: int = 40) -> np.ndarray:
    return np.full((360, 640, 3), value, np.uint8)


def _driver_verdict(m: WindowMetrics):
    """detect()의 driver 분기와 같은 조건을 지표만으로 평가한다."""
    t = THRESHOLDS["driver"]
    screwing = (m.coverage >= t["coverage"] and m.tool_peak >= t["tool_ratio"]
                and m.tool_travel >= t["tool_travel"])
    tapping = (m.tap_swings >= t["tap_swings"]
               and m.tap_amplitude >= t["tap_amplitude"]
               and m.tool_clear >= t["tool_clear"]
               and m.grip >= t["grip"])
    return screwing, tapping


def test_noise_blob_does_not_pass_as_tapping():
    """공구가 보이지 않는데 두드림으로 통과하면 안 된다.

    실제 영상에서 색상 0.02%, 두드림 3회로 제품 1의 Driver 단계가 잘못
    통과한 사례를 지표 수준에서 재현한다.
    """
    m = WindowMetrics(coverage=.5, tool_peak=.0002, tool_travel=.22,
                      tap_swings=3, tap_amplitude=.05, tool_clear=.0, grip=.0)
    screwing, tapping = _driver_verdict(m)
    assert not screwing
    assert not tapping


def test_real_screwing_still_passes():
    """제품 2에서 실제로 잡혔던 값은 그대로 통과해야 한다."""
    m = WindowMetrics(coverage=.5, tool_peak=.0027, tool_travel=1.85,
                      tap_swings=7, tap_amplitude=.05, tool_clear=.8, grip=.75)
    screwing, _ = _driver_verdict(m)
    assert screwing


def test_tapping_requires_visible_tool_and_grip():
    """두드림은 공구가 보이고 자루를 쥐었을 때만 인정한다."""
    base = dict(coverage=.5, tool_peak=.0009, tool_travel=.02,
                tap_swings=6, tap_amplitude=.05)
    assert not _driver_verdict(WindowMetrics(**base, tool_clear=.0, grip=.9))[1]
    assert not _driver_verdict(WindowMetrics(**base, tool_clear=.9, grip=.1))[1]
    assert _driver_verdict(WindowMetrics(**base, tool_clear=.9, grip=.9))[1]


def test_observed_peel_values_pass_by_spread():
    """실제 영상에서 관찰된 양손 간격이 의도한 경로로 통과해야 한다.

    이전 임계값(0.55)에서는 세 번 모두 이동량 fallback으로 통과해
    근거 문구가 실제 통과 사유와 달랐다.
    """
    t = THRESHOLDS["peel"]
    for peak, gain in ((.48, .14), (.49, .22), (.52, .20)):
        assert peak >= t["pair_peak"] and gain >= t["pair_gain"]


def test_engine_stays_on_first_step_without_motion():
    """움직임 없는 빈 화면에서는 어떤 단계도 통과하지 않는다."""
    engine = InspectionEngine()
    engine.start("TEST", 0.)
    frame = _blank()
    for index in range(120):
        engine.update(frame, [], index / 30.)
    assert engine.step_index == 0
    assert all(step.status != "PASS" for step in engine.steps)


def test_engine_records_cover_direction():
    engine = InspectionEngine()
    engine.start("TEST", 0.)
    engine.recognizer.last_cover_side = "화면 왼쪽"
    engine._note_cover_side(OFFICIAL_STEPS[0])
    assert engine.cover_sides["box_open"] == "화면 왼쪽"


def test_opposite_cover_direction_raises_guidance():
    """열 때와 닫을 때 방향이 다르면 확인 지침을 남긴다."""
    engine = InspectionEngine()
    engine.start("TEST", 0.)
    engine.recognizer.last_cover_side = "화면 왼쪽"
    engine._note_cover_side(OFFICIAL_STEPS[0])
    engine.recognizer.last_cover_side = "화면 오른쪽"
    engine._note_cover_side(OFFICIAL_STEPS[-1])
    kinds = [w["type"] for w in engine.warnings]
    assert "COVER_DIRECTION" in kinds


def test_same_cover_direction_is_silent():
    engine = InspectionEngine()
    engine.start("TEST", 0.)
    for step in (OFFICIAL_STEPS[0], OFFICIAL_STEPS[-1]):
        engine.recognizer.last_cover_side = "화면 왼쪽"
        engine._note_cover_side(step)
    assert not [w for w in engine.warnings if w["type"] == "COVER_DIRECTION"]


def test_finish_marks_incomplete_steps_missing():
    engine = InspectionEngine()
    engine.start("TEST", 0.)
    result = engine.finish("video.mp4", 10.)
    assert result.final_status == "REVIEW"
    assert all(item["status"] == "MISSING" for item in result.checklist)
    assert not engine.active


def test_recognizer_window_drops_old_samples():
    recognizer = ActionRecognizer()
    frame = _blank()
    for index in range(200):
        recognizer.update(frame, [(.5, .5)], index * .1)
    span = recognizer.samples[-1].t - recognizer.samples[0].t
    assert span <= 3.3
