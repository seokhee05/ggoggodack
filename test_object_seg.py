"""object_seg 세그멘테이션과 detector 판정에 대한 회귀 테스트.

합성 프레임으로 각 알고리즘이 의도한 대상에만 반응하는지 확인한다. 실제
현장 영상의 임계값 튜닝을 대신하지는 않지만, 리팩터링으로 판정이 조용히
망가지는 것은 막아 준다.
"""

from __future__ import annotations

import numpy as np

from object_seg import (SEG_HEIGHT, SEG_WIDTH, grip_strength, segment_tool,
                        segment_vinyl)


class _Landmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _blank(value: int = 40) -> np.ndarray:
    return np.full((SEG_HEIGHT * 2, SEG_WIDTH * 2, 3), value, np.uint8)


def _put_tool(frame: np.ndarray, cx: int, cy: int,
              width: int = 40, height: int = 26) -> None:
    """주황/황동 계열 공구 블록을 그린다 (BGR)."""
    frame[cy - height // 2:cy + height // 2,
          cx - width // 2:cx + width // 2] = (30, 120, 220)


def test_tool_found_when_present():
    frame = _blank()
    _put_tool(frame, 200, 220)
    tool = segment_tool(frame)
    assert tool.found
    assert tool.clear
    assert tool.center is not None
    # 좌표는 0~1로 정규화되어 대략 그린 위치에 있어야 한다.
    assert .25 < tool.center[0] < .40
    assert .50 < tool.center[1] < .70


def test_tool_absent_on_empty_frame():
    tool = segment_tool(_blank())
    assert not tool.found
    assert tool.center is None


def test_tiny_noise_is_not_a_tool():
    """색상 점 몇 개짜리 노이즈는 공구로 인정하지 않는다.

    실제 영상에서 색상 0.02%인 노이즈가 두드림으로 오인된 사례가 있었다.
    """
    frame = _blank()
    rng = np.random.default_rng(0)
    for _ in range(12):
        x = int(rng.integers(20, SEG_WIDTH * 2 - 20))
        y = int(rng.integers(TOP := 120, SEG_HEIGHT * 2 - 20))
        frame[y:y + 2, x:x + 2] = (30, 120, 220)
    tool = segment_tool(frame)
    assert not tool.clear


def test_tool_head_is_far_from_hand():
    """머리는 자루를 쥔 손에서 먼 쪽으로 잡혀야 한다."""
    frame = _blank()
    # 가로로 긴 공구를 그리고 왼쪽 끝을 손 위치로 준다.
    frame[210:230, 120:300] = (30, 120, 220)
    tool = segment_tool(frame, hands=[(0.20, 0.61)])
    assert tool.found and tool.head is not None
    # 손이 왼쪽에 있으므로 머리는 오른쪽 끝이어야 한다.
    assert tool.head[0] > 0.40


def test_vinyl_needs_specular_contrast():
    """균일한 밝은 면은 비닐이 아니다. 반사점과 밝기차가 함께 있어야 한다."""
    flat = _blank()
    flat[150:300, 100:400] = (230, 230, 230)
    assert not segment_vinyl(flat).found


def test_vinyl_found_on_speckled_highlights():
    frame = _blank(70)
    rng = np.random.default_rng(1)
    # 좁은 범위에 밝은 반사점을 흩뿌려 비닐 표면을 흉내낸다.
    for _ in range(400):
        x = int(rng.integers(150, 330))
        y = int(rng.integers(150, 290))
        frame[y:y + 3, x:x + 3] = (250, 250, 250)
    vinyl = segment_vinyl(frame)
    assert vinyl.found
    assert vinyl.area_ratio > 0


def test_grip_open_hand_is_low():
    # 손목(0)에서 끝마디가 중간마디보다 먼 = 편 손
    landmarks = [_Landmark(.5, .9)]
    for _ in range(21):
        landmarks.append(_Landmark(.5, .5))
    for tip, middle in ((8, 6), (12, 10), (16, 14), (20, 18)):
        landmarks[middle] = _Landmark(.5, .6)
        landmarks[tip] = _Landmark(.5, .3)
    assert grip_strength(landmarks) == 0.


def test_grip_closed_hand_is_high():
    landmarks = [_Landmark(.5, .9)]
    for _ in range(21):
        landmarks.append(_Landmark(.5, .5))
    for tip, middle in ((8, 6), (12, 10), (16, 14), (20, 18)):
        landmarks[middle] = _Landmark(.5, .3)
        landmarks[tip] = _Landmark(.5, .7)   # 끝이 손목 쪽으로 말림
    assert grip_strength(landmarks) == 1.


def test_grip_empty_is_zero():
    assert grip_strength(None) == 0.
    assert grip_strength([]) == 0.
