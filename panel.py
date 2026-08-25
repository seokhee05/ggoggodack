"""실시간 검사와 저장 영상 검사가 함께 쓰는 검사 화면.

두 실행 경로가 같은 화면을 보여주도록 레이아웃을 한 곳에 둔다. 왼쪽은
스켈레톤·세그멘테이션이 그려진 영상, 오른쪽은 현재 단계 주변 체크리스트,
아래는 판정 근거다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_WIDTH = 1280
PANEL_HEIGHT = 620
VIDEO_WIDTH = 900
VIDEO_HEIGHT = 506
# 한 화면에 다 넣으면 글씨가 작아지므로 현재 단계 주변만 보여준다.
VISIBLE_STEPS = 8

BACKGROUND = (16, 19, 24)
TITLE_COLOR = (245, 247, 250)
DONE_COLOR = (45, 205, 120)
ACTIVE_COLOR = (255, 190, 45)
WAIT_COLOR = (155, 165, 180)
LABEL_COLOR = (235, 238, 242)
REASON_COLOR = (255, 195, 60)
HINT_COLOR = (255, 145, 90)

_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
_font_cache: dict = {}


def korean_font(size: int):
    """한글이 나오는 폰트를 크기별로 캐시해 돌려준다."""
    if size not in _font_cache:
        for candidate in _FONT_CANDIDATES:
            if candidate.exists():
                _font_cache[size] = ImageFont.truetype(str(candidate), size)
                break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _step_mark(status: str, is_current: bool, active: bool):
    if status == "PASS":
        return "● 완료", DONE_COLOR
    if active and is_current:
        return "▶ 분석 중", ACTIVE_COLOR
    return "○ 탐색 중", WAIT_COLOR


def render(frame: np.ndarray, engine, event: dict,
           header: str = "", hint: str = "") -> np.ndarray:
    """검사 화면 한 장을 만든다.

    frame은 스켈레톤·세그멘테이션이 이미 그려진 영상이다. engine에서 단계
    상태를, event에서 판정 근거를 읽어 오른쪽과 아래에 배치한다.
    """
    steps = getattr(engine, "steps", []) or []
    definitions = engine.OFFICIAL_STEPS if hasattr(engine, "OFFICIAL_STEPS") else None
    if definitions is None:
        from inspection_engine import OFFICIAL_STEPS as definitions

    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    canvas[:VIDEO_HEIGHT, :VIDEO_WIDTH] = cv2.resize(
        frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    title_font = korean_font(25)
    item_font = korean_font(17)
    small_font = korean_font(14)
    total = len(definitions)

    draw.text((925, 20), header or "직원 기준 작업 검사",
              font=title_font, fill=TITLE_COLOR)
    draw.text((925, 57), f"완료: {engine.step_index}/{total}",
              font=item_font, fill=DONE_COLOR)

    pending = next((i for i, s in enumerate(steps) if s.status != "PASS"), -1)
    start = (max(0, min(pending - 2, total - VISIBLE_STEPS)) if pending >= 0
             else max(0, total - VISIBLE_STEPS))
    y = 100
    for index in range(start, min(total, start + VISIBLE_STEPS)):
        status = steps[index].status if index < len(steps) else "WAIT"
        mark, color = _step_mark(status, index == pending,
                                 getattr(engine, "active", False))
        draw.text((925, y), mark, font=small_font, fill=color)
        # 번호는 mark에 있으므로 본문이 패널을 넘지 않게 두 줄로 접는다.
        label = definitions[index].label
        if len(label) > 24:
            draw.text((925, y + 22), label[:24], font=small_font, fill=LABEL_COLOR)
            draw.text((925, y + 42), label[24:], font=small_font, fill=LABEL_COLOR)
            y += 58
        else:
            draw.text((925, y + 22), label, font=small_font, fill=LABEL_COLOR)
            y += 48

    draw.text((20, 530), "현재 판정 근거", font=item_font, fill=REASON_COLOR)
    draw.text((20, 565), event.get("reason", "")[:95],
              font=small_font, fill=LABEL_COLOR)
    if hint:
        draw.text((960, 578), hint, font=small_font, fill=HINT_COLOR)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_segments(frame: np.ndarray, recognizer) -> None:
    """공구·비닐 세그멘테이션 결과를 영상 위에 표시한다."""
    from object_seg import draw_box, draw_point

    tool = getattr(recognizer, "last_tool", None)
    vinyl = getattr(recognizer, "last_vinyl", None)
    if tool is not None and tool.found:
        draw_box(frame, tool.box,
                 f"TOOL {'CLEAR' if tool.clear else 'DIM'}", (60, 200, 255))
        draw_point(frame, tool.head, "HEAD", (60, 200, 255))
    if vinyl is not None and vinyl.found:
        draw_box(frame, vinyl.box, f"VINYL x{vinyl.blobs}", (240, 160, 255))
