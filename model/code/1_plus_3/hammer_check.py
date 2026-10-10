"""망치 타격 검사. 파일 하나로 끝난다.

쓰는 법 (합치는 쪽에서):

    from hammer_check import HammerCheck

    check = HammerCheck("hammer.pt")
    for t, frame in 영상:
        check.feed(frame, t)            # 프레임마다 한 번
    통과, 근거 = check.verdict()
    check.reset()                       # 다음 제품으로 넘어갈 때

판정하는 것: 암레스트 옆면 연결부를 **두 군데** 내려찍었는가.

왜 이렇게 보는가. 화면 좌표는 아래로 갈수록 y 가 크다. 망치가 내려오면
cy 가 커지고 부딪혀 튕기면 다시 작아진다. 그래서 타격 한 번은 **cy 의
봉우리** 하나다. 좌표를 못 박지 않으므로 카메라를 옮겨도 따라간다.

여기 있는 숫자는 전부 실제 촬영본에서 재서 넣은 것이다. 근거는 각 상수
옆에 적어 두었다. 임계값을 옮기려면 그 숫자부터 다시 재는 것이 맞다.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

# ---- 검출 ---------------------------------------------------------------
IMAGE_SIZE = 512
# 낮게 잡아 두고 아래 구간 규칙으로 걸러낸다. 타격 순간은 모션블러 때문에
# 오히려 신뢰도가 낮다(실측 0.23~0.25). 한 문턱으로 자르면 그 순간이
# 통째로 날아간다.
TRACK_CONF = .15
SEED_CONF = .40

# ---- 구간(한 번 집어 든 것) --------------------------------------------
# 확실한 프레임 하나가 구간을 열고, 열린 구간 안에서는 흐릿한 프레임도
# 따라간다. 씨앗이 없는 구간은 통째로 버린다. 실측에서 오검출은 전부
# 외톨이 저신뢰 프레임이었다.
RUN_GAP = .15          # 이만큼 끊기면 다른 구간
RUN_SECONDS = .30      # 진짜 최소 0.33초, 헛것 최대 0.27초
RUN_CONF = .60         # 구간을 여는 신뢰도

# 같은 구간 안에서 상자가 이만큼보다 작아지면 망치가 아닌 다른 물건이다.
#
# crop015 에서 작업자가 망치를 내려놓은 뒤, 화면 오른쪽 끝의 작은 검은
# 부품이 0.42~0.55 로 망치라고 잡혔다. 상자 크기를 재 보면 0.63~1.55%
# 인데 진짜 망치는 3.84~6.15% 였다. 신뢰도만으로는 안 걸러진다.
#
# 절대 크기로는 못 잰다. 카메라마다 망치 상자 중앙값이 다르다.
#   crop003  검출 114회  4.103%
#   crop18   검출 298회  9.203%
#   crop20   검출 342회  9.533%
# 그래서 같은 구간 안에서 견준다. 내려찍는 순간은 모션블러로 상자가
# 작아지지만, 실측에서 그때도 구간 중앙값의 44%(crop003 계열),
# 80%(crop015) 였다.
SIZE_SHARE = .30

# ---- 타격 세기 ----------------------------------------------------------
# crop001 의 실제 타격은 cy 가 0.13~0.17 움직였고 손 옮김은 0.02~0.05
# 였다. 0.05 를 경계로 두면 둘이 갈린다.
MIN_RISE = .05
MIN_FALL = .03
# 타격 간격이 0.25초까지 좁았다. 이보다 큰 값을 쓰면 빠른 연타의 뒷타를
# 삼킨다(실제로 4번을 2번으로 세는 버그가 있었다).
MIN_INTERVAL = .12

# 짧게 들었다 치는 타격을 놓치던 자리.
#
# MIN_RISE 하나로 자르면 "내려온 폭"만 본다. 그런데 망치를 조금만 들고
# 치면 내려온 폭은 작아도 부딪혀 튕기는 폭은 그대로 크다. crop015 세
# 번째 타격이 그랬다. 내려온 폭 0.047(문턱 0.05 에 0.003 모자람), 튕긴
# 폭 0.096. 내려찍은 것이 분명한데 후보가 아예 열리지 않아 사라졌다.
#
# 그래서 두 갈래로 센다. 내려온 폭이 MIN_RISE 를 넘으면 그것만으로
# 타격이고, 못 넘으면 **내려온 폭 + 튕긴 폭**이 MIN_SWING 을 넘어야
# 타격이다. 손을 옮기는 잔움직임은 양쪽 다 작아서 이 합을 못 넘는다.
#
#   crop001 실제 타격   내려온 폭 0.13~0.17  → 합 0.16 이상
#   손 옮김             한쪽 0.02~0.05       → 합 0.10 이하
#   crop015 놓친 타격   0.047 + 0.096        → 합 0.143
MIN_RISE_SEED = .03
MIN_SWING = .12

# ---- 통과 조건 ----------------------------------------------------------
# 규정은 연결부 두 곳이므로 내려찍기도 2회를 기대한다. 실측(30fps)에서
# 제품 3개 모두 2회였다.
MIN_STRIKES = 2
MIN_SECONDS = .30
# 제품이 바뀌면 망치가 1초 이상 사라진다. 그 전에는 다음 제품 몫으로
# 세지 않는다. 한 번 길게 치는 것이 두 제품으로 세이던 자리다.
GAP_SECONDS = 1.


class Mark(NamedTuple):
    """한 프레임에서 본 망치."""
    time: float
    cx: float
    cy: float


class Strike(NamedTuple):
    time: float
    cx: float
    cy: float
    depth: float


def find_strikes(marks: Sequence[Mark], min_rise: float = MIN_RISE,
                 min_fall: float = MIN_FALL,
                 min_interval: float = MIN_INTERVAL,
                 min_seed: float = MIN_RISE_SEED,
                 min_swing: float = MIN_SWING) -> List[Strike]:
    """cy 의 봉우리를 센다. 봉우리 = 망치가 가장 낮게 내려간 순간."""
    if len(marks) < 3:
        return []
    heights = [m.cy for m in marks]

    strikes: List[Strike] = []
    valley = heights[0]           # 봉우리 직전의 가장 높은 지점(= cy 최소)
    peak: Optional[int] = None    # 아직 확정 안 된 봉우리 후보

    def record(at: int, floor: float) -> None:
        moment = marks[at].time
        if strikes and moment - strikes[-1].time < min_interval:
            return
        strikes.append(Strike(moment, marks[at].cx, marks[at].cy,
                              heights[at] - floor))

    for index, height in enumerate(heights):
        if peak is None:
            if height < valley:
                valley = height
            elif height - valley >= min(min_rise, min_seed):
                peak = index          # 얕게 열어 두고 튕긴 폭까지 보고 정한다
        else:
            if height >= heights[peak]:
                peak = index
                continue
            rise = heights[peak] - valley
            fall = heights[peak] - height
            if fall < min_fall:
                continue              # 아직 덜 튕겼다. 후보를 들고 기다린다
            if rise >= min_rise or rise + fall >= min_swing:
                record(peak, valley)
                valley = height
            # 못 넘긴 후보는 버리는데, valley 는 그대로 둔다. 내려오는
            # 중의 잔떨림이었다면 아직 더 내려갈 것이고, 그 깊이는 처음
            # 지점부터 재야 맞다.
            peak = None
    # 구간 끝에서 아직 안 내려온 봉우리도 타격이다. 부딪힌 자리에 망치를
    # 둔 채 화면에서 사라지는 경우가 있다. 튕긴 폭이 없으니 이때는
    # 내려온 폭만으로 판단한다.
    if peak is not None and heights[peak] - valley >= min_rise:
        record(peak, valley)
    return strikes


class HammerCheck:
    """프레임을 넣으면 타격을 세고, 두 번 쳤는지 답한다."""

    def __init__(self, weights: str = "hammer.pt",
                 conf: float = TRACK_CONF) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self._model = None
        self.reason_off = ""
        # 화면에 그리고 싶을 때 쓴다
        self.last_box: Optional[Tuple[float, float, float, float]] = None
        self.last_conf = 0.
        self.calls = 0
        self.seconds = 0.
        self.reset()

    # -- 상태 -------------------------------------------------------------
    def reset(self) -> None:
        """다음 제품으로 넘어간다. 앞 제품의 타격을 지운다."""
        self.marks: List[Mark] = []     # 세기로 한 프레임들
        self.seen = 0
        self._run: List[Tuple[float, float, float, float, float]] = []
        self._run_ok = False
        self._await_gap = False
        self._quiet_since: Optional[float] = None

    def hold(self) -> None:
        """통과한 뒤 부른다. 망치를 한 번 내려놓기 전까지 세지 않는다.

        안 그러면 한 번 길게 치는 것이 제품 두 개로 세어진다.
        """
        self.reset()
        self._await_gap = True

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if not self.weights.exists():
            self.reason_off = f"가중치가 없습니다: {self.weights}"
            return False
        return True

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights), task="detect")
        return self._model

    # -- 한 프레임 ---------------------------------------------------------
    def look(self, frame) -> Optional[Tuple[float, float, float]]:
        """이 프레임에서 망치를 찾는다. (cx, cy, 신뢰도) 또는 None."""
        self.last_box = None
        self.last_conf = 0.
        if not self.available:
            return None
        started = time.perf_counter()
        try:
            result = self._load().predict(frame, conf=self.conf,
                                          imgsz=IMAGE_SIZE, verbose=False)[0]
        except Exception as exc:
            self.reason_off = f"검출 중 오류: {exc}"
            return None
        self.calls += 1
        self.seconds += time.perf_counter() - started
        best = None
        for box in result.boxes:
            score = float(box.conf[0])
            if best is None or score > best[2]:
                x, y, w, h = (float(v) for v in box.xywhn[0])
                best = (x, y, score, w, h)
        if best is None:
            return None
        self.last_box = (best[0], best[1], best[3], best[4])
        self.last_conf = best[2]
        return best[0], best[1], best[2]

    def feed(self, frame, t: float,
             found: Optional[Tuple[float, float, float]] = None) -> Optional[
                 Tuple[float, float, float]]:
        """프레임 하나를 넣는다. 이미 찾아 둔 결과를 넣어도 된다."""
        if found is None:
            found = self.look(frame)
            area = (self.last_box[2] * self.last_box[3]) if self.last_box else 0.
        else:
            area = (self.last_box[2] * self.last_box[3]) if self.last_box else 0.
        # 앞 제품이 방금 통과했다면 망치를 한 번 내려놓을 때까지 센다.
        if self._await_gap:
            if found is not None:
                self._quiet_since = None
                return found
            if self._quiet_since is None:
                self._quiet_since = t
            elif t - self._quiet_since >= GAP_SECONDS:
                self._await_gap = False
            return None
        if found is not None:
            self._feed(t, found, area)
        return found

    # -- 깜빡임 걸러내기 ---------------------------------------------------
    def _feed(self, t: float, found: Tuple[float, float, float],
              area: float) -> None:
        """구간이 조건을 채울 때까지 모아 두었다가 한꺼번에 넘긴다.

        앞을 못 보므로 이렇게 한다. 조건을 채우기 전에 끊기면 버린다.
        판정이 0.3초 늦어지는 대신 오검출이 사라진다.
        """
        if self._run and t - self._run[-1][0] > RUN_GAP:
            self._run = []
            self._run_ok = False
        self._run.append((t, found[0], found[1], found[2], area))

        if self._run_ok:
            self._tally(t, found, area)
            return
        long_enough = self._run[-1][0] - self._run[0][0] >= RUN_SECONDS
        sure_enough = max(x[3] for x in self._run) >= RUN_CONF
        if long_enough and sure_enough:
            self._run_ok = True
            for when, cx, cy, conf, box in self._run:
                self._tally(when, (cx, cy, conf), box)

    def _run_size(self) -> float:
        """이 구간에서 망치 상자가 보통 얼마나 컸는지.

        기준을 RUN_CONF(.60)로 잡는다. crop015 의 검은 부품은 .55 로
        잡혔으니 이 기준이면 가짜가 기준 크기를 끌어내리지 못한다.
        구간이 열렸다는 것 자체가 .60 프레임이 하나는 있다는 뜻이다.
        """
        sure = sorted(box for *_, conf, box in self._run
                      if conf >= RUN_CONF and box > 0)
        return sure[len(sure) // 2] if sure else 0.

    def _tally(self, t: float, found: Tuple[float, float, float],
               area: float) -> None:
        usual = self._run_size()
        if usual > 0 and 0 < area < usual * SIZE_SHARE:
            return                    # 망치가 아닌 다른 물건이다
        self.seen += 1
        self.marks.append(Mark(t, found[0], found[1]))

    # -- 답 ---------------------------------------------------------------
    @property
    def strikes(self) -> int:
        return len(find_strikes(self.marks))

    @property
    def held_seconds(self) -> float:
        return (self.marks[-1].time - self.marks[0].time) if self.marks else 0.

    def verdict(self) -> Tuple[bool, str]:
        """(통과, 근거 한 줄)."""
        if not self.available:
            return False, f"[cannot judge] {self.reason_off}"
        if not self.marks:
            return False, "[waiting] no hammer in view"
        count = self.strikes
        span = self.held_seconds
        if count >= MIN_STRIKES and span >= MIN_SECONDS:
            return True, (f"[strikes confirmed] {count} of {MIN_STRIKES} "
                          f"strikes, hammer {span:.2f}s over "
                          f"{len(self.marks)} frames")
        if count == 0 and span < MIN_SECONDS:
            return False, "[waiting] hammer-like flicker, filtered as a false hit"
        return False, (f"[waiting] {count} of {MIN_STRIKES} strikes, "
                       f"hammer {span:.2f}s over {len(self.marks)} frames")
