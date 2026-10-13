"""v11 판정 로직 (화면 그리기·모델 실행 제외, 판정만)

체크리스트 (위에서부터 순서대로 체크)
  덮개 열림
  제품 1~3 각각: ① 안쪽면 → ② 힌지 → ③ 옆면 해머 → ④ 옆면 → ⑤ 윗면
  덮개 닫힘
  ※ PPT 2단계 '안쪽면 → 옆면'을 ①과 ④로 나눔. 실제 작업은 들어 올려 안쪽면을 보고,
    힌지를 구부리고, 해머를 친 뒤, 제품을 세워 옆면·윗면을 본다 (crop18 기준).
    하나로 묶으면 옆면이 해머 뒤에야 보여서 체크가 20초씩 멈춰 있었음.

판정 기준
  덮개   : PPT STEP1 (그물 면적이 처음 2초 기준의 50% 미만 1초 → 열림, 50% 이상 1초 → 닫힘)
  안쪽면·옆면 : PPT STEP2 (conf ≥ 0.70, 마스크 면적 ≥ 3%, 각 0.3초 이상, 안쪽면 → 옆면 순서)
  힌지   : [새 기준] 해머가 처음 보이기 직전 HINGE_WINDOW 초 안에, 손목이 힌지를 쥔 채
           위아래로 왕복(PPT STEP3 구부림 동작)한 시간이 합 0.8초 이상
  해머   : PPT STEP4 (망치 높이 봉우리 2개 이상 = 2타)
  윗면   : PPT STEP5 (양손 장갑 0.2초 → 양팔-장갑 연결 0.2초 → TOP 0.2초 / Review TOP 0.4초),
           진행 구조는 seokhee05 step4 v15 와 같음. 해머 뒤에만 인정.
           TOP_RELAX = True 면 양손·양팔 대신 장갑/팔 하나만 있어도 인정 (답변 참고)

제품 넘기기
  - 해머를 친 제품에 해머가 또 통과 → 다음 제품
  - 해머를 친(또는 옆면까지 본) 제품 뒤에 안쪽면이 새로 보임 → 다음 제품을 들어 올린 것
  - 윗면까지 끝나면 다음 제품
  - 넘길 때 근거가 없는 단계는 누락(빨간 X)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
for root in (_HERE, _HERE / "merge/model/code", _HERE / "model/code"):
    for sub in ("1_plus_3", "2_hinge", "4_top"):
        p = root / sub
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

import cover_check as cc  # noqa: E402
import hinge_check as hc  # noqa: E402
from hammer_check import HammerCheck  # noqa: E402

PARTS = 3
ANALYZE_FPS = 15              # 무거운 모델은 초당 15번 (30fps 영상이면 2프레임마다)
hc.ANALYZE_FPS = ANALYZE_FPS

# ---- 안쪽면 (PPT STEP2) ----
CONF_TH = {"inside": 0.70, "side": 0.70, "top": 0.70}
AREA_TH = 0.03
STEP1_MIN = 0.3
MERGE_GAP = 0.5               # 이 간격 안에서 다시 보이면 이어서 셈 (v9 와 같음)
SIDE_GATE = 0.15              # 안쪽면이 이만큼 쌓이기 전 옆면은 세지 않음 (순서 강제, v9 와 같음)
STEP1_STALE = 6.0             # 이 시간 동안 아무 면도 안 보이면 쌓던 값을 버림
CONF_KEEP = 0.70              # 이어서 셀 때 최소 conf. PPT 그대로면 0.70 (0.50 으로 낮추는 안은 답변 참고)

# ---- 힌지 (새 기준) ----
HINGE_WINDOW = 4.0            # 해머가 처음 보인 시각 기준 직전 몇 초를 볼지
HINGE_MIN_BEND = 0.8          # 그 안에서 '쥐고 왕복' 시간 합 (PPT 구부림 합 0.8초)

# ---- 해머 ----
HAMMER_STALE = 5.0            # 1타만 치고 끝난 타격이 다음 제품과 합쳐지지 않게
COOLDOWN = 3.0

# ---- 윗면 (PPT STEP5) ----
STEP4_PAIR_HOLD = 0.20
STEP4_POSE_HOLD = 0.20
STEP4_TOP_DIRECT = 0.20
STEP4_TOP_REVIEW = 0.40
STEP4_RESET_GAP = 1.0         # v15 와 같음 (v9 는 2.5)
GAP_READY = 0.45              # v15 '양손 간격' 표시 기준 (화면 표시용)

# 실제 작업 순서: 들어 올려 안쪽면 → 힌지 구부림 → 해머 → 세워서 옆면 → 윗면
# (PPT 2단계 '안쪽면 → 옆면'을 두 칸으로 나눔. 옆면은 해머 뒤에 제품을 세울 때 보이기 때문)
STEP_KEYS = ["inside", "hinge", "hammer", "side", "top"]
# 화면 체크리스트·합격 판정에 쓰는 단계 (옆면은 화면에서 뺌. 검출은 계속 하고 제품 넘기기 판단과 JSON 기록에만 씀)
CHECK_KEYS = ["inside", "hinge", "hammer", "top"]
STEP_LABELS = {"inside": "안쪽면 확인", "hinge": "힌지 작동 확인", "hammer": "옆면 해머 확인",
               "side": "옆면 확인", "top": "윗면 육안 확인"}
NEXT_AFTER_SIDE = 4.0         # 옆면까지 본 제품 뒤에 이만큼 지나 안쪽면이 새로 보이면 다음 제품 (해머를 빼먹은 제품용)
TOP_RELAX = True              # True: 윗면 = TOP + 장갑/팔 아무거나 하나 (PPT 는 양손 장갑 + 양팔 연결 둘 다)


def passes(view, conf, area):
    return view in CONF_TH and conf >= CONF_TH[view] and area >= AREA_TH


# =====================================================================
# MediaPipe 결과 변환
# =====================================================================
def mp_result(lm: Optional[np.ndarray]):
    """(33,3) 배열 → cover_check.arms_from_pose 가 받는 Tasks API 결과 모양"""
    if lm is None:
        return None
    pts = [SimpleNamespace(x=float(x), y=float(y), visibility=float(v)) for x, y, v in lm]
    return SimpleNamespace(pose_landmarks=[pts])


def pose_arm(lm, w, h, arm):
    """v15 pose_arm 과 같은 기준"""
    if lm is None:
        return None
    out = []
    for i in ((11, 13, 15) if arm == "L" else (12, 14, 16)):
        x, y, v = lm[i]
        th = 0.35 if i in (13, 14) else 0.45
        inside = -0.08 <= x <= 1.08 and -0.08 <= y <= 1.08
        out.append(((float(x * w), float(y * h)) if v >= th and inside else None, float(v)))
    return out


def _arm_glove_score(arm_pose, glove, diag):
    """v15 _arm_glove_score 그대로"""
    if arm_pose is None or arm_pose[1][0] is None:
        return None
    shoulder, elbow, wrist = arm_pose[0][0], arm_pose[1][0], arm_pose[2][0]
    g = glove["center"]
    d = math.dist(elbow, g) / diag
    if not (0.025 <= d <= 0.38):
        return None
    score = d
    if wrist is not None:
        score -= max(0.0, 0.055 - math.dist(wrist, g) / diag) * 0.65
    if shoulder is not None:
        a = np.array(elbow, float) - np.array(shoulder, float)
        b = np.array(g, float) - np.array(elbow, float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-6 and nb > 1e-6:
            cos = float(np.dot(a, b) / (na * nb))
            score += 0.10 if cos < -0.55 else (0.04 if cos < -0.10 else 0.0)
    return score - min(0.03, glove["conf"] * 0.015)


def fuse_pose_to_gloves(lm, gloves, pair, w, h):
    """v15 fuse_pose_to_gloves 그대로. 반환 {'L': {...}, 'R': {...}, 'arms': (lp, rp)}"""
    lp, rp = pose_arm(lm, w, h, "L"), pose_arm(lm, w, h, "R")
    diag = math.hypot(w, h)
    cands = list(pair) if pair is not None else sorted(gloves, key=lambda d: d["conf"], reverse=True)[:4]
    out = {a: {"glove": None, "proxy": None} for a in ("L", "R")}
    opts = []
    for arm, pose in (("L", lp), ("R", rp)):
        for gi, g in enumerate(cands):
            s = _arm_glove_score(pose, g, diag)
            if s is not None:
                opts.append((s, arm, gi))
    used_a, used_g = set(), set()
    for _, arm, gi in sorted(opts):
        if arm in used_a or gi in used_g:
            continue
        out[arm]["glove"] = cands[gi]
        out[arm]["proxy"] = cands[gi]["center"]
        used_a.add(arm)
        used_g.add(gi)
    out["arms"] = (lp, rp)
    return out


# =====================================================================
# 단계별 검출기 (각각 '근거가 생긴 시각'을 돌려줌)
# =====================================================================
class ViewDetector:
    """안쪽면(inside)·옆면(side) 각각 PPT 기준(conf ≥ 0.70, 면적 ≥ 3%, 0.3초)을 채우면 근거를 냄.
    inside → side 순서는 제품에 배정할 때 강제한다 (안쪽면 근거가 있는 제품에만 옆면 인정)."""
    STALE = 3.0

    def __init__(self):
        self.acc = {"inside": 0.0, "side": 0.0}
        self.last = {"inside": None, "side": None}
        self.last_pass = {"inside": -1e9, "side": -1e9}

    def phase(self, view):
        a = self.acc[view]
        return f"{'안쪽면' if view == 'inside' else '옆면'} {a:.1f}/{STEP1_MIN}초" if a > 0 else ""

    def feed(self, t, view, conf, area):
        """반환: 'inside' / 'side' / None"""
        for v in ("inside", "side"):
            if self.last[v] is not None and t - self.last[v] > self.STALE:
                self.acc[v], self.last[v] = 0.0, None
        if view not in ("inside", "side") or area < AREA_TH or t - self.last_pass[view] < COOLDOWN:
            return None
        lt = self.last[view]
        keep = lt is not None and t - lt <= MERGE_GAP and conf >= CONF_KEEP
        if not (passes(view, conf, area) or keep):
            return None
        if lt is not None and t - lt <= MERGE_GAP:
            self.acc[view] += t - lt
        self.last[view] = t
        if self.acc[view] >= STEP1_MIN:
            self.acc[view], self.last[view] = 0.0, None
            self.last_pass[view] = t
            return view
        return None


class HammerDetector:
    def __init__(self, weights):
        self.check = HammerCheck(weights)
        self.run_start = None     # 마지막 통과한 타격 묶음에서 망치가 처음 보인 시각

    def look(self, frame):
        return self.check.look(frame)

    def feed(self, t, found, box):
        ch = self.check
        ch.last_box = box
        area = (box[2] * box[3]) if box else 0.0
        if ch._await_gap:
            if found is not None:
                ch._quiet_since = None
            elif ch._quiet_since is None:
                ch._quiet_since = t
            elif t - ch._quiet_since >= 1.0:
                ch._await_gap = False
            return False
        if found is not None:
            ch._feed(t, found, area)
        m = ch.marks
        if m and t - m[-1].time > HAMMER_STALE:
            ch.reset()
            return False
        ok, _ = ch.verdict()
        if ok:
            self.run_start = m[0].time if m else t
            ch.hold()
            return True
        return False


class HingeTracker:
    """손목이 힌지를 쥐고 위아래로 왕복하는지 프레임마다 기록 (PPT STEP3 의 구부림 후보 조건).
    판정은 해머가 통과했을 때 직전 HINGE_WINDOW 초를 보고 한다."""

    def __init__(self):
        self.judge = hc.LiveJudge()
        self.static = hc.StaticFilter()
        self.held = None
        self._flag_cache = {}

    def feed(self, t, hinge_boxes, kp):
        st = self.static(hinge_boxes)
        self.held = hc.held_box(hinge_boxes, kp, st)
        self.judge.push(t, self.held, kp, None)

    def bend_seconds(self, t0, t1):
        j = self.judge
        s = 0
        for i, t in enumerate(j.times):
            if t0 <= t < t1 and j.holds[i] is not None:
                f = self._flag_cache.get(i)
                if f is None:
                    f = j._flag_at(i)
                    if i + j.half < len(j.times):     # 앞뒤 창이 다 찼을 때만 저장
                        self._flag_cache[i] = f
                if f:
                    s += 1
        return s / ANALYZE_FPS

    def live_bend(self, now):
        """화면 표시용: 최근 HINGE_WINDOW 초 동안 쥐고 왕복한 시간"""
        return self.bend_seconds(now - HINGE_WINDOW, now + 1e-6)


class TopDetector:
    """PPT STEP5 기준. 진행 구조(ACTIVE 시작, review 인정, 리셋)는 v15 와 같음"""

    def __init__(self):
        self.last_pass = -1e9
        self._reset()

    def _reset(self):
        self.state = "WAITING"
        self.ev_pair = self.ev_pose = self.ev_top = False
        self.pair_n = self.pose_n = self.direct_n = self.review_n = 0
        self.last_relevant = None

    def phase(self):
        if self.state != "ACTIVE":
            return ""
        return "윗면 확인 중" if (self.ev_pair and self.ev_pose) else ("팔 연결 확인 중" if self.ev_pair else "장갑 감지 중")

    def feed(self, t, status, direct, has_pair, two_arm, proxy_count):
        top_cand = status in ("TOP_DIRECT", "TOP_REVIEW")
        hand = has_pair or proxy_count > 0
        review_sup = status == "TOP_REVIEW" and (has_pair or two_arm or proxy_count > 0)
        if top_cand or hand or status == "PLASTIC_RISK":
            self.last_relevant = t
        if t - self.last_pass < COOLDOWN:
            return False
        if self.state == "WAITING" and hand:
            self._reset()
            self.state = "ACTIVE"
            self.last_relevant = t
        if self.state != "ACTIVE":
            return False
        self.pair_n = self.pair_n + 1 if has_pair else 0
        self.pose_n = self.pose_n + 1 if two_arm else 0
        self.direct_n += 1 if direct else 0
        self.review_n += 1 if review_sup else 0
        f = ANALYZE_FPS
        if TOP_RELAX:      # 장갑 하나 또는 팔-장갑 연결 하나만 있어도 손 근거로 인정
            self.pair_n = self.pair_n + 1 if hand else 0
            self.pose_n = self.pair_n
        if not self.ev_pair and self.pair_n / f >= STEP4_PAIR_HOLD:
            self.ev_pair = True
        if self.ev_pair and not self.ev_pose and self.pose_n / f >= STEP4_POSE_HOLD:
            self.ev_pose = True
        if not self.ev_top and (self.direct_n / f >= STEP4_TOP_DIRECT or self.review_n / f >= STEP4_TOP_REVIEW):
            self.ev_top = True
        if self.ev_pair and self.ev_pose and self.ev_top:
            self.last_pass = t
            self._reset()
            return True
        if self.last_relevant is not None and t - self.last_relevant > STEP4_RESET_GAP:
            self._reset()
        return False


# =====================================================================
# 박스(사이클) 단위 + 제품 순서
# =====================================================================
class Product:
    def __init__(self, k, t):
        self.k, self.start = k, t
        self.ev = {s: None for s in STEP_KEYS}   # 단계별 근거 시각
        self.closed = None                       # 넘어간 시각
        self.hinge_bend = None                   # 해머 직전 쥐고 왕복한 시간 (근거 표시용)
        self.side_seen = None                    # 해머 전에 옆면이 비친 시각
        self.top_pending = None                  # 해머 전에 잡힌 윗면 근거 (해머를 빼먹은 제품용)
        self.warned = set()                      # 누락 경고를 이미 띄운 단계

    def missed(self, s):
        """누락 = 근거가 없는데 뒤 단계가 이미 진행됐거나, 제품이 넘어감"""
        if self.ev[s] is not None:
            return False
        if self.closed is not None:
            return True
        later = CHECK_KEYS[CHECK_KEYS.index(s) + 1:]
        return any(self.ev[q] is not None for q in later)

    def status(self, s):
        """'done' / 'miss' / 'current' / 'wait'. 한 단계를 빼먹어도 다음 단계는 계속 체크된다"""
        if self.ev[s] is not None:
            return "done"
        if self.missed(s):
            return "miss"
        first_open = next((q for q in CHECK_KEYS if self.ev[q] is None and not self.missed(q)), None)
        return "current" if (first_open == s and self.closed is None) else "wait"

    def complete(self):
        return all(self.ev[s] is not None for s in CHECK_KEYS)


class CycleJudge:
    WAIT_COVER, INSPECTING, DONE = "덮개 열림 대기", "검사 중", "덮개 닫힘"

    def __init__(self, cover_weights, hammer_weights, W, H):
        self.W, self.H = W, H
        self.cover = cc.CoverCheck(cover_weights)
        self.hammer_w = hammer_weights
        self.boxes_done = 0
        self.last_rows = None
        self.history = []
        self.events = []
        self._new_box()

    def _new_box(self):
        self.state = self.WAIT_COVER
        self.cover.reset()
        self.open_t = self.close_t = None
        self.views = ViewDetector()
        self._ham_first = self._ham_last = None
        self.hinge = HingeTracker()
        self.hammer = HammerDetector(self.hammer_w)
        self.top = TopDetector()
        self.products: List[Product] = []
        self.extra = []            # 제품 3개를 넘는 동작 (경고용)
        self.warnings = []         # (시각, 제품번호, 단계 라벨) 누락 경고 (화면 팝업용)
        self.result = None
        self.gap_norm = None
        self.fused = None

    # -- 제품 순서 --------------------------------------------------------
    @property
    def cur(self) -> Optional[Product]:
        if self.products and self.products[-1].closed is None:
            return self.products[-1]
        return None

    def _open_next(self, t):
        if len(self.products) >= PARTS:
            return None
        p = Product(len(self.products), t)
        self.products.append(p)
        return p

    def _close(self, p, t, why):
        if p is None or p.closed is not None:
            return
        if p.ev["hammer"] is None:
            # 해머를 빼먹은 제품: 해머 기준으로 볼 수 없으니 제품 구간 전체에서 판정
            if p.ev["hinge"] is None:
                bend = self.hinge.bend_seconds(p.start, t)
                if bend >= HINGE_MIN_BEND:
                    p.ev["hinge"], p.hinge_bend = p.start, bend
                    self.events.append((t, f"제품 {p.k + 1} 힌지 작동 확인 (해머 없음, 제품 구간 구부림 {bend:.1f}초)"))
            if p.ev["top"] is None and p.top_pending is not None:
                p.ev["top"] = p.top_pending
                self.events.append((t, f"제품 {p.k + 1} 윗면 육안 확인 (해머 없음)"))
        p.closed = t
        miss = [STEP_LABELS[s] for s in CHECK_KEYS if p.ev[s] is None]
        self.events.append((t, f"제품 {p.k + 1} 끝 ({why})" + (f" · 누락: {', '.join(miss)}" if miss else "")))

    def _assign(self, t, step):
        p = self.cur
        if step == "inside":
            if p is not None and p.ev["inside"] is not None:
                st_ = p.ev["side"] if p.ev["side"] is not None else p.side_seen
                side_done = st_ is not None and t - st_ >= NEXT_AFTER_SIDE
                if p.ev["hammer"] is None and not side_done:
                    return                               # 같은 제품을 다시 본 것
                self._close(p, t, "다음 제품 들어 올림")
                p = None
            if p is None:
                p = self._open_next(t)
                if p is None:
                    self.extra.append((t, step))
                    self.events.append((t, "제품 3개 이후 안쪽면 감지 (반영 안 함)"))
                    return
        else:
            if p is None or p.ev[step] is not None:
                return
            if step == "side" and (p.ev["inside"] is None or p.ev["hammer"] is None):
                if p.ev["inside"] is not None and p.side_seen is None:
                    p.side_seen = t                      # 제품을 세워 본 흔적 (다음 제품 판단용)
                return                                   # 안쪽면 → (힌지·해머) → 옆면 순서. 해머 전에 비친 옆면은 무시
            if step == "top" and p.ev["hammer"] is None:
                if p.top_pending is None:
                    p.top_pending = t                    # 해머 전 윗면: 해머를 빼먹은 경우에만 나중에 인정
                return
        p.ev[step] = t
        self.events.append((t, f"제품 {p.k + 1} {STEP_LABELS[step]}"))
        if step == "top":
            self._close(p, t, "윗면까지 확인")

    # -- 프레임마다 ------------------------------------------------------
    def _hinge_check_at(self, hs):
        """망치가 처음 보인 시각 hs 직전 구부림으로 지금 제품의 힌지 판정 (해머 2타를 기다리지 않음)"""
        p = self.cur
        if p is None or p.ev["hinge"] is not None or p.ev["hammer"] is not None:
            return
        bend = self.hinge.bend_seconds(hs - HINGE_WINDOW, hs)
        if bend >= HINGE_MIN_BEND:
            p.ev["hinge"] = hs
            p.hinge_bend = bend
            self.events.append((hs, f"제품 {p.k + 1} 힌지 작동 확인 (해머 직전 {HINGE_WINDOW:.0f}초 중 구부림 {bend:.1f}초)"))
        else:
            p.hinge_checked_no = True

    def feed_hammer(self, t, found, box):
        """망치는 매 프레임. 망치가 나타나면 직전 HINGE_WINDOW 초의 구부림으로 힌지를 판정"""
        if self.state != self.INSPECTING:
            return
        seen = found is not None and found[2] >= 0.6
        if seen:
            if self._ham_last is None or t - self._ham_last > 1.5:
                self._ham_first = t                      # 망치가 새로 나타남
            self._ham_last = t
            if t - self._ham_first >= 0.2:               # 0.2초 이상 보이면 진짜 망치로 보고 힌지 판정
                self._hinge_check_at(self._ham_first)
        if not self.hammer.feed(t, found, box):
            return
        hs = self.hammer.run_start
        bend = self.hinge.bend_seconds(hs - HINGE_WINDOW, hs)
        p = self.cur
        if p is not None and p.ev["hammer"] is not None:
            self._close(p, t, "다음 제품 해머")
        p = self.cur or self._open_next(t)
        if p is None:
            self.extra.append((t, "hammer"))
            self.events.append((t, "제품 3개 이후 해머 감지 (반영 안 함)"))
            return
        p.hinge_bend = bend
        if bend >= HINGE_MIN_BEND and p.ev["hinge"] is None:
            p.ev["hinge"] = hs
            self.events.append((hs, f"제품 {p.k + 1} 힌지 작동 확인 (해머 직전 {HINGE_WINDOW:.0f}초 중 구부림 {bend:.1f}초)"))
        elif bend < HINGE_MIN_BEND:
            self.events.append((hs, f"제품 {p.k + 1} 힌지 근거 없음 (해머 직전 구부림 {bend:.1f}초 < {HINGE_MIN_BEND}초)"))
        p.ev["hammer"] = t
        self.events.append((t, f"제품 {p.k + 1} 옆면 해머 확인"))
        self._update_warnings(t)

    def feed(self, t, s):
        """s: cover mp seg hinge kp s4_status s4_direct pair gloves"""
        arms = cc.arms_from_pose(mp_result(s["mp"])) if s["mp"] is not None else None
        self.cover.feed(None, t, arms, area=s["cover"])
        self.fused = fuse_pose_to_gloves(s["mp"], s["gloves"], s["pair"], self.W, self.H)
        self.gap_norm = s.get("gap")
        if self.state == self.WAIT_COVER:
            ok, _ = self.cover.verdict_open()
            if ok:
                self.state = self.INSPECTING
                self.open_t = t
                self.events.append((t, "덮개 열림"))
            return
        if self.state != self.INSPECTING:
            return
        v, c, a = s["seg"]
        if self.cover.base > 0 and s["cover"] >= cc.SHARE * self.cover.base:
            v = "none"        # 덮개(그물)가 다시 덮이는 중: 세그가 그물을 안쪽면으로 잘못 잡아서 무시
        got = self.views.feed(t, v, c, a)
        if got:
            self._assign(t, got)
        self.hinge.feed(t, s["hinge"], s["kp"])
        fz = self.fused
        two = fz["L"]["proxy"] is not None and fz["R"]["proxy"] is not None
        proxies = int(fz["L"]["proxy"] is not None) + int(fz["R"]["proxy"] is not None)
        if self.top.feed(t, s["s4_status"], s["s4_direct"], s["pair"] is not None, two, proxies):
            self._assign(t, "top")
        self._update_warnings(t)
        ok, _ = self.cover.verdict_close()
        if ok:
            self.close_t = t
            self._close(self.cur, t, "덮개 닫힘")
            self._update_warnings(t)
            self.state = self.DONE
            self.events.append((t, "덮개 닫힘"))
            self._finish(t)

    def _finish(self, t):
        for k in range(len(self.products), PARTS):          # 아예 손대지 않은 제품
            self.warnings.append((t, k + 1, "제품 검사 안 함"))
            self.events.append((t, f"⚠ 제품 {k + 1} 검사 안 함"))
        missing = {}
        for k in range(PARTS):
            p = self.products[k] if k < len(self.products) else None
            miss = [STEP_LABELS[s] for s in CHECK_KEYS if p is None or p.ev[s] is None]
            if miss:
                missing[k + 1] = miss if p is not None else ["제품 검사 안 함"]
        self.result = {"box": self.boxes_done + 1, "open": self.open_t, "close": t,
                       "products": [{s: p.ev[s] for s in STEP_KEYS} | {"hinge_bend_s": p.hinge_bend}
                                    for p in self.products],
                       "missing": missing, "pass": not missing,
                       "extra_after_3": [(round(a, 1), STEP_LABELS[b]) for a, b in self.extra]}
        self.history.append(self.result)
        self.boxes_done += 1
        self.last_rows = self.checklist()        # 다음 박스 덮개가 열릴 때까지 화면에 유지

    def _update_warnings(self, t):
        for p in self.products:
            for st_ in CHECK_KEYS:
                if st_ not in p.warned and p.missed(st_):
                    p.warned.add(st_)
                    self.warnings.append((t, p.k + 1, STEP_LABELS[st_]))
                    self.events.append((t, f"⚠ 제품 {p.k + 1} {STEP_LABELS[st_]} 누락"))

    def active_warning(self, t, hold=3.0):
        """화면에 띄울 누락 경고 (최근 hold 초 안의 것들)"""
        return [w for w in self.warnings if t - w[0] <= hold]

    def maybe_restart(self, t):
        """덮개가 닫히고 2초 뒤 다음 박스 준비 (화면은 다음 박스 덮개가 열릴 때까지 이전 결과 유지)"""
        if self.state == self.DONE and t - self.close_t > 2.0:
            self._new_box()

    def showing_previous(self):
        return self.state == self.WAIT_COVER and self.last_rows is not None

    def early_evidence(self):
        """지금 제품에서 앞 단계보다 먼저 근거가 잡힌 단계 (화면 안내용)"""
        p = self.cur
        if p is None:
            return []
        out, gap = [], False
        for s in CHECK_KEYS:
            if p.ev[s] is None:
                gap = True
            elif gap:
                out.append(STEP_LABELS[s].split(" ")[0])
        return out

    # -- 화면용 ---------------------------------------------------------
    def checklist(self):
        """[(라벨, 상태, 시각)] 위에서부터 순서대로. 상태: done/miss/current/wait"""
        rows = [("덮개 열림", "done" if self.open_t is not None else
                 ("current" if self.state == self.WAIT_COVER else "wait"), self.open_t)]
        for k in range(PARTS):
            p = self.products[k] if k < len(self.products) else None
            for s in CHECK_KEYS:
                label = f"제품 {k + 1} · {STEP_LABELS[s]}"
                if p is None:
                    first_open = (self.state == self.INSPECTING and self.cur is None
                                  and len(self.products) == k and s == "inside")
                    st = "miss" if self.state == self.DONE else ("current" if first_open else "wait")
                    rows.append((label, st, None))
                else:
                    st = p.status(s)
                    # 화면 시각 = 순서대로 체크된 시각 (앞 단계보다 근거가 먼저 생겼어도 앞 단계가 체크될 때 같이 체크)
                    prev = [p.ev[q] for q in CHECK_KEYS[:CHECK_KEYS.index(s) + 1] if p.ev[q] is not None]
                    rows.append((label, st, max(prev) if (st == "done" and prev) else None))
        rows.append(("덮개 닫힘", "done" if self.close_t is not None else "wait", self.close_t))
        # 화면에는 'current' 는 하나만 (가장 위)
        seen = False
        out = []
        for lab, st, tt in rows:
            if st == "current":
                if seen:
                    st = "wait"
                seen = True
            out.append((lab, st, tt))
        return out

    def parts_done(self):
        return sum(p.complete() for p in self.products)
