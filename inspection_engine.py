"""직원 기준영상(check.mp4)의 반복 작업 순서를 따르는 검사 엔진."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

Point = Tuple[float, float]

@dataclass(frozen=True)
class StepDefinition:
    id: str; label: str; detector: str; required_hits: int = 1; minimum_seconds: float = .8

OFFICIAL_STEPS: List[StepDefinition] = [StepDefinition("box_open", "준비. 박스 덮개·보호망 열기", "large_move", 1, 1.5)]
for n in range(1, 4):
    OFFICIAL_STEPS += [
        StepDefinition(f"item{n}_visual", f"제품 {n}. 들어 올려 외관·흔들림 확인", "lift_visual", 1, 1.5),
        StepDefinition(f"item{n}_driver", f"제품 {n}. Driver Screw 2Point 확인", "driver", 2, 1.),
        StepDefinition(f"item{n}_operate", f"제품 {n}. 뒤집어 Knob·Hinge 작동 확인", "two_hand_operate", 1, 1.5),
        StepDefinition(f"item{n}_under", f"제품 {n}. Damper·Rib·이음부·Gap 확인", "edge_check", 1, 1.5),
        StepDefinition(f"item{n}_return", f"제품 {n}. 포장 후 박스에 반환", "large_move", 1, 1.2)]
OFFICIAL_STEPS.append(StepDefinition("box_close", "마무리. 보호망·덮개 닫기", "large_move", 1, 1.5))

@dataclass
class MotionSample:
    t: float; hands: Tuple[Point, ...]; driver_ratio: float; driver_center: Optional[Point]
@dataclass
class StepResult:
    id: str; label: str; status: str="WAIT"; hits: int=0; required_hits: int=1
    completed_at_seconds: Optional[float]=None; evidence: str=""; confidence: float=0.
@dataclass
class InspectionResult:
    barcode: str; started_at: str; ended_at: str; duration_seconds: float; final_status: str
    checklist: List[Dict]; warnings: List[Dict]; video: str
    reference_workflow: str="직원 기준영상 기반: 박스 열기 + 제품 3개 반복 검사 + 박스 닫기"
    prototype_notice: str="검사 행동과 순서를 확인하는 프로토타입이며 실제 부품 양불량은 REVIEW 대상입니다."

class ActionRecognizer:
    def __init__(self): self.samples: Deque[MotionSample]=deque(maxlen=180)
    def reset(self): self.samples.clear()
    def clear_window(self): self.samples.clear()
    @staticmethod
    def _driver(frame):
        small=cv2.resize(frame,(320,180)); hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([5,105,95]),np.array([28,255,255])); mask[:48]=0
        count=cv2.countNonZero(mask); ratio=count/float(mask.size)
        if count<10: return ratio,None
        m=cv2.moments(mask,True)
        return (ratio,None) if not m["m00"] else (ratio,(m["m10"]/m["m00"]/320.,m["m01"]/m["m00"]/180.))
    def update(self,frame,hands:Sequence[Point],t):
        ratio,center=self._driver(frame); self.samples.append(MotionSample(t,tuple(sorted(hands,key=lambda p:p[0])[:2]),ratio,center))
        while self.samples and t-self.samples[0].t>3.2: self.samples.popleft()
    def detect(self,detector):
        s=list(self.samples)
        if len(s)<15: return False,0.,"동작을 관찰하는 중"
        coverage=sum(bool(x.hands) for x in s)/len(s); two=sum(len(x.hands)>=2 for x in s)/len(s)
        pts=np.asarray([x.hands[0] for x in s if x.hands],np.float32)
        travel=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()) if len(pts)>1 else 0.
        span=float(np.ptp(pts[:,0])+np.ptp(pts[:,1])) if len(pts)>1 else 0.
        pair=np.asarray([np.hypot(x.hands[0][0]-x.hands[1][0],x.hands[0][1]-x.hands[1][1]) for x in s if len(x.hands)>=2])
        pair_span=float(np.ptp(pair)) if len(pair)>2 else 0.
        dpts=np.asarray([x.driver_center for x in s if x.driver_center],np.float32)
        dtravel=float(np.linalg.norm(np.diff(dpts,axis=0),axis=1).sum()) if len(dpts)>1 else 0.; dpeak=max((x.driver_ratio for x in s),default=0.)
        if detector=="driver": raw=coverage>=.25 and dpeak>=.0012 and dtravel>=.035; score=min(1.,dpeak/.003)*min(1.,dtravel/.10); reason=f"Driver 이동 {dtravel:.2f}, 색상 {dpeak*100:.2f}%"
        elif detector=="two_hand_operate": raw=two>=.28 and pair_span>=.09 and travel>=.34; score=min(1.,pair_span/.16)*min(1.,travel/.65); reason=f"양손 작동 변화 {pair_span:.2f}, 이동 {travel:.2f}"
        elif detector=="lift_visual": raw=coverage>=.40 and travel>=.52 and span>=.20; score=min(1.,travel/.90)*min(1.,span/.35); reason=f"들기·회전 이동 {travel:.2f}, 범위 {span:.2f}"
        elif detector=="edge_check": raw=coverage>=.45 and travel>=.62 and span>=.24; score=min(1.,travel/.95)*min(1.,span/.38); reason=f"하부·가장자리 탐색 {travel:.2f}, 범위 {span:.2f}"
        else: raw=coverage>=.35 and travel>=.48 and span>=.18; score=min(1.,travel/.80)*min(1.,span/.32); reason=f"제품·덮개 이동 {travel:.2f}, 범위 {span:.2f}"
        return raw,score,reason

class InspectionEngine:
    def __init__(self,nominal_seconds=70.,warning_seconds=18.):
        self.nominal_seconds=nominal_seconds; self.warning_seconds=warning_seconds; self.recognizer=ActionRecognizer(); self.active=False; self.barcode=""; self.steps=[]; self.warnings=[]; self.step_index=0; self.started_monotonic=0.; self.step_started_time=0.; self.started_iso=""; self.last_warning_time=-999.
    def start(self,barcode,now):
        self.active=True; self.barcode=barcode; self.started_monotonic=now; self.step_started_time=now; self.started_iso=datetime.now().isoformat(timespec="seconds"); self.step_index=0; self.warnings=[]; self.last_warning_time=-999.; self.steps=[StepResult(x.id,x.label,required_hits=x.required_hits) for x in OFFICIAL_STEPS]; self.recognizer.reset()
    @property
    def current_definition(self): return OFFICIAL_STEPS[self.step_index] if self.step_index<len(OFFICIAL_STEPS) else None
    @property
    def current_result(self): return self.steps[self.step_index] if self.step_index<len(self.steps) else None
    def update(self,frame,hands,now):
        if not self.active: return {"event":"idle","message":"바코드를 스캔해 주세요.","reason":""}
        if self.step_index>=len(self.steps): return {"event":"complete","message":"직원 기준 작업 순서가 모두 확인되었습니다.","reason":""}
        self.recognizer.update(frame,hands,now); d=self.current_definition; r=self.current_result; elapsed=now-self.step_started_time
        if elapsed<d.minimum_seconds: return {"event":"progress","message":d.label,"reason":f"현재 단계 관찰 {elapsed:.1f}초"}
        detected,score,reason=self.recognizer.detect(d.detector)
        if detected:
            r.hits+=1; r.evidence=reason; r.confidence=round(score,3); self.recognizer.clear_window(); self.step_started_time=now
            if r.hits>=r.required_hits:
                r.status="PASS"; r.completed_at_seconds=round(now-self.started_monotonic,2); self.step_index+=1
                if self.step_index>=len(self.steps): return {"event":"complete","message":"모든 제품의 기준 작업이 확인되었습니다.","reason":reason}
                return {"event":"step_complete","message":"완료: "+r.label,"reason":reason,"next":self.current_definition.label}
            return {"event":"hit","message":f"{r.label}: {r.hits}/{r.required_hits} Point","reason":reason}
        if elapsed>=self.warning_seconds and now-self.last_warning_time>=8:
            self.last_warning_time=now; warning={"time_seconds":round(now-self.started_monotonic,2),"type":"POSSIBLE_MISSING","message":"현재 작업이 확인되지 않습니다.","expected":d.label}; self.warnings.append(warning); return {"event":"warning","message":warning["message"],"reason":reason}
        return {"event":"progress","message":"확인 중: "+d.label,"reason":reason,"score":round(score,3)}
    def finish(self,video_path,now,forced=False):
        for x in self.steps:
            if x.status!="PASS": x.status="MISSING"
        passed=all(x.status=="PASS" for x in self.steps); out=InspectionResult(self.barcode,self.started_iso,datetime.now().isoformat(timespec="seconds"),round(now-self.started_monotonic,2),"PASS" if passed and not forced else "REVIEW",[asdict(x) for x in self.steps],list(self.warnings),video_path); self.active=False; return out
