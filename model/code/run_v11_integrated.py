"""
========================================================================
Armrest 검사원 행동 추적 - 통합 실행기 v11
========================================================================
v10 → v11
  1. 체크리스트를 세로 순차 목록으로 (덮개 열림 → 제품1~3 × 4단계 → 덮개 닫힘)
     앞 단계가 체크돼야 다음 단계가 체크됨. 제품을 넘길 때 빠진 단계는 빨간 X
  2. 힌지 새 기준: 해머를 치기 직전 4초 안에 힌지를 쥐고 위아래로 구부린 시간 합 0.8초 이상
  3. 윗면: seokhee05 step4 v15 구조 + PPT 기준(장갑 0.2초/팔-장갑 0.2초/TOP 0.2초·Review 0.4초)
  4. MediaPipe: v15 와 같은 Pose(mp.solutions, model_complexity=1) 사용
     - mediapipe 0.10.21 이면 v15 와 똑같이 동작 (v15 requirements 와 같은 버전)
     - 0.10.3x 처럼 mp.solutions 가 없으면 Tasks API(pose_landmarker_lite)로 자동 대체
     - 화면: v15 처럼 어깨-팔꿈치-장갑 연결선, 양손 장갑 간격(gap) 표시
  판정 기준 숫자는 v11_logic.py 맨 위에 모아 둠

사용법
  python model\\code\\run_v11_integrated.py "C:\\영상\\crop003.mov"
결과
  model\\result_<영상이름>_v11.mp4 / .json
========================================================================
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

HERE = Path(__file__).resolve().parent                 # model/code
BASE = HERE.parent.parent                              # kkokkodack
sys.path.insert(0, str(HERE))
import v11_logic as L  # noqa: E402
import hinge_check as hc  # noqa: E402
from step4_glove_top_v3 import find_pair  # noqa: E402
from ultralytics import YOLO  # noqa: E402

try:   # v15 와 같은 segment_status (파일 이름이 버전마다 달라서 순서대로 찾음)
    from step4_glove_pose_dashboard_v15_final import segment_status  # noqa: E402
except ImportError:
    from step4_glove_pose_dashboard_v8_hybrid_top import segment_status  # noqa: E402

import mediapipe as mp  # noqa: E402

D13, DH, D4 = HERE / "1_plus_3", HERE / "2_hinge", HERE / "4_top"
SEG_W = D13 / "armrest_3class_best.pt"
COVER_W = D13 / "cover.pt"
HAMMER_W = D13 / "hammer.pt"
POSE_TASK = D13 / "pose_landmarker_lite.task"
YOLO_POSE_W = D13 / "yolov8n-pose.pt"
HINGE_W = DH / "hinge_det_runs/round2/weights/best.pt"
GLOVE_W = D4 / "glove_runs/glove_detector_v2/weights/best.pt"
FONT_R = BASE / "fonts/NotoSansCJK-Regular.ttc"
FONT_B = BASE / "fonts/NotoSansCJK-Bold.ttc"

PANEL_W = 470

print("[INFO] 모델 불러오는 중...")
seg_model = YOLO(str(SEG_W))
SEG_NAMES = seg_model.names
TOP_ID = next(k for k, v in SEG_NAMES.items() if v == "top")
SIDE_ID = next(k for k, v in SEG_NAMES.items() if v == "side")
hinge_model = YOLO(str(HINGE_W))
yolo_pose = YOLO(str(YOLO_POSE_W))
glove_model = YOLO(str(GLOVE_W))


class Pose:
    """MediaPipe Pose. v15 와 같은 mp.solutions 를 우선, 없으면 Tasks API"""

    def __init__(self):
        self.kind = None
        if hasattr(mp, "solutions"):
            self.det = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                              smooth_landmarks=True, enable_segmentation=False,
                                              min_detection_confidence=0.45, min_tracking_confidence=0.45)
            self.kind = "solutions"
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(POSE_TASK),
                                                   delegate=mp_python.BaseOptions.Delegate.CPU),
                running_mode=mp_vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.45, min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45)
            self.det = mp_vision.PoseLandmarker.create_from_options(opts)
            self.kind = "tasks"
            print(f"[WARN] mediapipe {mp.__version__} 에는 mp.solutions 가 없어 Tasks API(lite)로 대체합니다. "
                  "v15 와 똑같이 하려면: pip install mediapipe==0.10.21")
        print(f"[INFO] MediaPipe Pose: {self.kind} (mediapipe {mp.__version__})")

    def __call__(self, frame, t):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.kind == "solutions":
            rgb.flags.writeable = False
            r = self.det.process(rgb)
            if not r.pose_landmarks:
                return None
            return np.array([[p.x, p.y, p.visibility] for p in r.pose_landmarks.landmark])
        r = self.det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        return np.array([[p.x, p.y, p.visibility] for p in r.pose_landmarks[0]]) if r.pose_landmarks else None

    def close(self):
        self.det.close()


# ---------------------------------------------------------------------
# 세그 (v9 detect_seg 와 같은 기준)
# ---------------------------------------------------------------------
def _largest_component(mask_bin):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask_bin
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i for i, a in enumerate(areas, start=1) if a >= areas.max() * 0.30]
    return np.isin(labels, keep).astype(np.uint8)


def _is_skin(frame, mask_bin, th=0.40):
    H = frame.shape[0]

    def ratio(ys, xs):
        if len(xs) < 30:
            return 0.0
        hsv = cv2.cvtColor(frame[ys, xs].reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        return float((((h <= 25) | (h >= 160)) & (s >= 20) & (s <= 200) & (v >= 60)).mean())

    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 50:
        return False
    top = mask_bin.copy()
    top[int(H * 0.30):, :] = 0
    yt, xt = np.where(top > 0)
    return ratio(ys, xs) >= th or ratio(yt, xt) >= th - 0.10


def seg_view(frame, result):
    H, W = frame.shape[:2]
    if result.boxes is None or len(result.boxes) == 0:
        return "none", 0.0, 0.0, None
    confs = result.boxes.conf.cpu().numpy()
    i = int(confs.argmax())
    if confs[i] < 0.35:
        return "none", 0.0, 0.0, None
    view, conf, area, mask = SEG_NAMES[int(result.boxes.cls[i])], float(confs[i]), 0.0, None
    if result.masks is not None and len(result.masks) > i:
        m = cv2.resize(result.masks.data[i].cpu().numpy(), (W, H), interpolation=cv2.INTER_NEAREST)
        mask = _largest_component((m > 0.5).astype(np.uint8))
        area = float(mask.sum()) / (W * H)
        if view in ("inside", "side") and _is_skin(frame, mask):
            return "none", 0.0, 0.0, None
    return view, conf, area, mask


def yolo_wrists(frame):
    H, W = frame.shape[:2]
    p = yolo_pose.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
    if p.keypoints is None or not len(p.keypoints.xy):
        return None
    i = int((p.boxes.xywh[:, 2] * p.boxes.xywh[:, 3]).argmax())
    return np.concatenate([p.keypoints.xy[i].cpu().numpy() / [W, H],
                           p.keypoints.conf[i].cpu().numpy()[:, None]], 1)


# ---------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------
_fonts = {}


def font(sz, bold=False):
    k = (sz, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(str(FONT_B if bold else FONT_R), sz)
    return _fonts[k]


SEG_COLORS = {"inside": (0, 220, 80), "side": (255, 180, 0), "top": (80, 180, 255)}   # BGR
BLUE, YELLOW, WHITE = (238, 145, 35), (0, 220, 255), (245, 245, 245)                  # v15 색


def draw_arms_v15(img, fused, pair, gap, lm=None):
    """v15 draw_arm 과 같은 모양: 어깨-팔꿈치(파랑), 팔꿈치-장갑, 장갑 상자, 양손 간격선.
    덮개 판정과 같은 기준(cover_check.arm_trusted)을 못 넘는 팔은 그리지 않음 (엉뚱한 선 방지)"""
    if fused is None:
        return
    lp, rp = fused["arms"]
    res = L.mp_result(lm)
    trusted = {"L": False, "R": False}
    if res is not None:
        trusted = {"L": L.cc.arm_trusted(res.pose_landmarks[0], "left"),
                   "R": L.cc.arm_trusted(res.pose_landmarks[0], "right")}
    diag = float(np.hypot(*img.shape[:2]))
    for arm, pose in (("L", lp), ("R", rp)):
        if pose is None or not trusted[arm]:
            continue
        sh, el, wr = (v[0] for v in pose)
        if sh is not None and el is not None:
            cv2.line(img, tuple(map(int, sh)), tuple(map(int, el)), BLUE, 3, cv2.LINE_AA)
        for pt, r in ((sh, 5), (el, 6)):
            if pt is not None:
                cv2.circle(img, tuple(map(int, pt)), r, YELLOW, -1, cv2.LINE_AA)
        if wr is not None:
            cv2.circle(img, tuple(map(int, wr)), 5, WHITE, 1, cv2.LINE_AA)
        g = fused[arm]["glove"]
        if g is None:
            continue
        x1, y1, x2, y2 = map(int, g["bbox"])
        gc = tuple(map(int, g["center"]))
        cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 2, cv2.LINE_AA)
        if el is not None and np.hypot(el[0] - gc[0], el[1] - gc[1]) <= 0.25 * diag:
            cv2.line(img, tuple(map(int, el)), gc, BLUE, 3, cv2.LINE_AA)
        cv2.circle(img, gc, 6, YELLOW, -1, cv2.LINE_AA)
    if pair is not None and gap is not None:
        p1, p2 = tuple(map(int, pair[0]["center"])), tuple(map(int, pair[1]["center"]))
        cv2.line(img, p1, p2, YELLOW, 2, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2 - 35, (p1[1] + p2[1]) // 2 - 10)
        cv2.putText(img, f"gap {gap:.2f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)


def box_px(b, W, H):
    return int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)


# 체크리스트 색 (RGB, 사진처럼 밝은 카드)
CARD, INK, MUTE = (246, 247, 249), (60, 64, 72), (150, 155, 165)
OK_G, MISS_R, CUR_B = (96, 186, 98), (225, 80, 80), (66, 133, 244)


def icon(d, cx, cy, r, st):
    if st == "done":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=OK_G)
        d.line([cx - r * .45, cy + r * .02, cx - r * .1, cy + r * .38, cx + r * .48, cy - r * .32],
               fill=(255, 255, 255), width=max(3, r // 4))
    elif st == "miss":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=MISS_R)
        k = r * .38
        d.line([cx - k, cy - k, cx + k, cy + k], fill=(255, 255, 255), width=max(3, r // 4))
        d.line([cx - k, cy + k, cx + k, cy - k], fill=(255, 255, 255), width=max(3, r // 4))
    elif st == "current":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=CUR_B, width=3)
        d.ellipse([cx - r * .35, cy - r * .35, cx + r * .35, cy + r * .35], fill=CUR_B)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(200, 203, 208), width=2)


def live_hint(cj, label):
    """현재 항목 아래 작은 글씨"""
    if cj.state != cj.INSPECTING:
        return ""
    early = cj.early_evidence()
    pre = ("먼저 감지: " + ", ".join(early) + " · ") if early else ""
    return pre + _hint(cj, label)


def _hint(cj, label):
    if "안쪽면" in label:
        return cj.views.phase("inside")
    if label.endswith("옆면 확인"):
        return cj.views.phase("side")
    if "힌지" in label:
        b = cj.hinge.live_bend(cj._now)
        return f"최근 {L.HINGE_WINDOW:.0f}초 구부림 {b:.1f}/{L.HINGE_MIN_BEND}초 (해머 칠 때 판정)" if b > 0 else "해머 직전 구부림 대기"
    if "해머" in label:
        n = cj.hammer.check.strikes
        return f"{n}/2타" if n else ""
    if "윗면" in label:
        return cj.top.phase()
    return ""


def render(frame, cj, ov):
    H, W = frame.shape[:2]
    img = frame.copy()
    if ov["cover_mask"] is not None and cj.state != cj.INSPECTING:   # 검사 중엔 박스 테두리 그물이 잡혀 거슬려서 숨김
        cm = cv2.resize(ov["cover_mask"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        img[cm > 0] = (img[cm > 0] * 0.6 + np.array([200, 200, 0]) * 0.4).astype(np.uint8)
    v, c, a = ov["seg"]
    if ov["seg_mask"] is not None and L.passes(v, c, a) and cj.state == cj.INSPECTING:
        col = np.array(SEG_COLORS[v])
        m = ov["seg_mask"] > 0
        img[m] = (img[m] * 0.5 + col * 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(ov["seg_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, tuple(int(x) for x in col), 3)
    if ov["held"] is not None and cj.state == cj.INSPECTING and cj.hinge.live_bend(cj._now) > 0 \
            and cj.hinge.bend_seconds(cj._now - 1.0, cj._now + 1e-6) > 0:
        x1, y1, x2, y2 = box_px(ov["held"], W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, "hinge", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    hb = ov["hammer_box"]
    if hb is not None and ov["hammer_conf"] >= 0.6:
        cx, cy, bw, bh = hb
        x1, y1, x2, y2 = int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 3)
        cv2.putText(img, f"hammer {ov['hammer_conf']:.0%}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    if cj.state == cj.INSPECTING or (cj.state == cj.WAIT_COVER and not cj.showing_previous()):
        draw_arms_v15(img, cj.fused, ov["pair"], ov["gap"], ov["mp"])

    canvas = np.zeros((H, W + PANEL_W, 3), np.uint8)
    canvas[:, :W] = img
    warns = cj.active_warning(cj._now)
    canvas[:, W:] = CARD[::-1]
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    X = W + 22

    # 머리
    prev = cj.showing_previous()
    if prev:
        res = cj.history[-1]
        d.text((X, 14), f"박스 #{res['box']} 결과", font=font(21, True), fill=INK)
        d.text((X, 46), "다음 박스 덮개 열림 대기", font=font(16, True), fill=MUTE)
        rows = cj.last_rows
    else:
        st_col = {cj.WAIT_COVER: MUTE, cj.INSPECTING: CUR_B, cj.DONE: OK_G}[cj.state]
        d.text((X, 14), f"박스 #{cj.boxes_done + (0 if cj.state == cj.DONE else 1)} 검사 체크리스트", font=font(21, True), fill=INK)
        d.text((X, 46), f"{cj.state}  ·  완료 제품 {cj.parts_done()}/{L.PARTS}", font=font(16, True), fill=st_col)
        rows = cj.checklist()
    result = cj.history[-1] if (prev or cj.state == cj.DONE) and cj.history else None
    top, bottom = 82, H - (78 if result else 30)
    rh = min(44, (bottom - top) / len(rows))
    r = int(min(13, rh * 0.32))
    for i, (lab, st, tt) in enumerate(rows):
        y = top + i * rh
        cy = int(y + rh / 2)
        if st == "current":
            d.rounded_rectangle([W + 8, y + 2, W + PANEL_W - 10, y + rh - 2], radius=8, fill=(228, 238, 253))
        icon(d, X + r, cy, r, st)
        colr = INK if st in ("done", "current") else (MISS_R if st == "miss" else MUTE)
        hint = live_hint(cj, lab) if (st == "current" and not prev) else ""
        fs = 16 if rh >= 36 else 14
        d.text((X + 2 * r + 12, cy - (fs // 2 + (7 if hint else 0)) - 2), lab,
               font=font(fs, st == "current"), fill=colr)
        if hint:
            d.text((X + 2 * r + 12, cy + 3), hint, font=font(11), fill=CUR_B)
        if tt is not None:
            d.text((W + PANEL_W - 70, cy - 9), f"{tt:.1f}s", font=font(13), fill=MUTE)
        if i < len(rows) - 1 and ("닫힘" in rows[i + 1][0] or lab.endswith("윗면 육안 확인") or i == 0):
            d.line([X, y + rh, W + PANEL_W - 14, y + rh], fill=(222, 224, 228), width=1)

    if result:
        ok = result["pass"]
        y0 = H - 70
        d.rounded_rectangle([W + 10, y0, W + PANEL_W - 10, H - 10], radius=10, fill=OK_G if ok else MISS_R)
        miss = result["missing"]
        d.text((X, y0 + 6), "PASS · 모든 단계 확인" if ok else f"FAIL · 누락 제품 {', '.join(map(str, miss))}",
               font=font(19, True), fill=(255, 255, 255))
        if not ok:
            short = {"안쪽면 확인": "안쪽면", "힌지 작동 확인": "힌지", "옆면 해머 확인": "해머",
                     "윗면 육안 확인": "윗면", "제품 검사 안 함": "검사 안 함"}
            line = " / ".join(f"{k}: " + ",".join(short.get(m, m) for m in v) for k, v in miss.items())
            d.text((X, y0 + 34), line[:46], font=font(13), fill=(255, 240, 240))

    # 누락 경고 팝업 (단계를 빼먹으면 3초 동안 영상 가운데에 표시하고, 검사는 다음 단계로 계속 진행)
    if warns:
        cw, head = 460, 92
        lines = [f"제품 {k} · {lab}" + ("" if lab == "제품 검사 안 함" else "을 하지 않았습니다") for _, k, lab in warns[-3:]]
        chh = head + 70 + 30 * len(lines)
        x0, y0 = (W - cw) // 2, max(20, (H - chh) // 2 - 30)
        d.rounded_rectangle([x0 + 6, y0 + 8, x0 + cw + 6, y0 + chh + 8], radius=18, fill=(0, 0, 0))
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + chh], radius=18, fill=(255, 255, 255))
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + head], radius=18, fill=(214, 96, 92))
        d.rectangle([x0, y0 + head - 18, x0 + cw, y0 + head], fill=(214, 96, 92))
        cx, cy0 = x0 + cw // 2, y0 + 14
        d.polygon([(cx, cy0), (cx - 34, cy0 + 62), (cx + 34, cy0 + 62)], outline=(255, 255, 255), width=6)
        d.line([cx, cy0 + 22, cx, cy0 + 42], fill=(255, 255, 255), width=6)
        d.ellipse([cx - 4, cy0 + 49, cx + 4, cy0 + 57], fill=(255, 255, 255))
        tw = d.textlength("검사 누락!", font=font(30, True))
        d.text((cx - tw / 2, y0 + head + 14), "검사 누락!", font=font(30, True), fill=(52, 60, 84))
        for i, ln in enumerate(lines):
            lw = d.textlength(ln, font=font(17))
            d.text((cx - lw / 2, y0 + head + 60 + 30 * i), ln, font=font(17), fill=(100, 106, 125))

    # 영상 아래 정보
    d.rectangle([0, H - 30, W, H], fill=(20, 20, 20))
    gap_txt = "-" if ov["gap"] is None else f"{ov['gap']:.2f}"
    d.text((12, H - 26), f"t={ov['t']:.1f}s   덮개 {ov['cover_area'] * 100:.1f}%/기준 {cj.cover.base * 100:.1f}%   "
                         f"[seg] {v} {c:.0%}{' ✓' if L.passes(v, c, a) else ''}   "
                         f"[MediaPipe {ov['pose_kind']}] {'O' if ov['mp'] is not None else 'X'}   "
                         f"[장갑 간격] {gap_txt}",
           font=font(14), fill=(200, 220, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------
def run(src, out):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PANEL_W, H))
    step = max(1, int(round(fps / L.ANALYZE_FPS)))
    cj = L.CycleJudge(str(COVER_W), str(HAMMER_W), W, H)
    cj._now = 0.0
    pose = Pose()
    anchor = seed = None
    glove_prior, last_glove = None, None
    ov = {"cover_mask": None, "seg": ("none", 0.0, 0.0), "seg_mask": None, "held": None, "mp": None,
          "hammer_box": None, "hammer_conf": 0.0, "cover_area": 0.0, "t": 0.0, "pair": None, "gap": None,
          "pose_kind": pose.kind}
    fi = 0
    for _ in tqdm(range(total), desc=Path(src).name, unit="f"):
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        ov["t"] = cj._now = t
        cj.maybe_restart(t)
        found = cj.hammer.look(frame)
        ov["hammer_box"], ov["hammer_conf"] = cj.hammer.check.last_box, cj.hammer.check.last_conf
        cj.feed_hammer(t, found, cj.hammer.check.last_box)

        if fi % step == 0:
            sr = seg_model(frame, conf=0.20, verbose=False)[0]
            v, c, a, mask = seg_view(frame, sr)
            ov["seg"], ov["seg_mask"] = (v, c, a), mask
            s4, direct, _w, _s, anchor, seed = segment_status(frame, sr, TOP_ID, SIDE_ID, anchor, seed, fi, fps, W, H)
            gr = glove_model.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
            el = (fi - last_glove) / fps if last_glove is not None else 999.0
            gloves, pr = find_pair(gr.boxes, W, H, glove_prior if el <= 0.32 else None, el)
            pair, gap = None, None
            if pr is not None:
                pair, gap = [pr[1], pr[2]], pr[3]
                glove_prior, last_glove = (pr[1]["center"], pr[2]["center"]), fi
            elif el > 0.32:
                glove_prior = None
            ov["pair"], ov["gap"] = pair, gap
            area = cj.cover.look(frame)
            ov["cover_area"], ov["cover_mask"] = area, cj.cover.last_mask
            lm = pose(frame, t)
            ov["mp"] = lm
            hr = hinge_model.predict(frame, conf=hc.HINGE_CONF, imgsz=640, verbose=False)[0]
            hinge = hr.boxes.xyxy.cpu().numpy() / [W, H, W, H] if len(hr.boxes) else np.zeros((0, 4))
            cj.feed(t, {"cover": area, "mp": lm, "seg": (v, c, a), "hinge": hinge, "kp": yolo_wrists(frame),
                        "s4_status": s4, "s4_direct": direct is not None, "pair": pair, "gloves": gloves,
                        "gap": gap})
            ov["held"] = cj.hinge.held
        writer.write(render(frame, cj, ov))
        fi += 1
    cap.release()
    writer.release()
    pose.close()

    summary = {"video": str(src), "mediapipe": f"{mp.__version__} ({pose.kind})", "boxes": cj.history,
               "events": [{"t": round(t, 2), "event": e} for t, e in sorted(cj.events, key=lambda x: x[0])]}
    if cj.state == cj.INSPECTING:
        summary["unfinished"] = "영상이 끝날 때까지 덮개 닫힘이 감지되지 않음"
    Path(out).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    print(f"\n완료: 박스 {cj.boxes_done}개")
    for r in cj.history:
        print(f"  박스#{r['box']}: {'PASS' if r['pass'] else 'FAIL'}")
        for pn, miss in r["missing"].items():
            print(f"    제품{pn} 누락: {', '.join(miss)}")
    for t, e in sorted(cj.events, key=lambda x: x[0]):
        print(f"  {t:6.1f}s  {e}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python model\\code\\run_v11_integrated.py "영상경로"')
        sys.exit(1)
    src = Path(sys.argv[1])
    out = BASE / "model" / f"result_{src.stem}_v11.mp4"
    run(src, out)
    print("저장:", out)
    if platform.system() == "Windows":
        os.startfile(str(out))
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(out)])
