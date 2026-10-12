"""Step-4 inspection dashboard V8.

V8 change summary
-----------------
1) Keeps the V7 sequence-gated checklist idea.
2) Fixes the main complaint from V7: the 3rd checkbox no longer depends only on
   YOLO direct TOP. It can be satisfied by either:
      - direct TOP accumulation, or
      - review TOP accumulation (TOP_REVIEW) with concurrent hand/arm evidence.
3) Right-side wording is shortened.
4) Designed for short clips or long/full videos.

Important interpretation:
- Check marks mean visual evidence was observed inside the current Step-4 episode.
- This is still not a final ground-truth proof of human visual inspection.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from step4_glove_top_v3 import (
    get_segments,
    choose_side,
    white_surface_review,
    find_pair,
    draw_mask,
)

ROOT = Path(__file__).resolve().parent

# BGR palette
BG = (8, 9, 10)
PANEL = (12, 14, 16)
HEADER = (9, 11, 13)
WHITE = (245, 245, 245)
MUTED = (150, 158, 168)
DIM = (105, 112, 122)
GREEN = (72, 220, 88)
YELLOW = (0, 220, 255)
BLUE = (238, 145, 35)
CYAN = (235, 190, 65)
RED = (72, 72, 230)
LINE = (48, 52, 58)


def cvtxt(im, s, x, y, color=WHITE, scale=.55, thick=2):
    cv2.putText(im, str(s), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def ko_draw(im, items, font_path: Path):
    try:
        from PIL import Image, ImageDraw, ImageFont
        if not font_path.is_file():
            return False
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        d = ImageDraw.Draw(pil)
        cache = {}
        for text, xy, size, bgr, anchor in items:
            if size not in cache:
                cache[size] = ImageFont.truetype(str(font_path), size)
            rgbc = (bgr[2], bgr[1], bgr[0])
            d.text(xy, text, font=cache[size], fill=rgbc, anchor=anchor)
        im[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return True
    except Exception:
        return False


def pose_arm(pose_result, w, h, arm):
    if pose_result is None or not pose_result.pose_landmarks:
        return None
    ids = (11, 13, 15) if arm == 'L' else (12, 14, 16)
    pts = []
    for idx in ids:
        lm = pose_result.pose_landmarks.landmark[idx]
        threshold = .35 if idx in (13, 14) else .45
        inside = -.08 <= lm.x <= 1.08 and -.08 <= lm.y <= 1.08
        pt = (float(lm.x * w), float(lm.y * h)) if lm.visibility >= threshold and inside else None
        pts.append((pt, float(lm.visibility)))
    return pts


def _arm_glove_score(arm_pose, glove, diag):
    if arm_pose is None or arm_pose[1][0] is None:
        return None
    shoulder = arm_pose[0][0]
    elbow = arm_pose[1][0]
    wrist = arm_pose[2][0]
    g = glove['center']

    d = math.dist(elbow, g) / diag
    if not (.025 <= d <= .38):
        return None
    score = d

    if wrist is not None:
        wd = math.dist(wrist, g) / diag
        score -= max(0.0, .055 - wd) * .65

    if shoulder is not None:
        a = np.array(elbow, dtype=float) - np.array(shoulder, dtype=float)
        b = np.array(g, dtype=float) - np.array(elbow, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-6 and nb > 1e-6:
            cosang = float(np.dot(a, b) / (na * nb))
            if cosang < -0.55:
                score += .10
            elif cosang < -0.10:
                score += .04

    score -= min(.03, glove['conf'] * .015)
    return score


def fuse_pose_to_gloves(lp, rp, gloves, pair, w, h):
    diag = math.hypot(w, h)
    candidates = list(pair[1:3]) if pair is not None else sorted(
        gloves, key=lambda d: d['conf'], reverse=True
    )[:4]

    out = {a: {'glove': None, 'proxy': None, 'native_near': False} for a in ('L', 'R')}
    options = []
    for arm, pose in [('L', lp), ('R', rp)]:
        for gi, g in enumerate(candidates):
            s = _arm_glove_score(pose, g, diag)
            if s is not None:
                options.append((s, arm, gi))

    used_a, used_g = set(), set()
    for _, arm, gi in sorted(options):
        if arm in used_a or gi in used_g:
            continue
        g = candidates[gi]
        out[arm]['glove'] = g
        out[arm]['proxy'] = g['center']
        pose = lp if arm == 'L' else rp
        native = pose[2][0] if pose else None
        if native is not None:
            out[arm]['native_near'] = math.dist(native, g['center']) / diag <= .05
        used_a.add(arm)
        used_g.add(gi)
    return out


def segment_status(raw, seg_result, top_id, side_id, anchor, seed_idx, idx, fps, w, h):
    found = get_segments(seg_result, w, h)
    direct = max(
        (d for d in found if d['cls'] == top_id and d['conf'] >= .40 and d['area'] >= .03),
        key=lambda d: d['conf'], default=None
    )

    side = kind = white = None
    if direct is not None:
        anchor = {**direct, 'frame': idx}
        seed_idx = idx
        return 'TOP_DIRECT', direct, None, None, anchor, seed_idx

    if seed_idx is not None and anchor is not None and (idx - seed_idx) / fps <= .85:
        white = white_surface_review(raw, anchor, w, h, v_min=215, s_max=60)
        side, kind = choose_side(
            [d for d in found if d['cls'] == side_id],
            anchor, idx, fps, max_age=.65
        )

    if white is not None:
        status = 'TOP_REVIEW'
    elif kind == 'SIDE_REVIEW':
        status = 'TOP_REVIEW'
    elif kind == 'PLASTIC_RISK':
        status = 'PLASTIC_RISK'
    else:
        status = 'NO_TOP'

    return status, direct, white, side, anchor, seed_idx


def draw_mask_outline(frame, mask, color, thickness=2, alpha=0.0):
    if mask is None:
        return
    binary = (mask > .5).astype(np.uint8)
    if alpha > 0:
        draw_mask(frame, binary.astype(bool), color, alpha)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness, cv2.LINE_AA)


def draw_label_box(frame, text, x, y, fg, bg=(15, 18, 21), scale=.42):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(2, int(x)); y = max(th + 8, int(y))
    cv2.rectangle(frame, (x, y-th-8), (x+tw+10, y+3), bg, -1)
    cv2.rectangle(frame, (x, y-th-8), (x+tw+10, y+3), fg, 1)
    cv2.putText(frame, text, (x+5, y-3), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, 1, cv2.LINE_AA)


def draw_arm(frame, arm_pose, fused, hand_name):
    if arm_pose is None:
        return
    shoulder, elbow, native_wrist = (v[0] for v in arm_pose)

    if shoulder is not None and elbow is not None:
        cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), BLUE, 3, cv2.LINE_AA)
    if shoulder is not None:
        cv2.circle(frame, tuple(map(int, shoulder)), 5, YELLOW, -1, cv2.LINE_AA)
    if elbow is not None:
        cv2.circle(frame, tuple(map(int, elbow)), 6, YELLOW, -1, cv2.LINE_AA)

    if native_wrist is not None:
        cv2.circle(frame, tuple(map(int, native_wrist)), 5, WHITE, 1, cv2.LINE_AA)

    g = fused['glove']
    if g is None:
        return

    x1, y1, x2, y2 = map(int, g['bbox'])
    gc = tuple(map(int, g['center']))
    cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2, cv2.LINE_AA)

    if elbow is not None:
        cv2.line(frame, tuple(map(int, elbow)), gc, BLUE, 3, cv2.LINE_AA)
    cv2.circle(frame, gc, 6, YELLOW, -1, cv2.LINE_AA)

    draw_label_box(frame, f'{hand_name} {g["conf"]:.2f}', x1, y1-4, BLUE)


def draw_checkbox(canvas, x, y, mode):
    size = 24
    if mode == 'done':
        col = GREEN
        cv2.rectangle(canvas, (x, y), (x+size, y+size), col, 2, cv2.LINE_AA)
        cv2.line(canvas, (x+5, y+13), (x+10, y+18), col, 3, cv2.LINE_AA)
        cv2.line(canvas, (x+10, y+18), (x+20, y+6), col, 3, cv2.LINE_AA)
    elif mode == 'active':
        col = YELLOW
        cv2.rectangle(canvas, (x, y), (x+size, y+size), col, 2, cv2.LINE_AA)
    else:
        col = (135, 141, 151)
        cv2.rectangle(canvas, (x, y), (x+size, y+size), col, 2, cv2.LINE_AA)


def draw_badge(canvas, text, x, y, font_path, color=YELLOW):
    try:
        from PIL import Image, ImageDraw, ImageFont
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        d = ImageDraw.Draw(pil)
        f = ImageFont.truetype(str(font_path), 16)
        bbox = d.textbbox((0, 0), text, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        d.rounded_rectangle((x, y, x+tw+28, y+th+18), radius=6,
                            fill=(color[2], color[1], color[0]))
        d.text((x+14, y+7), text, font=f, fill=(15, 15, 15))
        canvas[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, default=ROOT / 'crop008.mov')
    ap.add_argument('--seg-model', type=Path, default=ROOT / 'armrest_3class_best.pt')
    ap.add_argument('--glove-model', type=Path,
                    default=ROOT / 'glove_runs' / 'glove_detector_v2' / 'weights' / 'best.pt')
    ap.add_argument('--start', type=float, default=11.0)
    ap.add_argument('--end', type=float, default=15.0)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'step4_pose_dashboard_v8_hybrid_top')
    ap.add_argument('--pair-hold', type=float, default=.20)
    ap.add_argument('--pose-hold', type=float, default=.20)
    ap.add_argument('--top-hold-direct', type=float, default=.20,
                    help='Direct top accumulated seconds needed for checkbox 3')
    ap.add_argument('--top-hold-review', type=float, default=.40,
                    help='Review top accumulated seconds needed for checkbox 3')
    ap.add_argument('--reset-gap', type=float, default=2.50)
    ap.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/malgun.ttf'))
    args = ap.parse_args()

    for label, path in [('video', args.video), ('seg model', args.seg_model), ('glove model', args.glove_model)]:
        if not path.is_file():
            raise FileNotFoundError(f'{label}: {path}')

    import mediapipe as mp
    from ultralytics import YOLO

    seg = YOLO(str(args.seg_model))
    glove_model = YOLO(str(args.glove_model))
    names = {int(k): str(v) for k, v in seg.names.items()}
    top_id = next((k for k, v in names.items() if v == 'top'), None)
    side_id = next((k for k, v in names.items() if v == 'side'), None)
    if top_id is None or side_id is None:
        raise ValueError(f'Expected top/side classes, got {names}')

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.full:
        start = 0
        end = total_frames
        start_s = 0.0
        end_s = total_frames / fps
        range_tag = 'full'
    else:
        start = max(0, round(args.start * fps))
        end = min(total_frames, round(args.end * fps))
        if end <= start:
            raise ValueError('Bad --start/--end')
        start_s = start / fps
        end_s = end / fps
        range_tag = f'{args.start:g}_{args.end:g}'

    print(f'Video range: {start_s:.2f}s ~ {end_s:.2f}s  ({end-start} frames)')

    view_w = 1080
    view_h = round(h * view_w / w)
    panel_w = 360
    header_h = 82
    out_w = view_w + panel_w
    out_h = header_h + view_h

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f'{args.video.stem}_{range_tag}_step4_v8'
    vpath = args.out / f'{stem}.mp4'
    cpath = args.out / f'{stem}.csv'

    writer = cv2.VideoWriter(
        str(vpath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f'Cannot write {vpath}')

    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=.45,
        min_tracking_confidence=.45,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    anchor = None
    seed_idx = None
    glove_prior = None
    last_glove = None

    n_pair = n_pose = n_direct = n_review = n_proxy = n_native = 0
    max_gap = 0.0
    processed = 0

    step_state = 'WAITING'  # WAITING -> ACTIVE -> COMPLETE
    ev_pair = False
    ev_pose = False
    ev_top = False
    top_evidence_mode = 'NONE'  # NONE / DIRECT / REVIEW
    pair_streak = 0
    pose_streak = 0
    top_direct_frames = 0
    top_review_frames = 0
    last_relevant_idx = None
    step_start_idx = None

    try:
        with cpath.open('w', newline='', encoding='utf-8-sig') as fp:
            cw = csv.writer(fp)
            cw.writerow([
                'frame', 'time_s', 'seg_status', 'top_direct', 'top_review_support',
                'two_gloves', 'gap_norm', 'pose_elbows', 'left_glove_proxy', 'right_glove_proxy',
                'left_native_wrist_near_glove', 'right_native_wrist_near_glove',
                'step_state', 'check_pair', 'check_pose', 'check_top', 'top_evidence_mode'
            ])

            for idx in range(start, end):
                ok, raw = cap.read()
                if not ok:
                    break
                processed += 1
                t = idx / fps

                # 1) Armrest segmentation
                sr = seg.predict(
                    raw, imgsz=512, conf=.20, device='cpu', retina_masks=True, verbose=False
                )[0]
                status, direct, white, side, anchor, seed_idx = segment_status(
                    raw, sr, top_id, side_id, anchor, seed_idx, idx, fps, w, h
                )

                # 2) Glove detection
                gr = glove_model.predict(
                    raw, imgsz=640, conf=.25, device='cpu', verbose=False
                )[0]
                elapsed = (idx - last_glove) / fps if last_glove is not None else 999.0
                prior = glove_prior if elapsed <= .32 else None
                all_gloves, pair = find_pair(gr.boxes, w, h, prior, elapsed)

                has_pair = pair is not None
                gap_norm = None
                if has_pair:
                    _, gl, gright, gap_norm = pair
                    max_gap = max(max_gap, gap_norm)
                    glove_prior = (gl['center'], gright['center'])
                    last_glove = idx
                    n_pair += 1
                elif elapsed > .32:
                    glove_prior = None

                # 3) MediaPipe Pose
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pr = pose_detector.process(rgb)
                lp = pose_arm(pr, w, h, 'L')
                rp = pose_arm(pr, w, h, 'R')
                pose_ok = any(p is not None and p[1][0] is not None for p in (lp, rp))
                if pose_ok:
                    n_pose += 1

                fused = fuse_pose_to_gloves(lp, rp, all_gloves, pair, w, h)
                proxy_count = int(fused['L']['proxy'] is not None) + int(fused['R']['proxy'] is not None)
                n_proxy += proxy_count
                n_native += int(fused['L']['native_near']) + int(fused['R']['native_near'])
                if direct is not None:
                    n_direct += 1

                # ------------------ Step-4 episode state machine ------------------
                two_arm_links = (fused['L']['proxy'] is not None and fused['R']['proxy'] is not None)
                top_candidate_now = status in ('TOP_DIRECT', 'TOP_REVIEW')
                hand_activity_now = has_pair or proxy_count > 0
                relevant_now = top_candidate_now or hand_activity_now or status == 'PLASTIC_RISK'
                review_supported_now = (status == 'TOP_REVIEW') and (has_pair or two_arm_links or proxy_count > 0)
                if review_supported_now:
                    n_review += 1

                if relevant_now:
                    last_relevant_idx = idx

                if step_state == 'WAITING' and top_candidate_now and hand_activity_now:
                    step_state = 'ACTIVE'
                    step_start_idx = idx
                    ev_pair = ev_pose = ev_top = False
                    top_evidence_mode = 'NONE'
                    pair_streak = pose_streak = top_direct_frames = top_review_frames = 0

                if step_state in ('ACTIVE', 'COMPLETE'):
                    pair_streak = pair_streak + 1 if has_pair else 0
                    pose_streak = pose_streak + 1 if two_arm_links else 0
                    if direct is not None:
                        top_direct_frames += 1
                    if review_supported_now:
                        top_review_frames += 1

                    if not ev_pair and pair_streak / fps >= args.pair_hold:
                        ev_pair = True
                    if ev_pair and not ev_pose and pose_streak / fps >= args.pose_hold:
                        ev_pose = True
                    if ev_pair and ev_pose and not ev_top:
                        if top_direct_frames / fps >= args.top_hold_direct:
                            ev_top = True
                            top_evidence_mode = 'DIRECT'
                        elif top_review_frames / fps >= args.top_hold_review:
                            ev_top = True
                            top_evidence_mode = 'REVIEW'

                    if ev_pair and ev_pose and ev_top:
                        step_state = 'COMPLETE'

                    if last_relevant_idx is not None and (idx - last_relevant_idx) / fps > args.reset_gap:
                        step_state = 'WAITING'
                        step_start_idx = None
                        ev_pair = ev_pose = ev_top = False
                        top_evidence_mode = 'NONE'
                        pair_streak = pose_streak = top_direct_frames = top_review_frames = 0

                # ------------------ Video overlays ------------------
                overlay = raw.copy()

                candidate_mask = None
                if direct is not None:
                    candidate_mask = direct['mask']
                    draw_mask_outline(overlay, candidate_mask, GREEN, thickness=2, alpha=.08)
                elif white is not None:
                    candidate_mask = white['mask']
                    draw_mask_outline(overlay, candidate_mask, YELLOW, thickness=2, alpha=.03)
                elif side is not None and status == 'TOP_REVIEW':
                    candidate_mask = side['mask']
                    draw_mask_outline(overlay, candidate_mask, YELLOW, thickness=2, alpha=.02)
                elif side is not None and status == 'PLASTIC_RISK':
                    draw_mask_outline(overlay, side['mask'], RED, thickness=2, alpha=.02)

                draw_arm(overlay, lp, fused['L'], '손(좌)')
                draw_arm(overlay, rp, fused['R'], '손(우)')

                matched_ids = {id(fused['L']['glove']), id(fused['R']['glove'])}
                for g in all_gloves:
                    if id(g) in matched_ids:
                        continue
                    x1, y1, x2, y2 = map(int, g['bbox'])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 110, 120), 1, cv2.LINE_AA)

                if has_pair:
                    p1 = tuple(map(int, pair[1]['center']))
                    p2 = tuple(map(int, pair[2]['center']))
                    cv2.line(overlay, p1, p2, YELLOW, 2, cv2.LINE_AA)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    draw_label_box(
                        overlay, f'gap {gap_norm:.3f}', mid[0]-35, mid[1]-8,
                        YELLOW, bg=(18, 20, 22), scale=.40
                    )

                # ------------------ Layout ------------------
                view = cv2.resize(overlay, (view_w, view_h))
                canvas = np.zeros((out_h, out_w, 3), np.uint8)
                canvas[:] = BG
                canvas[header_h:, :view_w] = view
                canvas[header_h:, view_w:] = PANEL

                cv2.rectangle(canvas, (0, 0), (out_w, header_h), HEADER, -1)
                cv2.line(canvas, (0, header_h-1), (out_w, header_h-1), LINE, 1)
                cv2.line(canvas, (view_w, header_h), (view_w, out_h), LINE, 1)

                frame_no = idx - start + 1
                total_no = end - start
                current_gloves = 2 if has_pair else sum(
                    g is not None for g in [fused['L']['glove'], fused['R']['glove']]
                )
                current_arms = int(lp is not None and lp[1][0] is not None) + int(
                    rp is not None and rp[1][0] is not None
                )
                evidence_count = int(ev_pair) + int(ev_pose) + int(ev_top)

                if step_state == 'WAITING':
                    current_name = '4단계 시작 대기'
                    current_col = MUTED
                elif step_state == 'COMPLETE':
                    if top_evidence_mode == 'DIRECT':
                        current_name = '근거 3개 확보 · direct top'
                    elif top_evidence_mode == 'REVIEW':
                        current_name = '근거 3개 확보 · review top'
                    else:
                        current_name = '근거 3개 확보'
                    current_col = GREEN
                elif direct is not None:
                    current_name = '진행 중 · 윗면 직접 검출'
                    current_col = GREEN
                elif review_supported_now:
                    current_name = '진행 중 · 윗면 보정 후보'
                    current_col = YELLOW
                elif status == 'PLASTIC_RISK':
                    current_name = '진행 중 · 비닐 간섭 검토'
                    current_col = RED
                else:
                    current_name = '진행 중 · 다음 근거 대기'
                    current_col = MUTED

                if direct is not None:
                    draw_badge(canvas, '윗면 직접 검출', 28, header_h + 22, args.font, GREEN)
                elif review_supported_now:
                    draw_badge(canvas, '윗면 보정 후보', 28, header_h + 22, args.font, YELLOW)

                px = view_w + 28
                row3_active = step_state != 'WAITING' and (direct is not None or review_supported_now)
                rows = [
                    (138, 'done' if ev_pair else ('active' if step_state != 'WAITING' and has_pair else 'wait')),
                    (244, 'done' if ev_pose else ('active' if step_state != 'WAITING' and two_arm_links else 'wait')),
                    (350, 'done' if ev_top else ('active' if row3_active else 'wait')),
                ]
                for yy, mode in rows:
                    draw_checkbox(canvas, px, header_h + yy, mode)

                cv2.line(canvas, (px, header_h + 430), (out_w - 24, header_h + 430), LINE, 1)

                top_sub = 'direct top 또는 review 후보 누적'
                top_small = 'review는 장갑/팔 근거 동반 시만 인정'
                top_mode_text = '미확정'
                if top_evidence_mode == 'DIRECT':
                    top_mode_text = 'direct top 기준 충족'
                elif top_evidence_mode == 'REVIEW':
                    top_mode_text = 'review top 기준 충족'

                ko_items = [
                    # Header
                    (f'SAVED VIDEO  {frame_no}/{total_no}   {t:.1f}s', (20, 12), 16, WHITE, 'la'),
                    ('확인 중: 제품 1 · 4단계 윗면 육안 검사', (20, 39), 15, GREEN, 'la'),
                    (f'장갑 {current_gloves}/2    팔 관절 {current_arms}/2    gap {gap_norm if gap_norm is not None else "-"}', (20, 63), 12, (110, 205, 235), 'la'),

                    # Right panel title
                    ('직원 기준 작업 검사', (px, header_h + 32), 23, WHITE, 'la'),
                    (f'검출 근거: {evidence_count}/3', (px, header_h + 66), 16, GREEN, 'la'),
                    ((('단계 대기' if step_state == 'WAITING' else ('근거 확보' if step_state == 'COMPLETE' else '단계 진행 중'))), (px, header_h + 92), 11, MUTED, 'la'),

                    # Row 1
                    ('양손 장갑 검출', (px + 40, header_h + 136), 15, WHITE, 'la'),
                    ('양손 장갑이 일정 시간 함께 검출', (px + 40, header_h + 160), 10, MUTED, 'la'),

                    # Row 2
                    ('팔 관절-장갑 연결', (px + 40, header_h + 242), 15, WHITE, 'la'),
                    ('팔 관절 + glove proxy 동시 확인', (px + 40, header_h + 266), 10, MUTED, 'la'),

                    # Row 3
                    ('암레스트 윗면 근거', (px + 40, header_h + 348), 15, WHITE, 'la'),
                    (top_sub, (px + 40, header_h + 372), 10, MUTED, 'la'),
                    (top_small, (px + 40, header_h + 392), 10, MUTED, 'la'),

                    # Current frame status
                    ('현재 상태', (px, header_h + 452), 11, MUTED, 'la'),
                    (current_name, (px, header_h + 477), 13, current_col, 'la'),
                    (f'윗면 체크 기준: {top_mode_text}', (px, header_h + 500), 10, MUTED, 'la'),

                    # Bottom note
                    ('※ 체크는 현재 4단계 구간 기준으로만 유지', (px, out_h - 48), 9, DIM, 'la'),
                    ('긴 비활성 구간 뒤에는 다음 제품으로 보고 초기화', (px, out_h - 28), 9, DIM, 'la'),
                ]

                if not ko_draw(canvas, ko_items, args.font):
                    cvtxt(canvas, f'SAVED VIDEO {frame_no}/{total_no} {t:.1f}s', 20, 26, WHITE, .50, 2)
                    cvtxt(canvas, 'STEP 4 TOP VISUAL CHECK', 20, 54, GREEN, .42, 1)
                    cvtxt(canvas, 'WORK INSPECTION', px, header_h + 36, WHITE, .58, 2)
                    cvtxt(canvas, f'EVIDENCE {evidence_count}/3', px, header_h + 64, GREEN, .42, 1)

                writer.write(canvas)

                cw.writerow([
                    idx + 1,
                    round(t, 3),
                    status,
                    int(direct is not None),
                    int(review_supported_now),
                    int(has_pair),
                    round(gap_norm, 4) if gap_norm is not None else '',
                    int(pose_ok),
                    int(fused['L']['proxy'] is not None),
                    int(fused['R']['proxy'] is not None),
                    int(fused['L']['native_near']),
                    int(fused['R']['native_near']),
                    step_state, int(ev_pair), int(ev_pose), int(ev_top), top_evidence_mode,
                ])

                if processed % 30 == 0:
                    print(f'Progress: {processed}/{end-start}')

    finally:
        cap.release()
        writer.release()
        pose_detector.close()

    print(f'Processed {processed} frames')
    print(f'Two gloves={n_pair}; Pose elbows={n_pose}; direct TOP={n_direct}; review-supported TOP={n_review}')
    print(f'Elbow-to-glove proxies(total assignments)={n_proxy}; native MP wrists near glove={n_native}')
    print(f'Max normalized glove gap={max_gap:.3f}')
    print('Video:', vpath)
    print('CSV:', cpath)


if __name__ == '__main__':
    main()
