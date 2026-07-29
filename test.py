import cv2
import json
import mediapipe as mp
import os
import re
import time
import urllib.request
from datetime import datetime


# =============================================================================
# 기본 설정
# =============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL = os.path.join(BASE, "pose_landmarker_full.task")
HAND_MODEL = os.path.join(BASE, "hand_landmarker.task")
ROI_CONFIG = os.path.join(BASE, "roi_config.json")
RECORDINGS_DIR = os.path.join(BASE, "recordings")
RESULTS_DIR = os.path.join(BASE, "results")

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FALLBACK_FPS = 30.0

# 손 중심점이 검사영역 안에 연속으로 머물러야 하는 시간
DWELL_SECONDS = 0.8

# 손이 거의 검출되지 않은 영상은 REVIEW로 처리
MIN_HAND_DETECTION_RATIO = 0.30


# 실제 영상에서 지정할 물리적 검사영역
ROI_DEFINITIONS = [
    ("knob", "Knob"),
    ("hinge", "Hinge"),
    ("screw_1", "Screw 1"),
    ("screw_2", "Screw 2"),
    ("damper_1", "Damper 1"),
    ("damper_2", "Damper 2"),
    ("rib_joint", "Rib / Joint"),
    ("gap", "Lining-Skin Gap"),
]

# 작업표준에 따른 검사 순서
# Knob과 Hinge는 뒤에서 재작동 검사로 한 번 더 확인
CHECKLIST_STEPS = [
    {"id": "knob_check", "label": "Knob 검사", "roi": "knob"},
    {"id": "hinge_check", "label": "Hinge 검사", "roi": "hinge"},
    {"id": "screw_1_check", "label": "Screw 1 검사", "roi": "screw_1"},
    {"id": "screw_2_check", "label": "Screw 2 검사", "roi": "screw_2"},
    {"id": "damper_1_check", "label": "Damper 1 검사", "roi": "damper_1"},
    {"id": "damper_2_check", "label": "Damper 2 검사", "roi": "damper_2"},
    {"id": "rib_joint_check", "label": "Rib·이음부 검사", "roi": "rib_joint"},
    {"id": "knob_recheck", "label": "Knob 재작동 검사", "roi": "knob"},
    {"id": "hinge_recheck", "label": "Hinge 재작동 검사", "roi": "hinge"},
    {"id": "gap_check", "label": "Lining–Skin Gap 검사", "roi": "gap"},
]


# =============================================================================
# 모델 및 파일 유틸리티
# =============================================================================
def download(path, url, name):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    if os.path.exists(path):
        os.remove(path)

    print(f"{name} 다운로드 중...")
    urllib.request.urlretrieve(url, path)
    print(f"{name} 다운로드 완료")


def safe_filename(value):
    value = value.strip()
    value = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value)
    return value[:80] or "UNKNOWN"


def ensure_directories():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def open_camera():
    # Windows에서는 CAP_DSHOW가 카메라 시작 지연을 줄이는 경우가 많음
    if os.name == "nt":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            f"웹캠 {CAMERA_INDEX}번을 열 수 없습니다. "
            "카메라 연결 및 Windows 카메라 권한을 확인하세요."
        )

    return cap


def camera_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if width <= 0:
        width = FRAME_WIDTH
    if height <= 0:
        height = FRAME_HEIGHT
    if fps <= 1 or fps > 120:
        fps = FALLBACK_FPS

    return width, height, fps


# =============================================================================
# ROI 설정
# =============================================================================
def normalize_roi(roi, width, height):
    x, y, w, h = roi
    return {
        "x": x / width,
        "y": y / height,
        "w": w / width,
        "h": h / height,
    }


def denormalize_roi(roi, width, height):
    return (
        int(roi["x"] * width),
        int(roi["y"] * height),
        int(roi["w"] * width),
        int(roi["h"] * height),
    )


def save_roi_config(rois):
    payload = {
        "camera_index": CAMERA_INDEX,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "rois": rois,
    }
    with open(ROI_CONFIG, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_roi_config():
    if not os.path.exists(ROI_CONFIG):
        return None

    try:
        with open(ROI_CONFIG, "r", encoding="utf-8") as file:
            payload = json.load(file)
        rois = payload.get("rois", {})
        required = {key for key, _ in ROI_DEFINITIONS}
        if required.issubset(rois.keys()):
            return rois
    except (OSError, ValueError, TypeError):
        pass

    return None


def capture_stable_frame(cap):
    frame = None
    # 자동 노출이 안정화되도록 여러 프레임을 버림
    for _ in range(30):
        ret, current = cap.read()
        if ret:
            frame = current
    if frame is None:
        raise RuntimeError("ROI 설정용 웹캠 영상을 읽을 수 없습니다.")
    return frame


def configure_rois(cap):
    frame = capture_stable_frame(cap)
    # 실제 녹화 영상도 좌우 반전하므로 ROI 설정 화면도 동일하게 맞춤
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rois = {}
    window_name = "ROI Setup"

    print()
    print("최초 ROI 설정을 시작합니다.")
    print("각 부품 영역을 마우스로 드래그한 뒤 Enter 또는 Space를 누르세요.")
    print("선택을 취소하려면 C를 누르세요.")

    for key, label in ROI_DEFINITIONS:
        guide = frame.copy()
        cv2.putText(
            guide,
            f"Select ROI: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        selected = cv2.selectROI(
            window_name,
            guide,
            showCrosshair=True,
            fromCenter=False,
        )
        x, y, w, h = [int(v) for v in selected]

        if w <= 0 or h <= 0:
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"{label} ROI가 선택되지 않았습니다.")

        rois[key] = normalize_roi((x, y, w, h), width, height)

    cv2.destroyWindow(window_name)
    save_roi_config(rois)
    print(f"ROI 설정 저장 완료: {ROI_CONFIG}")
    return rois


def draw_rois(frame, rois, active_roi=None):
    height, width = frame.shape[:2]

    for key, label in ROI_DEFINITIONS:
        x, y, w, h = denormalize_roi(rois[key], width, height)
        color = (0, 255, 255) if key == active_roi else (90, 180, 90)
        thickness = 3 if key == active_roi else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            frame,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def point_in_roi(point, roi, width, height):
    px, py = point
    x, y, w, h = denormalize_roi(roi, width, height)
    return x <= px <= x + w and y <= py <= y + h


# =============================================================================
# 랜드마크 그리기
# =============================================================================
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGER_COLORS = [
    (80, 80, 255),
    (80, 160, 255),
    (80, 255, 200),
    (80, 255, 80),
    (255, 140, 80),
]


def body_landmark_color(index):
    if index <= 10:
        return (0, 230, 255)
    if index in (11, 12, 23, 24):
        return (0, 255, 180)
    if index in (13, 15, 17, 19, 21):
        return (50, 160, 255)
    if index in (14, 16, 18, 20, 22):
        return (255, 80, 60)
    if index in (25, 27, 29, 31):
        return (100, 210, 255)
    return (255, 160, 100)


def body_connection_color(a, b):
    pair = {a, b}
    face = set(range(11))
    left_arm = {11, 13, 15, 17, 19, 21}
    right_arm = {12, 14, 16, 18, 20, 22}
    left_leg = {23, 25, 27, 29, 31}
    right_leg = {24, 26, 28, 30, 32}

    if pair <= face:
        return (0, 200, 255)
    if pair & left_arm:
        return (50, 140, 255)
    if pair & right_arm:
        return (255, 70, 50)
    if pair & left_leg:
        return (100, 200, 255)
    if pair & right_leg:
        return (255, 150, 80)
    return (80, 255, 150)


def finger_landmark_color(index):
    if index == 0:
        return (220, 220, 220)
    return FINGER_COLORS[(index - 1) // 4]


def finger_connection_color(a, b):
    def finger_index(index):
        return -1 if index == 0 else (index - 1) // 4

    fa = finger_index(a)
    fb = finger_index(b)
    if fa == fb and fa >= 0:
        return FINGER_COLORS[fa]
    return (200, 200, 200)


def draw_body(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    for a, b in BODY_CONNECTIONS:
        cv2.line(
            frame,
            points[a],
            points[b],
            body_connection_color(a, b),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(points):
        color = body_landmark_color(index)
        radius = 3 if index <= 10 else 5
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[a],
            points[b],
            finger_connection_color(a, b),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(points):
        color = finger_landmark_color(index)
        radius = 4 if index == 0 else 3
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, (255, 255, 255), 1, cv2.LINE_AA)


def hand_center(landmarks, width, height):
    # 손목과 손바닥 관절의 평균을 손 중심으로 사용
    palm_indices = (0, 5, 9, 13, 17)
    xs = [landmarks[i].x for i in palm_indices]
    ys = [landmarks[i].y for i in palm_indices]
    return (
        int(sum(xs) / len(xs) * width),
        int(sum(ys) / len(ys) * height),
    )


# =============================================================================
# 1단계: 바코드 입력 후 원본 영상 녹화
# =============================================================================
def record_video(cap, barcode, rois):
    width, height, fps = camera_properties(cap)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{safe_filename(barcode)}_{timestamp}"
    raw_path = os.path.join(RECORDINGS_DIR, f"{session_name}_raw.mp4")

    writer = cv2.VideoWriter(
        raw_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("원본 영상 저장 파일을 생성할 수 없습니다.")

    print()
    print(f"검사 녹화 시작: {raw_path}")
    print("검사가 끝나면 영상 창에서 Q를 누르세요.")

    started_at = datetime.now()
    written_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽지 못해 녹화를 종료합니다.")
            break

        # 저장 영상과 화면 좌표를 동일하게 유지
        frame = cv2.flip(frame, 1)
        writer.write(frame)
        written_frames += 1

        display = frame.copy()
        draw_rois(display, rois)
        cv2.circle(display, (20, 25), 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"REC  Product: {barcode}",
            (38, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Press Q when inspection is complete",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Inspection Recording", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    writer.release()
    cv2.destroyWindow("Inspection Recording")

    ended_at = datetime.now()
    if written_frames == 0:
        raise RuntimeError("저장된 영상 프레임이 없습니다.")

    metadata = {
        "barcode": barcode,
        "session_name": session_name,
        "recording_started_at": started_at.isoformat(timespec="seconds"),
        "recording_ended_at": ended_at.isoformat(timespec="seconds"),
        "raw_video": raw_path,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": written_frames,
    }

    print(f"원본 영상 저장 완료: {raw_path}")
    return metadata


# =============================================================================
# 2단계: 저장 영상 자동 분석
# =============================================================================
def create_landmarkers():
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=8,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return pose_options, hand_options


def analysis_screen_text(frame, current_index, completed, dwell_progress):
    total = len(CHECKLIST_STEPS)
    completed_count = sum(1 for value in completed if value)

    cv2.rectangle(frame, (0, 0), (520, 100), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Checklist: {completed_count}/{total}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if current_index < total:
        current = CHECKLIST_STEPS[current_index]
        cv2.putText(
            frame,
            f"WAIT: {current['id']}",
            (15, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"ROI dwell: {dwell_progress:.1f}/{DWELL_SECONDS:.1f}s",
            (15, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (200, 220, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "ALL CHECKS COMPLETED",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def analyze_video(metadata, rois):
    raw_path = metadata["raw_video"]
    capture = cv2.VideoCapture(raw_path)
    if not capture.isOpened():
        raise RuntimeError(f"저장 영상을 열 수 없습니다: {raw_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 1:
        fps = FALLBACK_FPS

    analyzed_path = os.path.join(
        RECORDINGS_DIR,
        f"{metadata['session_name']}_analyzed.mp4",
    )
    writer = cv2.VideoWriter(
        analyzed_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("분석 영상 저장 파일을 생성할 수 없습니다.")

    completed = [False] * len(CHECKLIST_STEPS)
    completion_times = [None] * len(CHECKLIST_STEPS)
    current_index = 0
    dwell_frames = 0
    required_dwell_frames = max(1, int(DWELL_SECONDS * fps))
    hand_detected_frames = 0
    processed_frames = 0

    pose_options, hand_options = create_landmarkers()

    print()
    print("저장 영상 자동 분석을 시작합니다.")

    with (
        mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_lm,
        mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hand_lm,
    ):
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_index = processed_frames
            processed_frames += 1
            timestamp_ms = int(frame_index * 1000.0 / fps)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                for pose_landmarks in pose_result.pose_landmarks:
                    draw_body(frame, pose_landmarks)

            centers = []
            if hand_result.hand_landmarks:
                hand_detected_frames += 1
                for hand_landmarks in hand_result.hand_landmarks:
                    draw_hand(frame, hand_landmarks)
                    center = hand_center(hand_landmarks, width, height)
                    centers.append(center)
                    cv2.circle(frame, center, 8, (255, 255, 255), 2, cv2.LINE_AA)

            active_roi_key = None
            if current_index < len(CHECKLIST_STEPS):
                current_step = CHECKLIST_STEPS[current_index]
                active_roi_key = current_step["roi"]
                inside = any(
                    point_in_roi(center, rois[active_roi_key], width, height)
                    for center in centers
                )

                if inside:
                    dwell_frames += 1
                    if dwell_frames >= required_dwell_frames:
                        completed[current_index] = True
                        completion_times[current_index] = round(timestamp_ms / 1000, 2)
                        print(
                            f"[완료] {current_step['label']} "
                            f"({completion_times[current_index]}초)"
                        )
                        current_index += 1
                        dwell_frames = 0
                else:
                    # 연속 체류만 검사로 인정
                    dwell_frames = 0

            dwell_progress = min(DWELL_SECONDS, dwell_frames / fps)
            draw_rois(frame, rois, active_roi=active_roi_key)
            analysis_screen_text(
                frame,
                current_index,
                completed,
                dwell_progress,
            )

            writer.write(frame)
            cv2.imshow("Saved Video Analysis", frame)

            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                print("사용자가 분석을 중단했습니다.")
                break

    capture.release()
    writer.release()
    cv2.destroyWindow("Saved Video Analysis")

    hand_ratio = (
        hand_detected_frames / processed_frames
        if processed_frames > 0
        else 0.0
    )
    video_quality = (
        "PASS"
        if hand_ratio >= MIN_HAND_DETECTION_RATIO
        else "REVIEW"
    )

    step_results = []
    for index, step in enumerate(CHECKLIST_STEPS):
        if completed[index]:
            status = "PASS"
        elif video_quality == "REVIEW":
            status = "REVIEW"
        else:
            status = "FAIL"

        step_results.append(
            {
                "id": step["id"],
                "label": step["label"],
                "roi": step["roi"],
                "status": status,
                "completed_at_seconds": completion_times[index],
            }
        )

    missing_steps = [
        item["label"]
        for item in step_results
        if item["status"] != "PASS"
    ]
    all_completed = all(completed)
    sequence_pass = all_completed
    storage_success = os.path.exists(analyzed_path)

    if all_completed and video_quality == "PASS" and storage_success:
        final_result = "PASS"
    elif video_quality == "REVIEW":
        final_result = "REVIEW"
    else:
        final_result = "FAIL"

    result = {
        **metadata,
        "analyzed_video": analyzed_path,
        "analysis_completed_at": datetime.now().isoformat(timespec="seconds"),
        "checklist": step_results,
        "sequence_check": "PASS" if sequence_pass else "FAIL",
        "missing_steps": missing_steps,
        "video_quality": video_quality,
        "hand_detection_ratio": round(hand_ratio, 4),
        "raw_video_saved": os.path.exists(raw_path),
        "analyzed_video_saved": storage_success,
        "final_result": final_result,
        "prototype_notice": (
            "현재 판정은 손이 ROI에 정해진 순서로 접근하고 일정 시간 "
            "체류했는지를 기준으로 합니다. 실제 부품 불량 판정은 포함하지 않습니다."
        ),
    }

    result_path = os.path.join(
        RESULTS_DIR,
        f"{metadata['session_name']}_result.json",
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"제품: {metadata['barcode']}")
    for item in step_results:
        print(f"[{item['status']}] {item['label']}")
    print(f"검사 순서: {result['sequence_check']}")
    print(f"영상 품질: {video_quality}")
    print(f"최종 판정: {final_result}")
    print(f"원본 영상: {raw_path}")
    print(f"분석 영상: {analyzed_path}")
    print(f"검사 결과: {result_path}")
    print("=" * 60)

    return result


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    ensure_directories()
    download(POSE_MODEL, POSE_URL, "포즈 모델(전신)")
    download(HAND_MODEL, HAND_URL, "손 모델(손가락)")

    cap = open_camera()

    try:
        rois = load_roi_config()
        if rois is None:
            rois = configure_rois(cap)
        else:
            print(f"기존 ROI 설정 사용: {ROI_CONFIG}")

        print()
        print("바코드 스캐너로 제품 바코드를 입력하세요.")
        print("일반 키보드로 직접 입력해도 됩니다.")
        barcode = input("제품 바코드: ").strip()
        if not barcode:
            raise RuntimeError("제품 바코드가 입력되지 않았습니다.")

        metadata = record_video(cap, barcode, rois)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    analyze_video(metadata, rois)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 프로그램을 종료했습니다.")
    except Exception as error:
        print(f"\n[오류] {error}")
