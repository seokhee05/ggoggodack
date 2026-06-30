import cv2
import mediapipe as mp
import urllib.request
import os
import time

# ── 모델 경로 ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
POSE_MODEL = os.path.join(BASE, "pose_landmarker_full.task")
HAND_MODEL = os.path.join(BASE, "hand_landmarker.task")

POSE_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def download(path, url, name):
    if not os.path.exists(path):
        print(f"{name} 다운로드 중...")
        urllib.request.urlretrieve(url, path)
        print(f"{name} 완료!")

# ── 전신 33 관절 연결 ──────────────────────────────────────────────────────
BODY_CONNECTIONS = [
    # 얼굴 (방향 파악용)
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 어깨
    (11, 12),
    # 왼팔
    (11, 13), (13, 15),
    # 왼손 (포즈 레벨)
    (15, 17), (15, 19), (15, 21), (17, 19),
    # 오른팔
    (12, 14), (14, 16),
    # 오른손 (포즈 레벨)
    (16, 18), (16, 20), (16, 22), (18, 20),
    # 몸통
    (11, 23), (12, 24), (23, 24),
    # 왼다리
    (23, 25), (25, 27),
    # 왼발
    (27, 29), (29, 31), (27, 31),
    # 오른다리
    (24, 26), (26, 28),
    # 오른발
    (28, 30), (30, 32), (28, 32),
]

# 부위별 랜드마크 색상 (BGR)
def lm_color(i):
    if i <= 10:                return (0,   230, 255)   # 얼굴  - 노랑
    if i in (11, 12):          return (0,   255, 180)   # 어깨  - 민트
    if i in (13, 15, 17, 19, 21): return (50, 160, 255) # 왼팔  - 주황
    if i in (14, 16, 18, 20, 22): return (255, 80,  60) # 오른팔- 파랑
    if i in (23, 24):          return (0,   255, 180)   # 골반  - 민트
    if i in (25, 27, 29, 31):  return (100, 210, 255)   # 왼다리- 하늘
    return                            (255, 160, 100)   # 오른다리- 살구

def conn_color(a, b):
    pair = {a, b}
    face      = set(range(11))
    left_arm  = {11, 13, 15, 17, 19, 21}
    right_arm = {12, 14, 16, 18, 20, 22}
    left_leg  = {23, 25, 27, 29, 31}
    right_leg = {24, 26, 28, 30, 32}
    if pair <= face:           return (0,   200, 255)
    if pair & left_arm:        return (50,  140, 255)
    if pair & right_arm:       return (255,  70,  50)
    if pair & left_leg:        return (100, 200, 255)
    if pair & right_leg:       return (255, 150,  80)
    return                            (80,  255, 150)   # 몸통

# ── 손가락 21 관절 연결 ────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
FINGER_COLORS = [
    (80,  80, 255),
    (80, 160, 255),
    (80, 255, 200),
    (80, 255,  80),
    (255,140,  80),
]

def finger_lm_color(i):
    if i == 0:  return (220, 220, 220)
    return FINGER_COLORS[(i - 1) // 4]

def finger_conn_color(a, b):
    def fi(i): return -1 if i == 0 else (i - 1) // 4
    fa, fb = fi(a), fi(b)
    if fa == fb and fa >= 0: return FINGER_COLORS[fa]
    return (200, 200, 200)

# ── 그리기 ─────────────────────────────────────────────────────────────────
def draw_body(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in BODY_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], conn_color(a, b), 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        col = lm_color(i)
        r   = 3 if i <= 10 else 5
        cv2.circle(frame, pt, r, col,           -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255,255,255),  1, cv2.LINE_AA)

def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], finger_conn_color(a, b), 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        col = finger_lm_color(i)
        r   = 4 if i == 0 else 3
        cv2.circle(frame, pt, r, col,           -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255,255,255),  1, cv2.LINE_AA)

# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    download(POSE_MODEL, POSE_URL, "포즈 모델(전신)")
    download(HAND_MODEL, HAND_URL, "손 모델(손가락)")

    pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    hand_opt = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=8,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start_ms = int(time.time() * 1000)

    with (
        mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm,
        mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(time.time() * 1000) - start_ms

            pose_r = pose_lm.detect_for_video(mp_img, ts_ms)
            hand_r = hand_lm.detect_for_video(mp_img, ts_ms)

            bodies = 0
            if pose_r.pose_landmarks:
                bodies = len(pose_r.pose_landmarks)
                for plm in pose_r.pose_landmarks:
                    draw_body(frame, plm, w, h)

            hands = 0
            if hand_r.hand_landmarks:
                hands = len(hand_r.hand_landmarks)
                for hlm in hand_r.hand_landmarks:
                    draw_hand(frame, hlm, w, h)

            cv2.putText(frame, f"People: {bodies}  Hands: {hands}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q to quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow("Pose Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
