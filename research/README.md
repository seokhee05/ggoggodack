"""
========================================================================
로컬 실행 가이드 — Armrest 검사원 행동 추적 시스템
========================================================================

[ 폴더 구조 ]
kkokkodack/
├── data/
│   ├── cycle/                  # crop001~020.mp4 영상
│   ├── inside/                 # 안쪽면 라벨 이미지 (빨간칠)
│   ├── inside_origin/          # 안쪽면 원본 이미지
│   ├── top/                    # 윗면 라벨
│   ├── top_origin/             # 윗면 원본
│   ├── side/                   # 옆면 라벨
│   └── side_origin/            # 옆면 원본
├── dataset/                    # (자동 생성) YOLO-seg 학습 데이터셋
├── augment_frames/             # (자동 생성) 영상에서 추출한 보강 프레임
├── weights/
│   └── armrest_3class_best.pt  # 학습된 모델 가중치
├── 01_setup.sh                 # 환경 설치
├── 02_build_dataset.py         # 학습 데이터셋 빌드
├── 03_extract_frames.py        # 영상에서 보강용 프레임 추출
├── 04_train.sh                 # 학습 실행
├── 05_run_inference.py         # 추론 파이프라인 (영상 → 결과 영상)
└── README.md


[ 1단계: 환경 설치 ]

  pip install ultralytics opencv-python-headless pillow numpy

  GPU가 있으면 PyTorch GPU 버전 설치:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118


[ 2단계: 데이터셋 빌드 ]

  python 02_build_dataset.py --data_root ./data --output ./dataset

  → inside/top/side 라벨+원본 쌍에서 마스크 역추출 → YOLO-seg 포맷 생성


[ 3단계: 영상에서 보강 프레임 추출 (선택) ]

  python 03_extract_frames.py --video_dir ./data/cycle --output ./augment_frames --fps 2

  → crop001~016에서 2fps로 프레임 추출
  → 이 중 side/inside/top이 잘 보이는 핵심 프레임만 골라서 라벨링
  → 라벨링 후 02_build_dataset.py 다시 실행하면 자동 병합


[ 4단계: 학습 ]

  yolo segment train \
    model=yolov8n-seg.pt \
    data=./dataset/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    device=0 \
    project=./runs \
    name=armrest_v4

  ※ CPU면 device=cpu, batch=4, imgsz=512
  ※ GPU면 epochs=50, imgsz=640, batch=16 권장


[ 5단계: 추론 ]

  python 05_run_inference.py --video ./data/cycle/crop017.mp4 --weights ./runs/armrest_v4/weights/best.pt

  → crop017~020으로 테스트

========================================================================
"""
