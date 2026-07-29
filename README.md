# Real-Time Pose & Hand Tracking

OpenCV와 MediaPipe Tasks를 사용하여 웹캠 영상에서 여러 사람의 전신 포즈와 손 랜드마크를 실시간으로 추적하는 Python 프로젝트입니다.

각 신체 부위와 손가락을 서로 다른 색상으로 시각화하며, 현재 감지된 사람과 손의 개수를 화면에 표시합니다.

## Features

- 실시간 웹캠 영상 처리
- 최대 4명의 전신 포즈 동시 추적
- 사람당 33개의 포즈 랜드마크 검출
- 최대 8개의 손 동시 추적
- 손당 21개의 손 랜드마크 검출
- 신체 부위 및 손가락별 색상 구분
- 감지된 사람과 손 개수 표시
- MediaPipe 모델 파일 자동 다운로드

## Tech Stack

| 구분 | 사용 기술 |
|---|---|
| Language | Python 3.11 |
| Computer Vision | OpenCV |
| Pose Estimation | MediaPipe Pose Landmarker |
| Hand Tracking | MediaPipe Hand Landmarker |
| Input | Webcam |

## Quick Start for Windows

### 1. 프로젝트 다운로드

GitHub 저장소에서 `Code → Download ZIP`을 누른 후 압축을 해제합니다.

Git이 이미 설치되어 있다면 다음 명령으로 내려받을 수도 있습니다.

```bat
git clone https://github.com/seokhee05/ggoggodack.git
cd ggoggodack
```

### 2. 전체 개발 환경 자동 설치

압축을 해제한 프로젝트 폴더에서 CMD를 열고 다음 명령을 실행합니다.

```bat
setup.bat
```

`setup.bat`은 다음 항목을 자동으로 설치하고 설정합니다.

- Python 3.11
- Git
- Visual Studio Code
- Python 가상환경 `.venv`
- MediaPipe
- OpenCV

> `setup.bat`은 Windows 10/11의 `winget`을 사용합니다. `winget`이 없다면 Microsoft Store에서 **App Installer**를 먼저 설치해야 합니다.

### 3. 프로그램 실행

```bat
.venv\Scripts\python.exe test.py
```

프로그램을 종료하려면 OpenCV 영상 창을 클릭한 후 `Q` 키를 누릅니다.

## Manual Installation

### Windows

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test.py
```

## Project Structure

```text
ggoggodack/
├── test.py               # 실시간 포즈 및 손 추적 프로그램
├── requirements.txt      # Python 라이브러리 목록
├── setup.bat             # Windows 개발 환경 자동 설치
├── README.md             # 프로젝트 설명 및 실행 방법
└── .gitignore            # Git에서 제외할 파일 목록
```

프로그램을 처음 실행하면 `pose_landmarker_full.task`와 `hand_landmarker.task`가 자동으로 다운로드됩니다. 모델 파일은 `.gitignore`에 등록되어 있으므로 GitHub에는 업로드되지 않습니다.

## How It Works

1. OpenCV를 이용하여 웹캠 프레임을 가져옵니다.
2. 입력 영상을 BGR 형식에서 RGB 형식으로 변환합니다.
3. 영상을 MediaPipe 이미지 객체로 변환합니다.
4. Pose Landmarker로 전신 랜드마크를 검출합니다.
5. Hand Landmarker로 손 랜드마크를 검출합니다.
6. 검출한 랜드마크와 관절 연결선을 영상에 표시합니다.
7. 감지된 사람과 손의 개수를 화면에 출력합니다.

## Default Configuration

| 설정 | 기본값 | 설명 |
|---|---:|---|
| Camera index | `0` | 사용할 웹캠 번호 |
| Frame width | `1280` | 입력 영상 너비 |
| Frame height | `720` | 입력 영상 높이 |
| Maximum poses | `4` | 동시에 검출할 최대 사람 수 |
| Maximum hands | `8` | 동시에 검출할 최대 손 수 |
| Detection confidence | `0.5` | 최소 검출 신뢰도 |
| Tracking confidence | `0.5` | 최소 추적 신뢰도 |
| Presence confidence | `0.5` | 최소 랜드마크 존재 신뢰도 |

## Configuration

다른 카메라를 사용하려면 `test.py`에서 카메라 번호를 변경합니다.

```python
cap = cv2.VideoCapture(1)
```

실행 속도가 느린 경우 해상도를 낮출 수 있습니다.

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## Dependencies

`requirements.txt`에는 다음 외부 Python 라이브러리가 포함됩니다.

```text
mediapipe
opencv-python
```

`os`, `time`, `urllib.request`는 Python 표준 라이브러리이므로 별도로 설치하지 않습니다.

## Troubleshooting

### `winget`을 찾을 수 없는 경우

Microsoft Store에서 **App Installer**를 설치한 뒤 새 CMD 창에서 `setup.bat`을 다시 실행합니다.

### Python 라이브러리를 찾을 수 없는 경우

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 웹캠이 열리지 않는 경우

- 다른 프로그램이 웹캠을 사용 중인지 확인합니다.
- Windows의 `설정 → 개인정보 및 보안 → 카메라`에서 카메라 접근을 허용합니다.
- `cv2.VideoCapture(0)`의 값을 `1` 또는 `2`로 변경합니다.

### 모델 다운로드에 실패하는 경우

인터넷 연결과 방화벽 설정을 확인합니다. 크기가 0KB인 `.task` 파일이 생성되었다면 삭제한 후 프로그램을 다시 실행합니다.

## Notes

- 카메라 영상은 로컬 컴퓨터에서 처리됩니다.
- 최초 실행 시 MediaPipe 모델 다운로드를 위해 인터넷 연결이 필요합니다.
- 검출 정확도와 실행 속도는 조명, 카메라 품질 및 컴퓨터 성능에 따라 달라질 수 있습니다.

## References

- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/tasks)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV](https://opencv.org/)
