# Real-Time Pose & Hand Tracking

OpenCV와 MediaPipe Tasks를 사용하여 웹캠 영상에서 여러 사람의 전신 포즈와 손 랜드마크를 실시간으로 추적하는 Python 프로젝트입니다.

각 신체 부위와 손가락을 서로 다른 색상으로 시각화하며, 현재 감지된 사람과 손의 개수를 화면에 표시합니다.

## Features

* 실시간 웹캠 영상 처리
* 최대 4명의 전신 포즈 동시 추적
* 사람당 33개의 포즈 랜드마크 검출
* 최대 8개의 손 동시 추적
* 손당 21개의 손 랜드마크 검출
* 신체 부위별 랜드마크 색상 구분
* 손가락별 랜드마크 색상 구분
* 감지된 사람 및 손 개수 표시
* MediaPipe 모델 파일 자동 다운로드
* Windows, macOS 및 Linux 지원

## Tech Stack

| 구분              | 사용 기술                     |
| --------------- | ------------------------- |
| Language        | Python                    |
| Computer Vision | OpenCV                    |
| Pose Estimation | MediaPipe Pose Landmarker |
| Hand Tracking   | MediaPipe Hand Landmarker |
| Input           | Webcam                    |

## Requirements

프로그램을 실행하려면 다음 환경이 필요합니다.

* Python 3.9 이상
* Python 3.11 권장
* pip 20.3 이상
* 웹캠
* 최초 실행 시 인터넷 연결

## Installation

### 1. Repository Clone

다음 명령어를 사용하여 저장소를 내려받습니다.

```bash
git clone https://github.com/seokhee05/ggoggodack.git
cd ggoggodack
```

Git을 사용하지 않는 경우 저장소 상단의 `Code → Download ZIP`을 선택한 후 압축을 해제할 수 있습니다.

### 2. Virtual Environment

프로젝트별 패키지 충돌을 방지하기 위해 가상환경 사용을 권장합니다.

#### Windows

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

다음 명령어를 사용하여 필요한 Python 라이브러리를 한 번에 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

프로젝트 폴더에서 다음 명령어를 실행합니다.

```bash
python test.py
```

Windows에서 `python` 명령어가 인식되지 않는 경우 다음 명령어를 사용합니다.

```bash
py test.py
```

프로그램을 처음 실행하면 다음 MediaPipe 모델 파일이 자동으로 다운로드됩니다.

```text
pose_landmarker_full.task
hand_landmarker.task
```

모델 파일은 `test.py`와 동일한 디렉터리에 저장되며, 이후 실행부터는 다시 다운로드하지 않습니다.

프로그램을 종료하려면 OpenCV 영상 창을 클릭한 후 `Q` 키를 누릅니다.

## Project Structure

```text
ggoggodack/
├── test.py                      # 실시간 포즈 및 손 추적 프로그램
├── requirements.txt             # Python 패키지 목록
├── README.md                    # 프로젝트 설명
├── pose_landmarker_full.task    # 최초 실행 시 자동 다운로드
└── hand_landmarker.task         # 최초 실행 시 자동 다운로드
```

`.task` 모델 파일은 최초 실행 시 자동으로 생성되므로 GitHub 저장소에 직접 업로드하지 않아도 됩니다.

## How It Works

프로그램은 웹캠에서 프레임을 읽은 후 다음 순서로 처리합니다.

1. OpenCV를 이용하여 웹캠 프레임을 가져옵니다.
2. 입력 영상을 BGR 형식에서 RGB 형식으로 변환합니다.
3. RGB 영상을 MediaPipe 이미지 객체로 변환합니다.
4. Pose Landmarker를 사용하여 전신 랜드마크를 검출합니다.
5. Hand Landmarker를 사용하여 손 랜드마크를 검출합니다.
6. 검출된 랜드마크와 관절 연결선을 영상 위에 표시합니다.
7. 감지된 사람과 손의 개수를 화면에 출력합니다.

## Default Configuration

기본 검출 설정은 다음과 같습니다.

| 설정                   |    기본값 | 설명              |
| -------------------- | -----: | --------------- |
| Camera index         |    `0` | 사용할 웹캠 번호       |
| Frame width          | `1280` | 입력 영상 너비        |
| Frame height         |  `720` | 입력 영상 높이        |
| Maximum poses        |    `4` | 동시에 검출할 최대 사람 수 |
| Maximum hands        |    `8` | 동시에 검출할 최대 손 수  |
| Detection confidence |  `0.5` | 최소 검출 신뢰도       |
| Tracking confidence  |  `0.5` | 최소 추적 신뢰도       |
| Presence confidence  |  `0.5` | 최소 랜드마크 존재 신뢰도  |

## Configuration

### Camera 변경

기본 카메라는 `0`번으로 설정되어 있습니다.

```python
cap = cv2.VideoCapture(0)
```

외장 웹캠 등 다른 카메라를 사용하려면 값을 `1` 또는 `2`로 변경합니다.

```python
cap = cv2.VideoCapture(1)
```

### Resolution 변경

기본 해상도는 1280×720입니다.

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

실행 속도가 느린 경우 해상도를 640×480으로 낮출 수 있습니다.

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Maximum Detection Count 변경

동시에 검출할 사람과 손의 최대 개수를 변경할 수 있습니다.

```python
num_poses=4
num_hands=8
```

검출 개수를 줄이면 시스템에 따라 처리 속도가 향상될 수 있습니다.

## Dependencies

`requirements.txt`에는 다음 외부 라이브러리가 포함됩니다.

```text
mediapipe
opencv-python
```

다음 모듈은 Python 표준 라이브러리이므로 별도로 설치하지 않습니다.

* `os`
* `time`
* `urllib.request`

## Troubleshooting

### Python 명령어를 찾을 수 없는 경우

Python이 설치되어 있는지 확인합니다.

Windows에서 Python을 설치할 때 다음 옵션을 선택해야 합니다.

```text
Add python.exe to PATH
```

설치가 끝난 후 기존 터미널을 닫고 새 터미널을 실행합니다.

### `No module named 'cv2'`

다음 명령어를 실행합니다.

```bash
python -m pip install opencv-python
```

### `No module named 'mediapipe'`

다음 명령어를 실행합니다.

```bash
python -m pip install mediapipe
```

### Webcam을 열 수 없는 경우

다음 항목을 확인합니다.

* 다른 프로그램에서 웹캠을 사용하고 있는지 확인
* 운영체제의 카메라 접근 권한 확인
* `cv2.VideoCapture(0)`의 카메라 번호 변경
* 외장 웹캠 연결 상태 확인

Windows에서는 다음 위치에서 카메라 권한을 확인할 수 있습니다.

```text
설정 → 개인정보 및 보안 → 카메라
```

### Model 다운로드에 실패하는 경우

최초 실행 시 모델 파일을 다운로드하려면 인터넷 연결이 필요합니다.

프로젝트 폴더에 크기가 0KB인 `.task` 파일이 생성된 경우 해당 파일을 삭제하고 프로그램을 다시 실행합니다.

### 프로그램이 느리게 실행되는 경우

다음 방법을 사용할 수 있습니다.

* 영상 해상도를 640×480으로 변경
* `num_poses` 값 감소
* `num_hands` 값 감소
* 다른 카메라 및 백그라운드 프로그램 종료

## Model Information

이 프로젝트는 MediaPipe에서 제공하는 다음 모델을 사용합니다.

* Pose Landmarker Full
* Hand Landmarker

모델은 프로그램 최초 실행 시 MediaPipe 모델 저장소에서 자동으로 다운로드됩니다.

## Notes

* 카메라 영상은 로컬 컴퓨터에서 처리됩니다.
* 검출 정확도와 실행 속도는 카메라 품질, 조명, 사람과 카메라 사이의 거리 및 컴퓨터 성능에 따라 달라질 수 있습니다.
* 여러 사람과 여러 손을 동시에 처리하면 단일 대상 처리보다 더 많은 연산이 필요합니다.

## References

* [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/tasks)
* [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
* [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
* [OpenCV](https://opencv.org/)
