# Barcode-Based Inspection Action Tracking

OpenCV와 MediaPipe Tasks를 이용해 웹캠 영상을 저장하고, 작업자가 표준 검사 절차를 빠짐없이 수행했는지 분석하는 프로토타입입니다.

제품 바코드를 입력하면 녹화가 시작됩니다. 검사가 끝난 뒤 저장 영상을 분석하여 Knob, Hinge, Screw, Damper, Rib·이음부 및 Lining–Skin Gap 검사 수행 여부와 순서를 `PASS`, `FAIL`, `REVIEW`로 기록합니다.

> 현재 버전은 작업자의 **검사 행동·순서·누락 여부**를 확인합니다. Screw 누락, Rib 파손, 실제 Gap 수치와 같은 제품 불량 자체를 판정하는 모델은 포함하지 않습니다.

## Features

- 바코드 기반 검사 세션 생성
- 바코드 입력 후 웹캠 녹화 시작
- 원본 검사 영상 MP4 저장
- 최초 실행 시 부품별 ROI 설정
- 최대 4명의 전신 포즈 추적
- 최대 8개의 손 및 손가락 랜드마크 추적
- 저장 영상 자동 분석
- 검사 순서 및 누락 항목 확인
- 항목별 `PASS / FAIL / REVIEW` 판정
- 랜드마크와 ROI가 표시된 분석 영상 저장
- 검사 결과 JSON 저장
- MediaPipe 모델 자동 다운로드

## Inspection Sequence

프로토타입의 기본 검사 순서는 다음과 같습니다.

```text
Knob
→ Hinge
→ Screw 1
→ Screw 2
→ Damper 1
→ Damper 2
→ Rib·이음부
→ Knob 재작동
→ Hinge 재작동
→ Lining–Skin Gap
```

손 중심점이 현재 검사 ROI 안에 `0.8초` 동안 연속으로 머무르면 해당 검사 동작을 완료한 것으로 처리합니다.

## Tech Stack

| 구분 | 사용 기술 |
|---|---|
| Language | Python 3.11 |
| Computer Vision | OpenCV |
| Pose Estimation | MediaPipe Pose Landmarker |
| Hand Tracking | MediaPipe Hand Landmarker |
| Input | Webcam, Barcode Scanner |
| Output | MP4, JSON |

## Quick Start for Windows

### 1. 프로젝트 다운로드

GitHub 저장소에서 `Code → Download ZIP`을 누른 후 압축을 해제합니다.

Git이 이미 설치되어 있다면 다음 명령으로 내려받을 수도 있습니다.

```bat
git clone https://github.com/seokhee05/ggoggodack.git
cd ggoggodack
```

### 2. 전체 개발 환경 자동 설치

프로젝트 폴더에서 PowerShell을 열고 다음 명령을 실행합니다.

```powershell
.\setup.bat
```

명령 프롬프트(CMD)를 사용하는 경우:

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

PowerShell에서 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\test.py
```

## First Run: ROI Setup

`roi_config.json`이 없는 최초 실행에서는 다음 검사영역을 순서대로 지정합니다.

1. Knob
2. Hinge
3. Screw 1
4. Screw 2
5. Damper 1
6. Damper 2
7. Rib / Joint
8. Lining-Skin Gap

각 검사영역을 마우스로 드래그한 뒤 `Enter` 또는 `Space`를 누릅니다. 설정 결과는 `roi_config.json`에 저장되고 이후 실행부터 자동으로 재사용됩니다.

ROI를 다시 설정하려면 프로그램을 종료한 뒤 `roi_config.json`을 삭제하고 다시 실행합니다.

## Usage

### 1. 바코드 입력

ROI 설정이 완료되면 터미널에 다음 메시지가 표시됩니다.

```text
제품 바코드:
```

USB 바코드 스캐너로 제품 바코드를 스캔하거나 키보드로 직접 입력하고 `Enter`를 누릅니다.

### 2. 검사 영상 녹화

바코드 입력과 동시에 녹화가 시작됩니다. 작업자가 표준 검사 절차를 수행한 뒤 영상 창을 클릭하고 `Q`를 누르면 녹화가 종료됩니다.

### 3. 저장 영상 자동 분석

녹화가 끝나면 저장된 영상 분석이 자동으로 시작됩니다.

```text
저장 영상 입력
→ 포즈·손 랜드마크 추출
→ 검사영역 접근 확인
→ 검사 순서 확인
→ 누락 항목 확인
→ PASS / FAIL / REVIEW 저장
```

분석 창에서도 `Q`를 누르면 분석을 중단할 수 있습니다.

## Result Status

| 상태 | 의미 |
|---|---|
| `PASS` | 정해진 순서로 검사영역에 접근하고 검사 동작을 완료함 |
| `FAIL` | 검사 동작을 누락했거나 정해진 순서를 완료하지 못함 |
| `REVIEW` | 손 검출률이 낮아 영상만으로 검사 수행 여부를 판단하기 어려움 |

결과 예시:

```text
제품: PRODUCT_001

[PASS] Knob 검사
[PASS] Hinge 검사
[PASS] Screw 1 검사
[FAIL] Screw 2 검사
[FAIL] Damper 1 검사
[FAIL] Damper 2 검사
[FAIL] Rib·이음부 검사
[FAIL] Knob 재작동 검사
[FAIL] Hinge 재작동 검사
[FAIL] Lining–Skin Gap 검사

검사 순서: FAIL
최종 판정: FAIL
```

## Output Files

실행 후 다음 파일이 생성됩니다.

```text
recordings/
├── PRODUCT_001_20260729_153010_raw.mp4
└── PRODUCT_001_20260729_153010_analyzed.mp4

results/
└── PRODUCT_001_20260729_153010_result.json
```

| 파일 | 설명 |
|---|---|
| `*_raw.mp4` | 웹캠에서 저장한 원본 검사 영상 |
| `*_analyzed.mp4` | 랜드마크, ROI 및 진행 상태가 표시된 분석 영상 |
| `*_result.json` | 바코드, 검사 시간, 항목별 결과, 누락 항목 및 최종 판정 |

`recordings/`, `results/`, `roi_config.json` 및 MediaPipe 모델은 `.gitignore`에 등록되어 GitHub에 업로드되지 않습니다.

## Project Structure

```text
ggoggodack/
├── test.py                       # 녹화 및 검사 행동 분석 프로그램
├── requirements.txt              # Python 라이브러리 목록
├── setup.bat                     # Windows 개발 환경 자동 설치
├── README.md                     # 프로젝트 설명 및 실행 방법
├── .gitignore                    # Git 제외 설정
├── roi_config.json               # 최초 실행 후 생성되는 ROI 설정
├── pose_landmarker_full.task     # 자동 다운로드되는 포즈 모델
├── hand_landmarker.task          # 자동 다운로드되는 손 모델
├── recordings/                   # 원본 및 분석 영상
└── results/                      # JSON 검사 결과
```

## How It Works

1. MediaPipe 모델을 확인하고 없으면 자동으로 다운로드합니다.
2. 웹캠을 연결합니다.
3. 저장된 ROI 설정을 불러오거나 최초 ROI 설정을 진행합니다.
4. 바코드 번호를 제품 ID로 입력받습니다.
5. 바코드 입력과 동시에 원본 영상 녹화를 시작합니다.
6. `Q` 입력 시 녹화를 종료합니다.
7. 저장 영상을 프레임 단위로 읽습니다.
8. Pose Landmarker와 Hand Landmarker로 랜드마크를 검출합니다.
9. 손 중심점이 현재 검사 ROI에 연속으로 머문 시간을 계산합니다.
10. 검사 순서와 누락 항목을 판정합니다.
11. 분석 영상과 JSON 결과를 저장합니다.

## Default Configuration

| 설정 | 기본값 | 설명 |
|---|---:|---|
| Camera index | `0` | 사용할 웹캠 번호 |
| Frame width | `1280` | 입력 영상 너비 |
| Frame height | `720` | 입력 영상 높이 |
| Fallback FPS | `30` | 카메라 FPS를 읽지 못할 때 사용하는 값 |
| ROI dwell time | `0.8초` | 검사를 완료한 것으로 인정하는 연속 체류 시간 |
| Minimum hand detection ratio | `0.30` | 영상 품질을 PASS로 판단하는 최소 손 검출 비율 |
| Maximum poses | `4` | 동시에 검출할 최대 사람 수 |
| Maximum hands | `8` | 동시에 검출할 최대 손 수 |
| Detection confidence | `0.5` | 최소 검출 신뢰도 |
| Tracking confidence | `0.5` | 최소 추적 신뢰도 |
| Presence confidence | `0.5` | 최소 랜드마크 존재 신뢰도 |

설정값은 `test.py` 상단에서 변경할 수 있습니다.

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DWELL_SECONDS = 0.8
MIN_HAND_DETECTION_RATIO = 0.30
```

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

## Dependencies

`requirements.txt`에는 다음 외부 Python 라이브러리가 포함됩니다.

```text
mediapipe
opencv-python
```

`json`, `os`, `re`, `time`, `urllib.request`, `datetime`는 Python 표준 라이브러리이므로 별도로 설치하지 않습니다.

## Troubleshooting

### `winget`을 찾을 수 없는 경우

Microsoft Store에서 **App Installer**를 설치한 뒤 새 터미널에서 `.\setup.bat`을 다시 실행합니다.

### PowerShell에서 `setup.bat`을 찾을 수 없는 경우

현재 폴더의 파일을 실행하려면 앞에 `.\`를 붙여야 합니다.

```powershell
.\setup.bat
```

### Python 라이브러리를 찾을 수 없는 경우

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 웹캠이 열리지 않는 경우

- 다른 프로그램이 웹캠을 사용 중인지 확인합니다.
- Windows의 `설정 → 개인정보 및 보안 → 카메라`에서 접근을 허용합니다.
- `test.py`의 `CAMERA_INDEX`를 `1` 또는 `2`로 변경합니다.

### ROI 위치가 맞지 않는 경우

프로그램을 종료한 뒤 `roi_config.json`을 삭제하고 다시 실행하여 ROI를 재설정합니다.

### 모델 다운로드에 실패하는 경우

인터넷 연결과 방화벽 설정을 확인합니다. 크기가 0KB인 `.task` 파일이 생성되었다면 삭제한 후 다시 실행합니다.

### 분석 결과가 계속 FAIL인 경우

- 체크리스트 순서를 정확하게 지켰는지 확인합니다.
- 손 중심점이 노란색 활성 ROI 안에 0.8초 이상 머물렀는지 확인합니다.
- ROI가 실제 부품 위치와 일치하는지 확인합니다.
- 카메라 초점, 조명 및 가림 상태를 확인합니다.

## Prototype Limitations

- ROI 접근과 체류 시간을 기준으로 검사 행동을 판단합니다.
- Knob과 Hinge가 실제로 정상 작동하는지는 판정하지 않습니다.
- Screw 및 Damper의 누락·마모·파손을 판정하지 않습니다.
- Rib의 균열·파손 및 이음부 불량을 판정하지 않습니다.
- Lining–Skin Gap의 실제 길이를 측정하지 않습니다.
- 실제 제품 불량 판정에는 별도의 객체 검출, 이상 탐지 또는 치수 측정 모델이 필요합니다.

## References

- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/tasks)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV](https://opencv.org/)
