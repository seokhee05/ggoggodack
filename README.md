# MediaPipe Pose & Hand Tracking

웹캠 영상을 이용하여 사람의 전신 포즈와 손가락 관절을 실시간으로 검출하고 화면에 표시하는 Python 프로그램입니다.

## 주요 기능

* 웹캠 영상 실시간 입력
* 최대 4명의 전신 포즈 검출
* 사람당 33개의 포즈 랜드마크 표시
* 최대 8개의 손 검출
* 손당 21개의 손가락 랜드마크 표시
* 신체 부위와 손가락별 색상 구분
* 감지된 사람과 손 개수 표시
* 필요한 MediaPipe 모델 자동 다운로드

## 개발 환경

* 개발 언어: Python
* 권장 Python 버전: Python 3.11
* 지원 Python 버전: Python 3.9~3.12
* 주요 라이브러리:

  * MediaPipe
  * OpenCV
* 입력 장치: 웹캠
* 지원 운영체제:

  * Windows
  * macOS
  * Linux

## 프로젝트 파일 구성

GitHub 저장소에는 다음 파일이 포함되어 있습니다.

```text
project/
├── test.py
├── requirements.txt
└── README.md
```

프로그램을 처음 실행하면 다음 모델 파일이 자동으로 다운로드됩니다.

```text
project/
├── test.py
├── requirements.txt
├── README.md
├── pose_landmarker_full.task
└── hand_landmarker.task
```

## 설치 방법

### 1. Python 설치

Python이 설치되어 있지 않다면 아래 공식 사이트에서 Python을 설치합니다.

* https://www.python.org/downloads/

Python 3.11 버전을 권장합니다.

Windows에서 Python을 설치할 때 반드시 다음 항목을 체크합니다.

```text
Add python.exe to PATH
```

### 2. 프로젝트 다운로드

GitHub 저장소 상단의 다음 메뉴를 선택합니다.

```text
Code → Download ZIP
```

다운로드한 ZIP 파일의 압축을 해제합니다.

Git이 설치되어 있다면 다음 명령어로 저장소를 받을 수도 있습니다.

```bash
git clone 저장소주소
```

예시:

```bash
git clone https://github.com/사용자이름/저장소이름.git
```

그다음 프로젝트 폴더로 이동합니다.

```bash
cd 저장소이름
```

### 3. 터미널 열기

`test.py`와 `requirements.txt`가 들어 있는 프로젝트 폴더에서 터미널을 엽니다.

VS Code를 사용하는 경우:

```text
터미널 → 새 터미널
```

Windows에서는 프로젝트 폴더의 빈 공간에서 마우스 오른쪽 버튼을 누른 후 다음을 선택해도 됩니다.

```text
터미널에서 열기
```

### 4. 필요한 라이브러리 설치

프로젝트 폴더에서 다음 명령어를 실행합니다.

```bash
python -m pip install -r requirements.txt
```

Windows에서 `python` 명령어가 작동하지 않는 경우 다음 명령어를 사용합니다.

```bash
py -m pip install -r requirements.txt
```

## 프로그램 실행

라이브러리 설치가 완료되면 다음 명령어를 실행합니다.

```bash
python test.py
```

Windows에서 `python` 명령어가 작동하지 않는 경우:

```bash
py test.py
```

처음 실행할 때 다음 MediaPipe 모델 파일이 자동으로 다운로드됩니다.

* `pose_landmarker_full.task`
* `hand_landmarker.task`

따라서 최초 실행 시에는 인터넷 연결이 필요합니다.

## 사용 방법

프로그램을 실행하면 웹캠 화면이 열리고 다음 정보가 표시됩니다.

* 사람의 전신 관절
* 손과 손가락 관절
* 감지된 사람 수
* 감지된 손 수

프로그램을 종료하려면 웹캠 영상 창을 클릭한 후 키보드의 `Q` 키를 누릅니다.

## 가상환경 사용 방법

다른 Python 프로젝트와 라이브러리가 충돌하는 것을 방지하려면 가상환경 사용을 권장합니다.

### Windows

가상환경을 생성합니다.

```bash
py -3.11 -m venv .venv
```

가상환경을 활성화합니다.

```bash
.venv\Scripts\activate
```

필요한 라이브러리를 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

프로그램을 실행합니다.

```bash
python test.py
```

### macOS 및 Linux

가상환경을 생성합니다.

```bash
python3 -m venv .venv
```

가상환경을 활성화합니다.

```bash
source .venv/bin/activate
```

필요한 라이브러리를 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

프로그램을 실행합니다.

```bash
python test.py
```

## 오류 해결

### `python` 명령어를 찾을 수 없는 경우

Python이 설치되어 있는지 확인하고, Windows에서는 설치할 때 다음 옵션을 체크했는지 확인합니다.

```text
Add python.exe to PATH
```

Python 설치 후에는 기존 터미널을 닫고 새로 열어야 합니다.

### `No module named 'cv2'` 오류

다음 명령어를 실행합니다.

```bash
python -m pip install opencv-python
```

### `No module named 'mediapipe'` 오류

다음 명령어를 실행합니다.

```bash
python -m pip install mediapipe
```

### 웹캠 화면이 열리지 않는 경우

다른 프로그램이 웹캠을 사용하고 있는지 확인합니다.

Windows에서는 다음 설정을 확인합니다.

```text
설정 → 개인정보 및 보안 → 카메라
```

Python 또는 데스크톱 앱의 카메라 접근을 허용해야 합니다.

카메라 번호가 다른 경우 `test.py`의 다음 코드를 확인합니다.

```python
cap = cv2.VideoCapture(0)
```

`0`을 `1` 또는 `2`로 변경하여 다시 실행합니다.

```python
cap = cv2.VideoCapture(1)
```

### 모델 다운로드에 실패하는 경우

인터넷 연결과 방화벽 설정을 확인합니다.

프로젝트 폴더에 크기가 0KB인 `.task` 파일이 생성되었다면 해당 파일을 삭제한 후 프로그램을 다시 실행합니다.

## requirements.txt

이 프로젝트에서 사용하는 외부 Python 라이브러리는 다음과 같습니다.

```text
mediapipe
opencv-python
```

`os`, `time`, `urllib.request`는 Python에 기본으로 포함된 표준 라이브러리이므로 별도로 설치하지 않습니다.
