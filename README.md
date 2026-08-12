# 암레스트 실시간 검사 시스템

공장 카메라 영상을 항상 표시하고, 바코드 스캔을 기준으로 박스별 녹화와
검사 행동 판정을 동시에 수행하는 Windows용 프로토타입입니다.

## 최종 운영 흐름

```text
프로그램 실행 → 카메라 상시 표시 → 바코드 스캔
→ 바코드별 영상 녹화 + 실시간 검사
→ 순서 오류·검사 지연 경고
→ 7개 검사 완료 시 자동 저장
→ 다음 바코드 대기
```

작업자는 ROI 사각형을 지정하거나 `C`, `Q` 키를 누르지 않습니다.

## 공식 체크리스트

1. Knob 작동 이상 확인
2. Hinge 작동 이상 확인
3. Screw 외관·누락 확인 — 2Point, Driver
4. Damper 누락 확인 — 2Point
5. Rib 파손 조각 확인 — 흔들기 동작
6. Knob·Hinge 굽힘/펴기 확인
7. Lining–Skin Ass’y Gap 확인

소리 검사는 별도 마이크를 사용하지 않고 흔들기와 굽힘·펴기 행동 수행 여부로
대체합니다.

## 판정 방식

- MediaPipe Hand Landmarker로 양손 위치를 실시간 추적합니다.
- 손 궤적의 이동 범위, 방향 전환, 양손 거리 변화를 계산합니다.
- Screw 검사는 영상에서 주황색 Driver 신호를 함께 사용합니다.
- 2Point 항목은 서로 분리된 두 번의 행동 신호가 있어야 완료됩니다.
- 강한 Driver 신호가 잘못된 순서에 나오면 즉시 순서 오류를 경고합니다.
- 한 단계가 18초 이상 진행되지 않으면 누락 가능성을 경고합니다.
- 7개 항목 완료 시 영상과 결과를 자동 저장합니다.

> 현재 프로그램은 검사 행동 수행 여부를 판단하는 프로토타입입니다. 암레스트의
> 실제 파손, Gap의 실제 수치, Screw 체결 상태 등 제품 자체의 양불량을 판정하는
> 모델은 아닙니다. 현장 영상으로 동작 임계값을 조정하고 검증해야 합니다.

## Windows 설치

프로젝트 폴더에서 `setup.bat`을 실행합니다.

```bat
setup.bat
```

설치되는 항목:

- Python 3.11
- Git
- Visual Studio Code
- Python 가상환경 `.venv`
- OpenCV
- MediaPipe
- Pillow

## 실행

```bat
.venv\Scripts\python.exe factory_app.py
```

프로그램을 실행하면 카메라가 바로 표시됩니다. 입력창에 바코드를 직접 입력하거나
USB 바코드 스캐너로 스캔한 뒤 `Enter`를 누르면 검사가 시작됩니다.

이미 촬영한 영상은 화면 아래의 `저장 영상 검사` 버튼을 누르고 MP4 파일과
제품 번호를 선택하면 분석할 수 있습니다. 원본 영상은 변경하지 않으며 결과는
`results/제품번호_saved_result.json`에 저장됩니다. 노란 원은 장갑을 포함해
프레임 변화로 추적한 작업 동작 위치입니다.

## 저장 파일

바코드가 `ABC123`이면 다음 파일이 생성됩니다.

```text
recordings/ABC123.mp4
results/ABC123_result.json
```

동일한 바코드가 다시 입력되면 기존 파일을 보호하도록 날짜와 시간이 붙습니다.

결과 JSON에는 다음 항목이 저장됩니다.

- 바코드
- 검사 시작·종료 시각
- 총 검사 시간
- 7개 항목별 PASS/MISSING
- 항목별 완료 시점
- 순서 오류 및 누락 가능성 경고 내역
- 최종 PASS/NG
- 영상 경로

## 파일 구조

```text
ggoggodack/
├── factory_app.py          # 카메라·바코드·GUI·영상 저장
├── inspection_engine.py    # 7개 실시간 행동 판정 엔진
├── requirements.txt
├── setup.bat
├── README.md
├── recordings/
├── results/
└── models/
```

## 기본 설정

`factory_app.py` 상단:

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
```

`inspection_engine.py`에서 검사 평균시간과 경고 시간을 설정합니다.

```python
InspectionEngine(nominal_seconds=70.0, warning_seconds=18.0)
```

현장 테스트에서 동작이 너무 쉽게 또는 어렵게 감지되면
`inspection_engine.py`의 각 detector 임계값을 조정합니다.
