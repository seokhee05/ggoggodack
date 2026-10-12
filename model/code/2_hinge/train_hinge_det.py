"""
힌지 검출 모델 학습 스크립트 (내 컴퓨터에서 실행)

준비:
  1) 바탕 화면\\라벨링\\label_round1 안에 images 폴더(사진)와 labels 폴더(txt)가 있어야 함
     (labels_yolo.zip 을 label_round1 안에서 "여기에 풀기")
  2) pip install ultralytics  (venv312 에 이미 있으면 생략)

실행:
  python train_hinge_det.py

결과:
  hinge_det_runs\\round2\\weights\\best.pt   <- 이 파일이 학습된 모델
  hinge_det_runs\\round2\\results.png        <- 학습 그래프
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ====== 환경에 맞게 수정 ======
SRC = Path(r"C:\Users\User\OneDrive\바탕 화면\라벨링\label_round1")   # images, labels 가 있는 폴더
WORK = Path(r"C:\Users\User\OneDrive\바탕 화면\라벨링\hinge_det_split")  # 학습용으로 나눠 담을 폴더(자동 생성)
VAL_VIDEOS = {"crop001", "crop005", "crop007"}   # 검증용 영상 (학습에는 안 씀)
# 추가 학습 데이터: (사진 폴더, 라벨 폴더). 라벨 txt 가 있는 사진만 학습(train)에 넣음
BASE = SRC.parent
EXTRA_SOURCES = [
    (BASE / "blue label original", BASE / "blue_label_yolo" / "labels"),   # 안쪽면 라벨의 파란 힌지 -> bbox 변환분
]
EPOCHS = 70
IMG_SIZE = 640
# ============================


def find_label_dir():
    """labels 폴더 위치 자동 찾기 (압축 푸는 방식에 따라 위치가 달라서)"""
    for cand in [SRC / "labels", SRC / "labels_yolo" / "labels", SRC / "labels_yolo"]:
        if cand.is_dir() and any(cand.glob("*.txt")):
            return cand
    raise SystemExit(f"라벨 txt 폴더를 못 찾았습니다. {SRC} 안에 labels 폴더가 있는지 확인하세요.")


def build_split():
    """사진을 영상 단위로 train / val 로 나눠 복사 (같은 영상이 양쪽에 섞이지 않게)"""
    label_dir = find_label_dir()
    print(f"라벨 폴더: {label_dir}")
    if WORK.exists():
        shutil.rmtree(WORK)
    for sp in ("train", "val"):
        (WORK / "images" / sp).mkdir(parents=True)
        (WORK / "labels" / sp).mkdir(parents=True)

    n = {"train": 0, "val": 0}
    for img in sorted((SRC / "images").glob("*.jpg")):
        video = img.stem.split("_")[0]           # crop006_t020.50 -> crop006
        sp = "val" if video in VAL_VIDEOS else "train"
        shutil.copy(img, WORK / "images" / sp / img.name)
        lbl = label_dir / (img.stem + ".txt")
        dst = WORK / "labels" / sp / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dst)
        else:
            dst.write_text("")                   # 라벨 없음 = 힌지 없는 사진
        n[sp] += 1

    for img_dir, lbl_dir in EXTRA_SOURCES:
        if not (img_dir.is_dir() and lbl_dir.is_dir()):
            print(f"[안내] 추가 데이터 폴더가 없어 건너뜀: {img_dir}")
            continue
        added = 0
        for lbl in sorted(lbl_dir.glob("*.txt")):
            img = img_dir / (lbl.stem + ".jpg")
            if not img.exists():
                continue
            shutil.copy(img, WORK / "images" / "train" / img.name)
            shutil.copy(lbl, WORK / "labels" / "train" / lbl.name)
            added += 1
        n["train"] += added
        print(f"추가 데이터 {img_dir.name}: {added}장")

    n_box = sum(1 for f in (WORK / "labels").rglob("*.txt") for l in f.read_text().splitlines() if l.strip())
    print(f"힌지 네모 {n_box}개")
    if n_box == 0:
        raise SystemExit("네모가 0개입니다. 라벨 파일이 비어 있지 않은지 확인하세요.")

    yaml = WORK / "data.yaml"
    yaml.write_text(
        f"path: {WORK.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: hinge\n",
        encoding="utf-8",
    )
    print(f"학습 {n['train']}장 / 검증 {n['val']}장")
    return yaml


def main():
    yaml = build_split()
    model = YOLO("yolov8n.pt")                   # 처음 실행 시 자동 다운로드
    model.train(
        data=str(yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        device="cpu",
        patience=20,
        fliplr=0.5,        # 좌우 반전 (작업자마다 힌지 방향이 다름)
        scale=0.8,         # 크게/작게 확대 (crop19·20처럼 확대된 영상 대응)
        translate=0.25,    # 위치 이동 (힌지가 화면 끝에서 잘리는 경우 대응)
        degrees=10,        # 약간 회전
        project=str(Path(__file__).resolve().parent / "hinge_det_runs"),   # 라벨링 폴더 안에 저장
        name="round2",
        exist_ok=True,
        plots=True,
    )
    print("\n학습 완료! hinge_det_runs/round2/weights/best.pt 가 모델 파일입니다.")
    print("results.csv 마지막 줄의 metrics/mAP50(B) 값이 0.6 이상이면 쓸 만합니다.")


if __name__ == "__main__":
    main()
