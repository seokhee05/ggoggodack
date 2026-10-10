"""
힌지 점 모델 + 박스 안 암레스트 모델을 한 번에 학습 (내 컴퓨터에서 실행)

실행 (라벨링 폴더에서):
  .\\venv312\\Scripts\\python.exe train_all.py

결과:
  hinge_pose_runs\\pose1\\weights\\best.pt      <- 힌지 점 3개(tip/pivot/body) 모델
  armrest_box_runs\\box1\\weights\\best.pt      <- 박스 안 암레스트 모델
"""
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# ---------- 설정 ----------
POSE_SRC = BASE / "label_round1" / "images"                       # 사진 + 점 라벨 JSON
ARM_SRC_CANDS = [BASE / "label_armrest" / "label_armrest" / "images",   # 1차 (60장)
                 BASE / "label_armrest" / "images",
                 BASE / "label_armrest2" / "images",                    # 2차 (180장)
                 BASE / "label_armrest2_a" / "label_armrest2" / "images",   # 2차 압축 a 를 푼 곳
                 BASE / "label_armrest2_b" / "label_armrest2" / "images"]   # 2차 압축 b 를 푼 곳
POSE_VAL = {"crop001", "crop005", "crop007"}                      # 검증용 영상 (학습에 안 씀)
ARM_VAL = {"crop005", "crop012", "crop20"}
KPT_NAMES = ["tip", "pivot", "body"]
POSE_EPOCHS = 100
ARM_EPOCHS = 80
RUN_POSE = False   # 힌지 점 모델은 이미 학습함 (다시 하려면 True)
RUN_ARM = True
# --------------------------


def video_of(stem):
    return stem.split("_")[0]


def box_of(shape, W, H):
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    x1, x2, y1, y2 = max(0, min(xs)), min(W, max(xs)), max(0, min(ys)), min(H, max(ys))
    return x1, y1, x2, y2


def make_dirs(root):
    if root.exists():
        shutil.rmtree(root)
    for sp in ("train", "val"):
        (root / "images" / sp).mkdir(parents=True)
        (root / "labels" / sp).mkdir(parents=True)


# ================= 1) 힌지 점 모델 =================
def build_pose():
    root = BASE / "hinge_pose_dataset"
    make_dirs(root)
    n = {"train": 0, "val": 0}
    n_kpt = 0
    for img in sorted(POSE_SRC.glob("*.jpg")):
        js = img.with_suffix(".json")
        sp = "val" if video_of(img.stem) in POSE_VAL else "train"
        lines = []
        if js.exists():
            d = json.load(open(js, encoding="utf-8"))
            W, H = d["imageWidth"], d["imageHeight"]
            boxes = [box_of(s, W, H) for s in d["shapes"] if s["shape_type"] == "rectangle" and s["label"] == "hinge"]
            pts = [s for s in d["shapes"] if s["shape_type"] == "point" and s["label"] in KPT_NAMES]
            kp = [dict() for _ in boxes]
            for s in pts:       # 점은 가장 가까운 힌지 네모에 붙임
                x, y = s["points"][0]
                if not boxes:
                    continue
                i = min(range(len(boxes)), key=lambda k: ((boxes[k][0] + boxes[k][2]) / 2 - x) ** 2 + ((boxes[k][1] + boxes[k][3]) / 2 - y) ** 2)
                kp[i][s["label"]] = (min(max(x, 0), W - 1), min(max(y, 0), H - 1))
            for (x1, y1, x2, y2), k in zip(boxes, kp):
                # 점이 네모 밖에 있으면 네모를 점까지 넓힘 (모델이 점을 네모 안에서 찾도록)
                for (px, py) in k.values():
                    x1, y1, x2, y2 = min(x1, px), min(y1, py), max(x2, px), max(y2, py)
                row = [0, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
                for name in KPT_NAMES:
                    if name in k:
                        row += [k[name][0] / W, k[name][1] / H, 2]
                        n_kpt += 1
                    else:
                        row += [0, 0, 0]      # 안 찍은 점 = 안 보이는 점
                lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row))
        shutil.copy(img, root / "images" / sp / img.name)
        (root / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(lines))
        n[sp] += 1
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
                    f"kpt_shape: [3, 3]\nflip_idx: [0, 1, 2]\nnames:\n  0: hinge\n", encoding="utf-8")
    print(f"[힌지 점] 학습 {n['train']}장 / 검증 {n['val']}장, 찍은 점 {n_kpt}개")
    return yaml


# ================= 2) 박스 안 암레스트 모델 =================
def build_arm():
    srcs = [c for c in ARM_SRC_CANDS if c.is_dir()]
    if not srcs:
        raise SystemExit("label_armrest 폴더를 못 찾았습니다.")
    print("[암레스트] 사진 폴더:", ", ".join(str(c) for c in srcs))
    root = BASE / "armrest_box_dataset"
    make_dirs(root)
    n = {"train": 0, "val": 0}
    n_box = 0
    for img in sorted(p for c in srcs for p in c.glob("*.jpg")):
        js = img.with_suffix(".json")
        sp = "val" if video_of(img.stem) in ARM_VAL else "train"
        lines = []
        if js.exists():
            d = json.load(open(js, encoding="utf-8"))
            W, H = d["imageWidth"], d["imageHeight"]
            for s in d["shapes"]:
                if s["shape_type"] == "rectangle" and s["label"] == "armrest":
                    x1, y1, x2, y2 = box_of(s, W, H)
                    lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                    n_box += 1
        shutil.copy(img, root / "images" / sp / img.name)
        (root / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(lines))
        n[sp] += 1
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: armrest\n", encoding="utf-8")
    print(f"[암레스트] 학습 {n['train']}장 / 검증 {n['val']}장, 네모 {n_box}개")
    return yaml


def main():
    if RUN_POSE:
        y = build_pose()
        YOLO("yolov8n-pose.pt").train(
            data=str(y), epochs=POSE_EPOCHS, imgsz=640, batch=8, device="cpu", patience=30,
            fliplr=0.5, scale=0.6, translate=0.2, degrees=10,
            project=str(BASE / "hinge_pose_runs"), name="pose1", exist_ok=True, plots=True)
        print("\n힌지 점 모델 완료: hinge_pose_runs/pose1/weights/best.pt")
    if RUN_ARM:
        y = build_arm()
        YOLO("yolov8n.pt").train(
            data=str(y), epochs=ARM_EPOCHS, imgsz=640, batch=8, device="cpu", patience=25,
            fliplr=0.5, scale=0.5, translate=0.1,
            project=str(BASE / "armrest_box_runs"), name="box2", exist_ok=True, plots=True)
        print("\n암레스트 모델 완료: armrest_box_runs/box2/weights/best.pt")
    print("\n두 모델 모두 끝! results.csv 마지막 줄 값을 알려주세요.")


if __name__ == "__main__":
    main()
