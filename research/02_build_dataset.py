"""
02_build_dataset.py
라벨 이미지(빨간칠)에서 마스크 역추출 → YOLO-seg 학습 포맷 변환
"""
import cv2, numpy as np, glob, os, re, random, shutil, argparse

random.seed(42)
CLASSES = {"inside": 0, "top": 1, "side": 2}


def get_mask(lab_img):
    """빨간칠 라벨 이미지에서 바이너리 마스크 추출."""
    hsv = cv2.cvtColor(lab_img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    k = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return filled


def collect_files(folder, prefix):
    """폴더 내 파일에서 prefix+숫자 패턴 매칭."""
    d = {}
    for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if "__MACOSX" in p or os.path.basename(p).startswith("."):
            continue
        if not os.path.isfile(p):
            continue
        m = re.search(prefix + r"(\d+)", os.path.basename(p))
        if m:
            d[int(m.group(1))] = p
    return d


def build(data_root, output, val_ratio=0.2):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output, "labels", split), exist_ok=True)

    pairs = []
    for name in CLASSES:
        lab_folder = os.path.join(data_root, name)
        ori_folder = os.path.join(data_root, f"{name}_origin")
        if not os.path.isdir(lab_folder) or not os.path.isdir(ori_folder):
            print(f"  [건너뜀] {name}: 폴더 없음")
            continue
        labs = collect_files(lab_folder, name)
        oris = collect_files(ori_folder, f"{name}_origin")
        common = sorted(set(labs) & set(oris))
        print(f"  {name}: label={len(labs)}, origin={len(oris)}, paired={len(common)}")
        for i in common:
            pairs.append((name, labs[i], oris[i]))

    random.shuffle(pairs)
    n_val = max(3, int(len(pairs) * val_ratio))

    stats = {"train": 0, "val": 0, "skip": 0}
    for idx, (name, lp, op) in enumerate(pairs):
        split = "val" if idx < n_val else "train"
        o = cv2.imread(op)
        l = cv2.imread(lp)
        if o is None or l is None:
            stats["skip"] += 1
            continue
        m = get_mask(l)
        m = cv2.resize(m, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            stats["skip"] += 1
            continue
        c = max(cnts, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            stats["skip"] += 1
            continue
        H, W = o.shape[:2]
        coords = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in approx)
        stem = f"{name}_{idx:04d}"
        cv2.imwrite(os.path.join(output, "images", split, f"{stem}.png"), o)
        with open(os.path.join(output, "labels", split, f"{stem}.txt"), "w") as f:
            f.write(f"{CLASSES[name]} {coords}\n")
        stats[split] += 1

    # data.yaml
    abs_out = os.path.abspath(output)
    with open(os.path.join(output, "data.yaml"), "w") as f:
        f.write(f"path: {abs_out}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {list(CLASSES.keys())}\n")

    print(f"\n결과: {stats}")
    print(f"data.yaml → {os.path.join(output, 'data.yaml')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output", default="./dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()
    build(args.data_root, args.output, args.val_ratio)
