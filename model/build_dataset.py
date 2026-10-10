import cv2, numpy as np, glob, os, random

random.seed(42)
ROOT = "/home/claude/seg_data"
OUT = "/home/claude/yolo_seg"
CLASSES = {"inside": 0, "top": 1}

def get_mask(lab):
    hsv = cv2.cvtColor(lab, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    k = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # 큰 커널로 내부 구멍 메움
    # 최대 컨투어만 남기고 내부 채움
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    big = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [big], -1, 255, -1)
    return filled

# 폴더 구조 생성
for split in ["train", "val"]:
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)

pairs = []
for name in CLASSES:
    for lp in sorted(glob.glob(f"{ROOT}/clean/{name}/label/*.png")):
        op = lp.replace("/label/", "/origin/")
        if os.path.exists(op):
            pairs.append((name, lp, op))

random.shuffle(pairs)
n_val = max(1, int(len(pairs) * 0.2))
val_set = set(range(n_val))

stats = {"train": 0, "val": 0, "skipped": 0}
for idx, (name, lp, op) in enumerate(pairs):
    split = "val" if idx in val_set else "train"
    o = cv2.imread(op)
    l = cv2.imread(lp)
    m = get_mask(l)
    m = cv2.resize(m, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_NEAREST)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        stats["skipped"] += 1
        continue
    c = max(cnts, key=cv2.contourArea)
    # 폴리곤 단순화
    eps = 0.002 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        stats["skipped"] += 1
        continue

    H, W = o.shape[:2]
    # YOLO-seg: class x1 y1 x2 y2 ... (정규화)
    coords = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in approx)
    stem = f"{name}{idx:03d}"
    cv2.imwrite(f"{OUT}/images/{split}/{stem}.png", o)
    with open(f"{OUT}/labels/{split}/{stem}.txt", "w") as f:
        f.write(f"{CLASSES[name]} {coords}\n")
    stats[split] += 1

# data.yaml
with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n")
    f.write(f"nc: {len(CLASSES)}\nnames: {list(CLASSES.keys())}\n")

print("생성 결과:", stats)
print("data.yaml 작성 완료")
