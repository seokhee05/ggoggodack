"""
run_batch.py
crop001~017 영상을 순차적으로 추론 처리.
각 결과는 result_cropXXX.mp4로 저장되고, 끝나면 결과 폴더를 자동으로 엶.
"""
import os, subprocess, time
from run_v3 import run_on_video

VIDEO_DIR = "/Users/songminseo/Desktop/kkokkodack/data/media/cycle_mp4"
OUT_DIR = "/Users/songminseo/Desktop/kkokkodack/results"
os.makedirs(OUT_DIR, exist_ok=True)

# 처리할 영상 번호 (001~017)
targets = [f"crop{str(i).zfill(3)}.mp4" for i in range(1, 18)]

summary = []
start_all = time.time()

for fname in targets:
    src = os.path.join(VIDEO_DIR, fname)
    if not os.path.exists(src):
        print(f"[건너뜀] {fname} 없음")
        continue

    out = os.path.join(OUT_DIR, f"result_{fname}")
    print(f"\n=== {fname} 처리 시작 ===")
    t0 = time.time()
    run_on_video(src, out)
    elapsed = time.time() - t0
    print(f"=== {fname} 완료 ({elapsed:.1f}초) ===")
    summary.append((fname, elapsed))

print(f"\n\n전체 완료! 총 소요시간: {(time.time()-start_all)/60:.1f}분")
print(f"결과 저장 위치: {OUT_DIR}")
for fname, t in summary:
    print(f"  {fname}: {t:.1f}초")

# 결과 폴더 자동으로 Finder에서 열기
subprocess.run(["open", OUT_DIR])