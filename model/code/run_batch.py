"""
run_batch.py
crop001~017 영상을 순차 처리. 전체 진행률/예상 남은 시간 표시.
"""
import os, subprocess, time
from run_v3 import run_on_video

VIDEO_DIR = "/Users/songminseo/Desktop/kkokkodack/data/media/cycle_mp4"
OUT_DIR = "/Users/songminseo/Desktop/kkokkodack/model/results"
os.makedirs(OUT_DIR, exist_ok=True)

targets = [f"crop{str(i).zfill(3)}.mp4" for i in range(1, 18)]

summary = []
start_all = time.time()
total_n = len(targets)

for i, fname in enumerate(targets, 1):
    src = os.path.join(VIDEO_DIR, fname)
    if not os.path.exists(src):
        print(f"[{i}/{total_n}] 건너뜀: {fname} 없음")
        continue

    out = os.path.join(OUT_DIR, f"result_{fname}")

    elapsed_so_far = time.time() - start_all
    avg_per_video = elapsed_so_far / (i - 1) if i > 1 else None
    eta_str = f"(예상 전체 남은시간: {avg_per_video * (total_n - i + 1) / 60:.1f}분)" if avg_per_video else ""

    print(f"\n=== [{i}/{total_n}] {fname} 처리 시작 {eta_str} ===")
    t0 = time.time()
    completed, warns = run_on_video(src, out)
    elapsed = time.time() - t0
    print(f"=== [{i}/{total_n}] {fname} 완료 ({elapsed:.1f}초, 부품 {completed}개, 경고 {len(warns)}건) ===")
    summary.append((fname, elapsed, completed, len(warns)))

print(f"\n\n전체 완료! 총 소요시간: {(time.time()-start_all)/60:.1f}분")
print(f"결과 저장 위치: {OUT_DIR}\n")
print(f"{'파일':15s} {'소요시간':>10s} {'부품수':>8s} {'경고':>6s}")
for fname, t, completed, nwarn in summary:
    print(f"{fname:15s} {t:>9.1f}s {completed:>8d} {nwarn:>6d}")

subprocess.run(["open", OUT_DIR])