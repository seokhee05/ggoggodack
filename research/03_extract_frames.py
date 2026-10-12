"""
03_extract_frames.py
crop001~016 영상에서 프레임 추출 (라벨링 보강용)
crop017~020은 테스트용이라 제외
"""
import os, glob, subprocess, argparse


def extract(video_dir, output, fps=2, train_only=True):
    os.makedirs(output, exist_ok=True)

    # train용 영상만 (crop001~016)
    videos = sorted(glob.glob(os.path.join(video_dir, "crop*.*")))
    for vp in videos:
        name = os.path.splitext(os.path.basename(vp))[0]  # crop001
        num = int("".join(filter(str.isdigit, name)) or "999")

        if train_only and num >= 17:
            print(f"  [건너뜀] {name} (테스트용)")
            continue

        out_dir = os.path.join(output, name)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-i", vp,
            "-vf", f"fps={fps}",
            os.path.join(out_dir, "f_%04d.png"),
        ]
        print(f"  {name} → {out_dir} ({fps}fps)")
        subprocess.run(cmd, capture_output=True)

    print(f"\n추출 완료. 각 영상 폴더에서 핵심 프레임을 골라 라벨링하세요.")
    print(f"라벨링 후 data/ 아래 side/inside/top + origin 폴더에 추가하고")
    print(f"02_build_dataset.py를 다시 실행하면 자동 병합됩니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="./data/cycle")
    parser.add_argument("--output", default="./augment_frames")
    parser.add_argument("--fps", type=float, default=2)
    args = parser.parse_args()
    extract(args.video_dir, args.output, args.fps)
