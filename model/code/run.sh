#!/bin/bash
# ============================================================
# 꼬꼬댁 실행 스크립트 — python3.11 고정
# 사용법:
#   bash run.sh crop017.mp4         # 영상 지정
#   bash run.sh                     # 기본 crop020.mp4
# ============================================================
PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/run_v9_integrated.py"

# mediapipe Metal 충돌 방지 환경변수
export MEDIAPIPE_DISABLE_GPU=1
export TF_FORCE_GPU_ALLOW_GROWTH=false

if [ -n "$1" ]; then
    "$PY" "$SCRIPT" "$1"
else
    "$PY" "$SCRIPT"
fi
