#!/bin/bash
set -e

SRC="hyperstarcop_zcu104_optimized.cpp"
OUT="hyperstarcop_zcu104_optimized"
SWEEP_SRC="sweep_hyperstarcop_zcu104.cpp"
SWEEP_OUT="sweep_hyperstarcop_zcu104"

if pkg-config --exists opencv4; then
  OPENCV_PKG="opencv4"
elif pkg-config --exists opencv; then
  OPENCV_PKG="opencv"
else
  echo "ERROR: OpenCV pkg-config not found"
  exit 1
fi

echo "Building $SRC -> $OUT"

g++ \
  -O3 \
  -DNDEBUG \
  -std=c++17 \
  -Wall \
  -Wextra \
  "$SRC" \
  -o "$OUT" \
  $(pkg-config --cflags --libs "$OPENCV_PKG") \
  -lvart-runner \
  -lxir \
  -lpthread

echo
echo "Built:"
ls -lh "$OUT"

echo
echo "Building $SWEEP_SRC -> $SWEEP_OUT"

g++ \
  -O3 \
  -DNDEBUG \
  -std=c++17 \
  -Wall \
  -Wextra \
  "$SWEEP_SRC" \
  -o "$SWEEP_OUT"

echo
echo "Built:"
ls -lh "$SWEEP_OUT"

echo
echo "Example:"
echo "./$OUT --profile all --runners 2 --pre-workers 2 --post-workers 1 --slots-per-runner 3 --iterations 500 --warmup 20 --pin"

echo
echo "Automatic staged sweep (up to 16 concurrent workers/slots):"
echo "./$SWEEP_OUT --resume"
