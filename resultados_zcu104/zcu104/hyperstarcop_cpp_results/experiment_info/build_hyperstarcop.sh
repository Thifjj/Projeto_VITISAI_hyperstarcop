#!/bin/bash
set -e

SRC="hyperstarcop_zcu104_benchmark.cpp"
OUT="hyperstarcop_benchmark"

echo "=== Checking compiler ==="
g++ --version | head -1

echo
echo "=== Checking VART/XIR headers ==="
test -f /usr/include/vart/runner.hpp || {
  echo "ERROR: /usr/include/vart/runner.hpp not found"
  echo "The target image may have runtime libraries but not development headers."
  exit 1
}

test -f /usr/include/xir/graph/graph.hpp || {
  echo "ERROR: /usr/include/xir/graph/graph.hpp not found"
  exit 1
}

echo
echo "=== OpenCV pkg-config ==="

if pkg-config --exists opencv4; then
  OPENCV_PKG="opencv4"
elif pkg-config --exists opencv; then
  OPENCV_PKG="opencv"
else
  echo "ERROR: OpenCV pkg-config package not found."
  exit 1
fi

echo "Using: $OPENCV_PKG"

echo
echo "=== Building ==="

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
echo "Run with:"
echo "./$OUT"
