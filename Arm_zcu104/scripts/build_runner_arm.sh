#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ARM_DIR="${PROJECT_ROOT}/Arm_zcu104"
BUILD_DIR="${ARM_DIR}/build-runner-arm"
INSTALL_DIR="${ARM_DIR}/install-executorch-zcu104-v3"
TOOLCHAIN="${ARM_DIR}/cmake/toolchain-zcu104.cmake"

if [[ ! -f "${TOOLCHAIN}" ]]; then
  echo "ERRO: toolchain ausente: ${TOOLCHAIN}" >&2
  exit 1
fi
if [[ ! -f "${INSTALL_DIR}/lib/cmake/ExecuTorch/executorch-config.cmake" ]]; then
  echo "ERRO: runtime ExecuTorch ainda nao foi instalado em ${INSTALL_DIR}" >&2
  exit 1
fi

cmake \
  -S "${ARM_DIR}/runner" \
  -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_INSTALL_DIR="${INSTALL_DIR}"

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

cp "${BUILD_DIR}/hyperstarcop_arm_executorch" \
  "${ARM_DIR}/runtime/hyperstarcop_arm_executorch"

file "${ARM_DIR}/runtime/hyperstarcop_arm_executorch"
sha256sum "${ARM_DIR}/runtime/hyperstarcop_arm_executorch"
