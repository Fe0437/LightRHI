#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Configuring ==="
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
    -DCMAKE_OSX_SYSROOT="${SDKROOT}" \
    -DLIGHT_RHI_BUILD_TESTS=ON \
    -DLIGHT_RHI_BUILD_EXAMPLES=OFF

echo ""
echo "=== Building ==="
cmake --build build --parallel

echo ""
echo "=== Running tests ==="
./build/tests/rhi_tests
