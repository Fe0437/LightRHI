# LightRHI Justfile
# Standalone build + test commands. Cross-platform: macOS / Linux / Windows.
# When LightRHI is used as a submodule, build is driven by the parent project.
#
# Prerequisites:
#   - CMake 3.28+  (Ninja generator required for C++23 module scanning)
#   - A Clang with C++23 module support and clang-scan-deps:
#       macOS   — Homebrew LLVM (brew install llvm; AppleClang lacks clang-scan-deps)
#       Linux   — clang / clang++ on PATH
#       Windows — LLVM/Clang on PATH (e.g. from the LLVM installer)
#   - Vulkan is provisioned automatically: a system SDK is used if present,
#     otherwise headers are fetched and the driver's loader is used via volk.
#   - Optional: VCPKG_ROOT pointing to a vcpkg checkout (used as the toolchain
#     file when set; not required).
#
# Usage:
#   just build          # configure + build (RelWithDebInfo)
#   just test           # build + run all tests
#   just test-smoke     # run only the unit tests (no GPU needed)
#   just build-asan     # ASAN + UBSAN build
#   just test-asan      # build-asan + run tests
#   just clean          # remove build directories

# just defaults to `sh`, which isn't present on a stock Windows. Use PowerShell
# there; POSIX shells everywhere else.
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

BUILD_DIR  := "build"
BUILD_TYPE := env_var_or_default("BUILD_TYPE", "RelWithDebInfo")
VCPKG_ROOT := env_var_or_default("VCPKG_ROOT", "")

_toolchain_flag := if VCPKG_ROOT != "" {
    "-DCMAKE_TOOLCHAIN_FILE=" + VCPKG_ROOT + "/scripts/buildsystems/vcpkg.cmake"
} else {
    ""
}

# Compiler selection is platform-specific. macOS must use Homebrew LLVM (and the
# SDK sysroot); Linux/Windows use whatever clang/clang++ is on PATH. Recipes
# below are single logical `cmake` lines (via `\` continuation), so these flags
# splice in cleanly regardless of the shell running them.
_compiler_flags := if os() == "macos" {
    "-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang " + \
    "-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ " + \
    "-DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)"
} else {
    "-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
}

# Filter used by `info` to surface just the "[LightRHI] ..." status lines.
_grep_lightrhi := if os() == "windows" {
    "Select-String -Pattern '\\[LightRHI\\]'"
} else {
    "grep '\\[LightRHI\\]' || true"
}

# Show the available standalone build and test recipes.
default:
    @just --list

# ── Configure ────────────────────────────────────────────────────────────────

# Configure the default standalone build directory with examples and tests.
_configure:
    cmake -B {{BUILD_DIR}} \
        -G Ninja \
        {{_toolchain_flag}} \
        -DCMAKE_BUILD_TYPE={{BUILD_TYPE}} \
        {{_compiler_flags}} \
        -DLIGHT_RHI_BUILD_EXAMPLES=ON \
        -DLIGHT_RHI_BUILD_TESTS=ON

# ── Build ─────────────────────────────────────────────────────────────────────

# Configure and build the default standalone tree.
build: _configure
    cmake --build {{BUILD_DIR}} --parallel

# Configure and build an ASAN/UBSAN test tree.
build-asan:
    cmake -B {{BUILD_DIR}}-asan \
        -G Ninja \
        {{_toolchain_flag}} \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        {{_compiler_flags}} \
        -DLIGHT_RHI_BUILD_TESTS=ON \
        -DLIGHT_RHI_BUILD_EXAMPLES=OFF \
        -DLIGHT_RHI_ENABLE_ASAN=ON \
        -DLIGHT_RHI_ENABLE_UBSAN=ON
    cmake --build {{BUILD_DIR}}-asan --parallel

# ── Test ──────────────────────────────────────────────────────────────────────

# Run all tests (unit + integration; integration requires a real GPU)
test: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure -V

# Run CPU-only smoke tests for handles, bitmasks, and format helpers.
test-smoke: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure -V -L smoke

# Run with AddressSanitizer
test-asan: build-asan
    ctest --test-dir {{BUILD_DIR}}-asan --output-on-failure -V

# ── Utilities ─────────────────────────────────────────────────────────────────

# Remove generated build directories used by the recipes.
# `cmake -E rm -rf` is portable — no rm / Remove-Item shell differences.
clean:
    cmake -E rm -rf {{BUILD_DIR}} {{BUILD_DIR}}-asan {{BUILD_DIR}}-tsan

# Print resolved backend for this platform
info:
    -@cmake -B {{BUILD_DIR}}-info {{_toolchain_flag}} -DLIGHT_RHI_BUILD_EXAMPLES=OFF -DLIGHT_RHI_BUILD_TESTS=OFF 2>&1 | {{_grep_lightrhi}}
    @cmake -E rm -rf {{BUILD_DIR}}-info
