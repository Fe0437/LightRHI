# LightRHI

LightRHI is a thin C++23 module-based rendering hardware interface. It exposes a
backend-neutral `rhi` API and selects one platform backend at CMake configure
time: Metal on Apple platforms, Vulkan elsewhere.

The project is bindless-first and Slang-first. GPU resources are represented by
small value handles, shaders address resources through bindless slots or GPU
addresses, and command recording stays explicit: resource transitions,
submission, and synchronization are caller-visible.

## Start here

- [ARCHITECTURE.md](ARCHITECTURE.md) explains the module layout, CMake targets,
  backend selection, bindless model, shader pipeline, and build requirements.
- [API_GUIDELINES.md](API_GUIDELINES.md) captures the public API rules used by
  the code in `source/rhi`.
- [AGENTS.md](AGENTS.md) contains local development constraints for contributors
  and coding agents working in this repository.

## Repository layout

```text
source/rhi/             Public backend-neutral C++23 module partitions
source/backend_metal/   Metal implementation and lightRHI module export
source/backend_vulkan/  Vulkan implementation and lightRHI module export
examples/               Small executable examples
tests/                  Smoke and GPU integration tests
cmake/                  Build helpers for warnings, sanitizers, Slang, shaders
tools/                  Shader artifact tooling
```

## Consumer usage

Consumers import the backend-facing module and link both CMake targets:

```cpp
import lightRHI;

auto device = rhi::createDevice({
    .EnableValidation = true,
    .AppName = "MyApp",
});
```

```cmake
target_link_libraries(my_app PRIVATE LightRHI::LightRHI LightRHI::Backend)
```

## Build

Use CMake 3.28+ with Ninja and a compiler that supports C++23 module dependency
scanning. On macOS, export `SDKROOT` before configure and build so
`clang-scan-deps` can find SDK headers.

```bash
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
ctest --test-dir build --output-on-failure
```

Install `slangc` to build shader-using examples and tests. See
[ARCHITECTURE.md](ARCHITECTURE.md#shader-pipeline) for the current shader
artifact pipeline.
