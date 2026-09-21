# LightRHI

LightRHI is a thin C++23 module-based rendering hardware interface. It exposes a
backend-neutral `rhi` API and selects one platform backend at CMake configure
time: Metal on Apple platforms, Vulkan elsewhere.

The project is bindless-first and Slang-first. GPU resources are represented by
small value handles, shaders address resources through bindless slots or GPU
addresses, and command recording stays explicit: resource transitions,
submission, and synchronization are caller-visible.

## Start here

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the module layout, CMake targets,
  backend selection, bindless model, shader pipeline, and build requirements.
- [API_GUIDELINES.md](docs/API_GUIDELINES.md) captures the public API rules used by
  the code in `source/rhi`.

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

## Dependencies

LightRHI resolves every dependency itself. Each is tried with `find_package()`
first, so a vcpkg or system install is used when present, and fetched into
`_deps/` otherwise. A parent project that adds LightRHI as a submodule does not
need to install or declare any of these.

| Dependency | Version | When | Why |
| --- | --- | --- | --- |
| Metal, Foundation, QuartzCore | system SDK | Apple | Frameworks the Metal backend links. |
| [metal-cpp](https://github.com/bkaradzic/metal-cpp) | `metal-cpp_27` | Apple | Header-only C++ wrapper for Metal; no Objective-C is compiled. |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | `vulkan-sdk-1.3.290.0` | non-Apple | Headers only — the loader is never linked (see below). |
| [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `v3.1.0` | non-Apple | Device memory suballocation for the Vulkan backend. |
| [volk](https://github.com/zeux/volk) | `1.3.270` | non-Apple | Loads Vulkan entry points at run time. |
| [Slang](https://github.com/shader-slang/slang) | `2026.13` | shaders | `slangc` compiles the `.slang` sources; a prebuilt release is fetched only when no usable `slangc` is on the system. |
| Python 3 | interpreter | shaders | Runs the shader artifact tooling in `tools/`. |

Only one backend's dependencies are resolved per build: `LIGHT_RHI_BACKEND`
selects Metal on Apple and Vulkan elsewhere, and the other backend's sources are
not configured.

The Vulkan backend needs no SDK install. It resolves entry points through volk
against the loader that ships with every GPU driver, so `vulkan-1.lib` /
`libvulkan.so` is never linked and only the headers are required.

### Build-time tools

| Tool | Minimum | Why |
| --- | --- | --- |
| CMake | 3.28 | C++23 module dependency scanning. |
| Ninja | any | The only generator that drives module scanning here. |
| Clang | 21 | C++23 named modules, built with `-fno-exceptions` and `-fno-rtti`. |

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
[ARCHITECTURE.md](docs/ARCHITECTURE.md#shader-pipeline) for the current shader
artifact pipeline.
