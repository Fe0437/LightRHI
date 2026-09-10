# LightRHI Architecture

## Overview

LightRHI is a thin, bindless GPU abstraction layer written in [C++23 modules](https://en.cppreference.com/w/cpp/language/modules). It targets one backend per platform — Metal on macOS, Vulkan on Linux/Windows — selected automatically at CMake configure time with no user-facing knob.

The design philosophy is **bindless-first**: every resource (buffer, texture, sampler) is accessed by a slot index that shaders read directly, with no per-draw descriptor binding. Push constants carry all per-draw root data. This maps cleanly to Metal Argument Buffers Tier 2 and Vulkan descriptor indexing.

## CMake targets

| Target              | Type   | Alias                  | Purpose                                   |
| ------------------- | ------ | ---------------------- | ----------------------------------------- |
| `LightRHI`        | SHARED | `LightRHI::LightRHI` | Backend-neutral interface (module`rhi`) |
| `LightRHIBackend` | SHARED | `LightRHI::Backend`  | Platform backend (module`lightRHI`)     |

Consumers link both:

```cmake
target_link_libraries(my_app PRIVATE LightRHI::LightRHI LightRHI::Backend)
```

Two targets exist because `LightRHI` (the interface) can be used as a submodule without pulling in the backend, and because the backend shared library carries a platform dependency (Metal frameworks or Vulkan loader) that the interface does not.

## Module structure

> C++23 modules replace `#include` with explicit import declarations. A *module unit* (`.cppm`) declares `export module name;` and explicitly marks what it exports. A *module partition* (e.g. `export module rhi:types;`) is a sub-unit of the same top-level module, allowing the interface to be split across files without being visible as separate imports to consumers.
> Reference: [P1779R3 — Standard library modules](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p1779r3.html) · [cppreference: Modules](https://en.cppreference.com/w/cpp/language/modules)

### `rhi` — the public interface

A primary module with nine partitions, all re-exported from the root unit:

```
rhi
├── :types        — Format, TextureDimension, QueueType, ResourceState, enums
├── :handles      — BufferHandle, TextureHandle, SamplerHandle, PipelineHandle, FenceHandle
├── :sync         — TextureBarrier, BufferBarrier, MemoryBarrier, SubresourceRange
├── :descriptors  — DeviceDesc, BufferDesc, TextureDesc, SamplerDesc, MappedBuffer
├── :pipeline     — ShaderDesc, GraphicsPipelineDesc, ComputePipelineDesc, RenderingDesc
├── :resources    — TextureView, BufferInfo, DrawIndirect/Dispatch arg structs
├── :bindless     — IBindlessHeap, BindlessLimits
├── :commandList  — ICommandList (pure virtual)
└── :device       — IDevice (pure virtual), SubmitDesc
```

The `rhi` partitions are an implementation detail — consumers never import them directly.

### `lightRHI` — the consumer-facing module

A single import that re-exports the entire `rhi` interface plus the platform factory:

```cpp
import lightRHI;

auto device = rhi::createDevice({ .EnableValidation = true });
rhi::BufferHandle buf = device->createBuffer({...});
```

`lightRHI` is implemented by `LightRHIBackend` (one shared library per platform). It:

- `export import rhi;` — re-exports all nine `rhi` partitions to the consumer
- Exports `rhi::createDevice(const DeviceDesc&) → unique_ptr<IDevice>`
- Implements all pure-virtual interfaces from `rhi`
- Keeps internal types in `rhi::metal::` / `rhi::vulkan::` (not exported)

## Bindless resource model

> **Bindless rendering** means shaders access any resource in a large heap using a plain integer index rather than binding individual descriptors before each draw. This enables GPU-driven rendering (the GPU decides what to draw without CPU intervention) and removes per-draw descriptor management overhead.
>
> - Metal: [Argument Buffers](https://developer.apple.com/documentation/metal/buffers/about_argument_buffers) — specifically [Tier 2](https://developer.apple.com/documentation/metal/mtlargumentbufferstier), which allows accessing resources across encoder boundaries and from any shader stage.
> - Vulkan: [VK_EXT_descriptor_indexing](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_descriptor_indexing.html) (promoted to Vulkan 1.2 core) — enables `nonuniformEXT` indexing into unbounded descriptor arrays.

All resources are identified by slot indices that are valid in shaders without explicit descriptor binding:

- `BufferHandle.Index` — slot in the buffer argument buffer
- `TextureHandle.Index` — slot in the texture argument buffer
- `SamplerHandle.Index` — slot in the sampler argument buffer

Shaders receive handles through push constants (root constants). The `IBindlessHeap` manages slot allocation and lifetime. On resource creation, the backend registers the resource's GPU address or descriptor into the heap automatically — no explicit registration call needed.

Metal implementation uses Argument Buffers Tier 2. Vulkan uses `VK_EXT_descriptor_indexing`.

## IDevice API surface

`IDevice` is the central object — one per physical GPU. It owns all resources and all command submission.

Resource lifecycle:

- `createBuffer / destroyBuffer`
- `createTexture / destroyTexture`
- `createSampler / destroySampler`
- `createGraphicsPipeline / createComputePipeline / destroyPipeline`

Mapping and addressing:

- `mapBuffer / unmapBuffer` — CPU access for `CpuToGpu` / `GpuToCpu` memory types
- `bufferAddress` — returns a `GpuAddress` (BDA / argument buffer address) for use as push constant

> **Buffer Device Address (BDA)** is the ability to hold a GPU-side pointer to a buffer as a plain 64-bit integer, pass it through push constants or another buffer, and dereference it in a shader without binding the buffer explicitly. In Metal this is the native memory model (all GPU resources have a 64-bit address on Apple Silicon). In Vulkan it requires `VK_KHR_buffer_device_address` (Vulkan 1.2 core).
>
> - Vulkan spec: [VK_KHR_buffer_device_address](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_buffer_device_address.html)
> - Metal: [MTLBuffer.gpuAddress](https://developer.apple.com/documentation/metal/mtlbuffer/gpuaddress)

Upload helpers (staging + submit + wait, for offline loading):

- `uploadBuffer`
- `uploadTexture`

> These internally create a `CpuToGpu` staging buffer, record a copy command, submit on the transfer queue, and block the CPU until complete. For streaming uploads in a render loop, use a ring buffer instead.

Submission and sync:

- `createCommandList → ICommandList`
- `submit(*cmd) → FenceHandle`
- `waitForFence / isFenceComplete`
- `waitIdle`

> LightRHI uses a **timeline semaphore** model for synchronisation. Each `submit` returns a monotonically increasing `FenceHandle` (backed by `MTLSharedEvent` on Metal and `VkSemaphore` with timeline semantics on Vulkan).
>
> - Metal: [MTLSharedEvent](https://developer.apple.com/documentation/metal/mtlsharedevent)
> - Vulkan: [VK_KHR_timeline_semaphore](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_timeline_semaphore.html) (Vulkan 1.2 core)

## ICommandList API surface

Commands are recorded into a `ICommandList` (backed by `MTLCommandBuffer` on Metal, `VkCommandBuffer` on Vulkan) then submitted atomically.

Barriers: `transition(TextureBarrier|BufferBarrier|MemoryBarrier)`, `flushBarriers`

> Barriers communicate resource state transitions to the GPU driver, enabling correct cache invalidation and pipeline hazard tracking. LightRHI's `ResourceState` enum maps to `MTLBarrierScope` / `MTLRenderStages` on Metal and `VkPipelineStageFlags2` + `VkAccessFlags2` on Vulkan (the Synchronisation 2 API).
>
> - Vulkan: [VK_KHR_synchronization2](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html) (Vulkan 1.3 core)
> - Metal: [Resource Synchronisation](https://developer.apple.com/documentation/metal/resource_synchronization)

Render pass: `beginRendering / endRendering`, `setViewport`, `setScissor`

> LightRHI uses **dynamic rendering** — no explicit render pass objects or framebuffer objects. Attachments are described inline at `beginRendering`. This maps to `VK_KHR_dynamic_rendering` (Vulkan 1.3 core) on Vulkan and directly to Metal's `MTLRenderCommandEncoder` model.
>
> - Vulkan: [VK_KHR_dynamic_rendering](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_dynamic_rendering.html)
> - Metal: [MTLRenderCommandEncoder](https://developer.apple.com/documentation/metal/mtlrendercommandencoder)

Draw: `draw`, `drawIndexed`, `drawIndirect`, `drawIndexedIndirect`, `drawIndirectCount`

> `drawIndirect` / `drawIndexedIndirect` read draw arguments (vertex count, instance count, etc.) from a GPU buffer, allowing the GPU to generate draw calls without CPU readback — the foundation of **GPU-driven rendering**.
>
> - Vulkan: [vkCmdDrawIndirect](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDrawIndirect.html)
> - Metal: [drawPrimitives:indirectBuffer:indirectBufferOffset:](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/drawprimitives_indirectbuffer_indirectbufferoffset_)

Compute: `dispatch`, `dispatchIndirect`

Resource ops: `copyBuffer`, `copyTexture`, `copyBufferToTexture`, `copyTextureToBuffer`, `clearTexture`, `clearDepthTexture`, `fillBuffer`

Debug: `beginDebugGroup / endDebugGroup / insertDebugLabel`

> Debug groups appear as labelled sections in GPU capture tools (Xcode GPU Frame Capture, RenderDoc). They have no effect outside of a capture.
>
> - Xcode: [GPU Frame Capture](https://developer.apple.com/documentation/metal/debugging_tools/capturing_gpu_command_data_programmatically)
> - RenderDoc: [renderdoc.org](https://renderdoc.org)

Push constants: `setPushConstants(const void* data, uint32_t size)`

> **Push constants** (Vulkan terminology; Metal calls them *inline constant data*) are small amounts of data (≤128 bytes, guaranteed by Vulkan) written directly into the command stream and accessible in any shader stage without a descriptor. LightRHI always binds them at buffer slot 30 on Metal, which leaves slots 0–29 available for resource tables.
>
> - Vulkan: [Push Constants](https://registry.khronos.org/vulkan/specs/latest/chapters/pipelines.html#vkCmdPushConstants)
> - Metal: [setVertexBytes:length:atIndex:](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/setvertexbytes_length_atindex_)

## Platform backend selection

```cmake
if(APPLE)
    set(LIGHT_RHI_BACKEND "Metal")
else()
    set(LIGHT_RHI_BACKEND "Vulkan")
endif()
```

The same `LightRHIBackend` target name and `LightRHI::Backend` alias are used on all platforms. No `#ifdef` is needed in consumer code.

## Shader pipeline

Shaders are authored in [Slang](https://shader-slang.com) and compiled at build time by `slangc`.

> **Slang** is a shading language and compiler developed at NVIDIA Research that transpiles a single source to SPIR-V, MSL, HLSL, GLSL, CUDA, and others. It adds modern language features (generics, interfaces, modules) on top of an HLSL-compatible syntax.
>
> - Language reference: [shader-slang.com/docs](https://shader-slang.com/slang/user-guide/)
> - GitHub: [github.com/shader-slang/slang](https://github.com/shader-slang/slang)
> - Targets reference: [Compilation Targets](https://shader-slang.com/slang/user-guide/compiling.html#compilation-targets)

```
shader.slang  ──slangc -target metal──►  shader.metal (MSL text)
              ──slangc -target spirv──►  shader.spv   (SPIR-V binary)
```

The CMake helper `light_rhi_compile_shaders(...)` (defined in `cmake/Shaders.cmake`) reads a `shaders_registry.generated.json` manifest and invokes `tools/compile_shaders.py` once per shader entry. Outputs are backend-specific artifact files:

```
<output-dir>/metal/<name>.generated.metal
<output-dir>/vulkan/<name>.generated.spv
```

The RHI core does not invoke Slang and does not embed shader source into C++. Runtime or test code loads artifact bytes and converts them to `ShaderDesc` through `ShaderArtifactView` and `toShaderDesc()`. For Metal development builds, `ShaderDesc::MslSource` is populated with MSL text and the backend JIT-compiles it via [`MTLDevice::newLibraryWithSource`](https://developer.apple.com/documentation/metal/mtldevice/newlibrary_with_source_options_completionhandler_). For Vulkan, `ShaderDesc::Spirv` is populated with SPIR-V words.

> For production use, pre-compiling to `.metallib` with `xcrun metal` + `xcrun metallib` avoids the runtime MSL compilation cost. The `ShaderDesc::Metallib` field accepts pre-compiled `.metallib` bytes for this purpose.

### Binding layout

All shaders use the same explicit Metal buffer slot assignment so the backend can bind resources without reflection:

| Slot | Content                  | Slang annotation                                     |
| ---- | ------------------------ | ---------------------------------------------------- |
| 0    | Buffer GPU-address table | `[[vk::binding(0, 0)]] StructuredBuffer<uint64_t>` |
| 1    | Texture slot table       | `[[vk::binding(1, 0)]] StructuredBuffer<uint64_t>` |
| 2    | Sampler slot table       | `[[vk::binding(2, 0)]] StructuredBuffer<uint64_t>` |
| 30   | Push constants           | `[[vk::push_constant]] ConstantBuffer<PC>`         |

`[[vk::push_constant]]` compiles to `[[buffer(30)]]` in Metal MSL output, matching the `kPushConstantSlot = 30` constant in the Metal command list encoder.

> `[[vk::push_constant]]` is a Slang/HLSL semantic attribute that maps to `layout(push_constant)` in GLSL/SPIR-V and to `[[buffer(30)]]` in Slang's Metal codegen. The `[[vk::binding(N, set)]]` attribute sets explicit descriptor binding numbers; for Metal, Slang uses the binding number directly as the `[[buffer(N)]]` index.
> Reference: [Slang — Resource Binding](https://shader-slang.com/slang/user-guide/a1-04-semantic-attributes.html)

For compute shaders the `[numthreads(X, Y, Z)]` annotation emits `[[max_total_threads_per_threadgroup(X)]]` in Metal, which the backend reads from the PSO via [`maxTotalThreadsPerThreadgroup()`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup).

> Metal compute shaders do not embed the thread group size in the shader binary — the host specifies it at dispatch time. Slang's `[numthreads]` annotation encodes the intended thread group size into the Metal PSO's `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup` metadata, which the backend queries automatically.

### Installing slangc

```
brew install shader-slang   # macOS
# or download from https://github.com/shader-slang/slang/releases
```

`cmake/Slang.cmake` searches `/opt/homebrew/bin`, `/usr/local/bin`, and `$SLANG_DIR/bin`. If not found, `LIGHT_RHI_SLANG_AVAILABLE` is set to `OFF` and shader compilation targets are skipped.

## Build requirements

- [CMake 3.28+](https://cmake.org/cmake/help/latest/release/3.28.html) — required for C++23 module dependency scanning (`CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API`)
- [Ninja](https://ninja-build.org) generator — the only CMake generator with reliable C++23 module support as of 2025 ([CMake module docs](https://cmake.org/cmake/help/latest/manual/cmake-cxxmodules.7.html))
- Clang with module support — on macOS, [Homebrew LLVM](https://formulae.brew.sh/formula/llvm) (`brew install llvm`) provides a recent enough Clang; the Apple-bundled Clang lacks `clang-scan-deps`
- `slangc` — Slang shader compiler ([install above](#installing-slangc))
- macOS: `SDKROOT` must be exported before both configure and build steps so `clang-scan-deps` can find system headers (`objc/runtime.h` etc.)

```bash
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ...
cmake --build build
```

## Sanitizers

[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html), [UBSan](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html), and [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) are opt-in per build:

```
cmake -DLIGHT_RHI_ENABLE_ASAN=ON -DLIGHT_RHI_ENABLE_UBSAN=ON ...
```

Sanitizer flags are applied with `PUBLIC` link visibility on both `LightRHI` and `LightRHIBackend` so they propagate to consuming test and example executables.

> Sanitizer flags must be on the **link** options of the static/shared library with `PUBLIC` visibility, not just `PRIVATE`, because the sanitizer runtime is linked into the final executable — not the intermediate library. Using `PRIVATE` on a static library means the sanitizer flags appear in the `.a`'s object files but the runtime symbols (`__asan_init`, `__ubsan_handle_*`) are never requested by the linker when linking the executable.
