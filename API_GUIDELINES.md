# API Guidelines

Keep the public API small, backend neutral, and data oriented. The API should
describe GPU work without exposing Vulkan, Metal, or platform-specific lifetime
rules.

## Target the latest platform API version — always

Backends always target the newest available graphics API version (currently
Metal 4 / latest Vulkan with descriptor buffers and bindless extensions).
Never write a fallback path, a version check, or an `#if` branch to keep an
older API version working alongside the new one.

This is deliberate, not an oversight: LightRHI minimizes maintenance surface
by carrying exactly one implementation per backend. Supporting an older OS
release or an older GPU generation means carrying a second code path
indefinitely — that ongoing maintenance cost is explicitly rejected in favor
of a single, current implementation. Concretely:

- Metal backend targets Metal 4 (`MTL4CommandQueue`/`MTL4CommandBuffer`/
  `MTL4Compiler`/`MTL4ArgumentTable`/`MTLResidencySet`), which requires
  macOS 26 / iOS 26 as the minimum deployment target. Do not add classic-Metal
  (`MTLCommandQueue`/`MTLArgumentEncoder`) fallback paths for older OS
  versions — devices/OSes that don't support Metal 4 are simply unsupported.
- Vulkan backend targets the latest available extensions (descriptor
  buffers, bindless descriptor indexing, etc.) rather than the oldest
  common denominator. Do not gate new Vulkan usage behind extension
  availability checks with a fallback; if the extension is unavailable,
  the device is unsupported.
- When a newer API version changes how something works (e.g. Metal 4
  replacing per-encoder `useResources()` residency with `MTLResidencySet`),
  migrate the implementation outright rather than keeping both mechanisms
  side by side.

## Resource identity

Resources are represented by opaque index handles.

Good:

```cpp
BufferHandle
TextureHandle
PipelineHandle
```

Bad:

```cpp
Buffer*
Texture*
Pipeline*
```

Handles are value types. A default-constructed handle is invalid and must be
cheap to copy, compare, and pass through push constants when appropriate.

## Public shape

Prefer value types and descriptor structs for creation parameters:

```cpp
auto buffer = device->CreateBuffer({
    .Size = 1024,
    .Usage = BufferUsage::Storage | BufferUsage::TransferDst,
});
```

Do not forward-declare ordinary classes or structs; include the defining
header or restructure the implementation so every named dependency is
complete where it is declared. A private nested PIMPL declaration (`struct
Impl;`) is the sole exception.

Avoid inheritance unless it is needed for a backend-neutral interface boundary.
Current interface boundaries are `IDevice`, `ICommandList`, and `IBindlessHeap`.

Do not expose backend objects, backend enums, or backend handles from the public
`rhi` module. Consumers should be able to `import lightRHI;` and stay portable.

## Errors and ownership

Prefer `std::expected` for recoverable public API errors.

No exceptions in public API contracts. Backend implementations may use internal
helpers, but errors that cross the public API boundary should be explicit return
values or validity checks.

Use RAII internally, but public GPU resources remain explicit handles with
matching destroy calls until a concrete ownership wrapper is needed.

## Naming

### Summary table

| Category | Convention | Example |
|---|---|---|
| Types, structs, enums | `PascalCase` | `BufferDesc`, `LoadOp` |
| Public functions, methods, public properties | `PascalCase` | `CreateBuffer`, `AdapterName` |
| Descriptor / struct fields | `PascalCase` | `.Size`, `.Usage`, `.Bytecode` |
| Private member variables | `_camelCase` | `_device`, `_allocator` |
| Private member functions | `_camelCase` | `_makeShaderModule`, `_loadLibrary` |
| Local / parameter variables | `lowerCamelCase` | `vertexCount`, `dstOffset` |

### Public API examples

```cpp
// IDevice
auto buffer  = device->CreateBuffer({.Size = 4096, .Usage = BufferUsage::Storage});
auto name    = device->AdapterName();
auto fence   = device->Submit(*cmd);
device->WaitForFence(fence);

// ICommandList
cmd->Begin();
cmd->BeginRendering({.Color = {{.Texture = rt}}});
cmd->SetPipeline(pso);
cmd->SetPushConstants(MyConstants{...});
cmd->Draw(3);
cmd->EndRendering();
cmd->End();

// IBindlessHeap
uint32_t maxBufs = heap.MaxBuffers();
GpuAddress addr  = heap.HeapAddress();
```

### Descriptor fields

All struct fields used in designated-initializer call sites are `PascalCase`:

```cpp
BufferDesc desc{
    .Size  = 4096,
    .Usage = BufferUsage::Storage,
};
```

### Private members

Private data members and private helper functions use `_camelCase` (underscore
prefix + lower camelCase):

```cpp
class MetalDevice final : public IDevice {
    NS::SharedPtr<MTL::Device> _device;    // private data
    MTL::Library* _loadLibrary(const ShaderDesc& sd); // private helper
};
```

### Local variables and parameters

All local variables and function parameters use `lowerCamelCase`:

```cpp
void UploadData(BufferHandle dstBuf, uint64_t byteCount) {
    auto stagingBuffer = CreateBuffer({.Size = byteCount});
    uint32_t rowPitch  = ComputeRowPitch(format, width);
}
```

## Initialization

Use uniform initialization with braces for variable initialization.

Good:

```cpp
int x{5};
std::vector<int> values{1, 2, 3};
BufferDesc desc{
    .Size = 4096,
    .Usage = BufferUsage::Storage,
};
```

Bad:

```cpp
int x = 5;
std::vector<int> values = {1, 2, 3};
BufferDesc desc = BufferDesc{.Size = 4096};
```
