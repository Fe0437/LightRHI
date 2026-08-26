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

## High cohesion, low coupling

### Prefer explicit contracts over runtime type inspection

Do not use `dynamic_cast` or `typeid` to recover behavior or data that is
missing from a base interface. Express required capabilities through the
stable interface, a tagged value or variant for a closed set of
representations, or composition for independently optional capabilities.
Runtime type inspection is reserved for external framework interop boundaries
that impose type-erased or base-typed objects; any such exception belongs in
architecture documentation rather than being justified locally.

Group code around one GPU abstraction, invariant, or resource lifecycle. The
operations and private data that jointly implement a command list, device,
pipeline, or resource table should stay together when they change for the same
reason. Do not create a header/source pair for every small preparation step or
descriptor transformation merely to shorten a file; that increases the module
dependency graph and makes one operation harder to follow.

Split code when the new component has an independent contract: a reusable pure
policy, a backend-neutral public abstraction, a backend implementation behind
that abstraction, or compile-time optional instrumentation. Implementation-only
helpers remain private to their cohesive module unless another real consumer
exists.

Dependencies point from volatile backend code toward the stable public `rhi`
contract, never from the public contract toward Metal or Vulkan. Likewise, a
shared interface exposes only what every implementation needs. A backend-only
object, mode flag, generic `extra` field, or nullable parameter understood by a
single backend is evidence that backend policy has leaked into the shared API;
move it behind the backend boundary or define a properly owned abstraction.

Use these checks when choosing a boundary:

- Can the component's responsibility be stated without joining unrelated jobs
  with “and”?
- Do its pieces change together under the same invariant and lifetime?
- Does the dependency direction keep platform-specific code below the public
  API?
- Does a proposed split remove coupling or create an independently useful
  contract? If it only moves lines, keep the code together.

## Errors and ownership

Prefer `std::expected` for recoverable public API errors.

No exceptions in public API contracts. Backend implementations may use internal
helpers, but errors that cross the public API boundary should be explicit return
values or validity checks.

Use RAII internally, but public GPU resources remain explicit handles with
matching destroy calls until a concrete ownership wrapper is needed.

## Public API documentation

Document every exported type, free function, and interface method with Doxygen
comments. Describe the contract from the caller's point of view: when to call
the operation, valid call order, ownership and lifetime, required resource
state or synchronization, and the meaning of results or unsupported behavior.
Use the backslash command form consistently: `\brief`, `\param`, `\return`,
`\pre`, and `\note` rather than the equivalent `@` commands.

Do not document backend machinery in the public declaration. Vulkan objects,
Metal encoders, allocation strategies, and implementation anecdotes belong in
the backend source. Public comments explain the stable LightRHI behavior that a
caller can rely on. Avoid comments that merely repeat a name; descriptor-field
comments should clarify units, defaults, optionality, or how the field affects
the operation that consumes the descriptor.

Function names are the first summary of a contract, but they do not need to
encode every important behavior. A non-trivial implementation should read as
orchestration of cohesive, specifically named operations, or carry Doxygen that
explains its caller-visible contract and non-obvious invariants or side effects.
Do both when an important operation still requires several distinct stages.
This applies to private backend functions too; omit documentation only when the
name, signature, and short body make the complete behavior immediately clear.
Do not manufacture readability by extracting vaguely named helpers—each
extracted operation must have one specific role.

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

## Settings, descriptors, and parameters

LightRHI uses the same vocabulary as its consumers, with descriptors forming
the public API boundary:

| Suffix | Meaning | LightRHI use |
|---|---|---|
| `Settings` | Persistent behavioral or policy choices retained across operations | Use only when an RHI component genuinely retains tunable policy; no current public type needs it |
| `Desc` | Complete declarative input consumed by one create, build, submit, render, or dispatch operation | `DeviceDesc`, `BufferDesc`, `ComputePipelineDesc`, `SubmitDesc`, `RenderingDesc` |
| `Params` | Domain values used by computation rather than resource lifetime or API policy | Reserve for shader/domain data; do not use it as a synonym for an API descriptor |
| `Configuration` | A named assembly of already-defined, interchangeable pieces used to instantiate one object when several compositions are valid | Prefer a descriptive alias or instance name that identifies the composition |

Do not use `Config` or `Configuration` as a generic synonym for an RHI
descriptor or retained policy. Those roles remain `Desc` and `Settings`.
Configuration is reserved for the narrower case where an object can be
instantiated from multiple arrangements of already-defined components, such as
a descriptive `using` alias that injects a particular family of template
arguments. Prefer the full `Configuration` name for a new public type; `config`
is acceptable as an instance name only when its type genuinely represents such
a composition. Otherwise use a precise noun such as `State`, `Limits`,
`Features`, or `Resources`.

Local names mirror the type: `desc`, `settings`, `params`, and—only for the
composition case above—`configuration` or `config`. Avoid `cfg` or calling every
argument bundle `params`; those names erase the lifetime distinction expressed
by the type.

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
