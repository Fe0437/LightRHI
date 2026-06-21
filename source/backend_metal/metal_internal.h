// metal_internal.h — shared internal types for the LightRHI Metal backend.
//
// Include AFTER "import rhi;" in the including TU.
// Included by both metal_device.cpp and metal_command_list.cpp.
//
// Targets Metal 4 (MTL4CommandQueue/MTL4CommandBuffer/MTL4Compiler/
// MTL4ArgumentTable/MTLResidencySet) exclusively — see API_GUIDELINES.md's
// "Target the latest platform API version — always" section. This requires
// macOS 26 / iOS 26 as the minimum deployment target; there is no classic-
// Metal fallback path.

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
// NOTE: <Metal/MTL4AccelerationStructure.hpp> (for the concrete MTL4 AS
// descriptor/geometry types — Metal.hpp only forward-declares
// MTL4::AccelerationStructureDescriptor) must be included in each
// including TU's *global module fragment* (before "module lightRHI;"),
// not here — this header is included after the module declaration, and
// #include-ing a new header there would attach its declarations to the
// module purview, conflicting with the global-module forward declaration
// Metal.hpp already brought in.
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rhi::metal
{

    // ============================================================================
    // SlotPool
    // ============================================================================

    template <typename T> class SlotPool
    {
        std::vector<T>        _slots;
        std::vector<uint32_t> _freeList;
        mutable std::mutex    _mutex;

      public:
        explicit SlotPool(uint32_t capacity)
        {
            _slots.resize(capacity);
            _freeList.reserve(capacity);
            for (uint32_t i = capacity; i-- > 0;)
            {
                _freeList.push_back(i);
            }
        }
        [[nodiscard]] uint32_t alloc()
        {
            std::scoped_lock lk{_mutex};
            if (_freeList.empty())
            {
                throw std::runtime_error("[LightRHI] SlotPool exhausted");
            }
            uint32_t idx{_freeList.back()};
            _freeList.pop_back();
            return idx;
        }
        void free(uint32_t idx)
        {
            std::scoped_lock lk{_mutex};
            _slots[idx] = {};
            _freeList.push_back(idx);
        }
        [[nodiscard]] T &get(uint32_t idx) noexcept
        {
            return _slots[idx];
        }
        [[nodiscard]] const T &get(uint32_t idx) const noexcept
        {
            return _slots[idx];
        }
    };

    // ============================================================================
    // Internal resource records
    // ============================================================================

    struct MetalBuffer
    {
        NS::SharedPtr<MTL::Buffer> buffer;
        uint64_t                   size{0};
        BufferUsage                usage{};
    };

    struct MetalTexture
    {
        NS::SharedPtr<MTL::Texture> texture;
        TextureDesc                 desc{};
    };

    struct MetalSampler
    {
        NS::SharedPtr<MTL::SamplerState> state;
    };

    struct MetalAccelerationStructure
    {
        NS::SharedPtr<MTL::AccelerationStructure> as;
        AccelerationStructureType                 type{AccelerationStructureType::BottomLevel};
        uint64_t                                  size{0};
    };

    struct MetalPipeline
    {
        NS::SharedPtr<MTL::RenderPipelineState>  renderPso;
        NS::SharedPtr<MTL::ComputePipelineState> computePso;
        // Rasterizer/depth state applied to the encoder when this pipeline is set
        NS::SharedPtr<MTL::DepthStencilState> depthStencilState;
        MTL::Winding                          winding{MTL::WindingCounterClockwise};
        MTL::CullMode                         cullMode{MTL::CullModeNone};
        MTL::TriangleFillMode                 fillMode{MTL::TriangleFillModeFill};
        float                                 depthBiasConstant{0.f};
        float                                 depthBiasSlope{0.f};
        bool                                  isCompute{false};
        uint32_t                              threadGroupSizeX{1};
        uint32_t                              threadGroupSizeY{1};
        uint32_t                              threadGroupSizeZ{1};
    };

    struct MetalCommandResources
    {
        NS::SharedPtr<MTL4::CommandAllocator> Allocator;
        NS::SharedPtr<MTL4::CommandBuffer>    CommandBuffer;
        FenceHandle                           CompletionFence;
    };

    // ============================================================================
    // MetalBindlessHeap — the public view of Metal's native bindless model.
    //
    // Metal does not need a Vulkan-style descriptor buffer: buffer GPU
    // addresses and texture gpuResourceIDs are stored directly in root/scene
    // data. The queue-attached MTLResidencySet is the global heap of resources
    // reachable through those IDs. This object reports that heap's capacity and
    // occupancy; HeapAddress is zero because no descriptor-buffer address needs
    // to be passed to Metal shaders.
    // ============================================================================

    class MetalBindlessHeap final : public IBindlessHeap
    {
      public:
        // Defined out-of-line in metal_device.cpp only — this header is
        // textually #include-d (post-import) into more than one translation
        // unit of the same named module, and this toolchain does not fold
        // duplicate class-body-inline definitions across them the way a
        // non-modular build would, causing "duplicate symbol" link errors.
        [[nodiscard]] uint32_t   MaxBuffers() const noexcept override;
        [[nodiscard]] uint32_t   MaxTextures() const noexcept override;
        [[nodiscard]] uint32_t   MaxSamplers() const noexcept override;
        [[nodiscard]] GpuAddress HeapAddress() const noexcept override;
        [[nodiscard]] uint32_t   UsedBuffers() const noexcept override;
        [[nodiscard]] uint32_t   UsedTextures() const noexcept override;
        [[nodiscard]] uint32_t   UsedSamplers() const noexcept override;

        void RegisterBuffer() noexcept;
        void RegisterTexture() noexcept;
        void RegisterSampler() noexcept;
        void UnregisterBuffer() noexcept;
        void UnregisterTexture() noexcept;
        void UnregisterSampler() noexcept;

      private:
        std::atomic<uint32_t> _usedBuffers{0};
        std::atomic<uint32_t> _usedTextures{0};
        std::atomic<uint32_t> _usedSamplers{0};
    };

    // ============================================================================
    // MetalDevice declaration (implementation in metal_device.cpp)
    //
    // Resource binding under MTL4 (see bindless_texture_test.slang's header
    // comment for the full story of what was tried and rejected first):
    //   - Buffer device addresses (`buffer->gpuAddress()`) and acceleration-
    //     structure handles (`as->gpuResourceID()`) are still plain integer
    //     values embedded directly in push constants and reconstructed
    //     in-shader (`(T device*)someAddr`, `DescriptorHandle<
    //     RaytracingAccelerationStructure>`) — unchanged from before, and
    //     unaffected by anything below.
    //   - A sampled texture is fully bindless when its gpuResourceID is stored
    //     in a DescriptorHandle<Texture2D> field in root/scene data. Slang's
    //     Metal layout consumes those same eight bytes as a texture resource.
    //     The texture is made globally reachable by the queue-attached
    //     residency set. No ICommandList::BindTexture call is required.
    //   - BindTexture/BindSampler remain as an explicit fixed-slot API for
    //     shaders which intentionally declare top-level register(tN)/register(sN)
    //     parameters; they are not the bindless path.
    //   - Push constants themselves also go through the argument table now
    //     (MTL4's ComputeCommandEncoder has no setBytes/setBuffer at all —
    //     "all binding goes through the argument table"): SetPushConstants
    //     copies the caller's bytes into a small per-command-list scratch
    //     buffer and binds its address via ArgumentTable::setAddress at the
    //     fixed kPushConstantSlot (30), matching the existing
    //     register(b30) convention.
    // ============================================================================

    class MetalDevice final : public IDevice
    {
      public:
        static constexpr uint32_t kMaxBuffers{1u << 20};
        static constexpr uint32_t kMaxTextures{1u << 20};
        static constexpr uint32_t kMaxSamplers{2048};
        static constexpr uint32_t kMaxPipelines{65536};
        static constexpr uint32_t kMaxAccelerationStructures{4096};

        explicit MetalDevice(const DeviceDesc &desc);
        ~MetalDevice() override;

        // IDevice
        [[nodiscard]] std::string_view AdapterName() const noexcept override;
        [[nodiscard]] uint64_t         VideoMemoryBytes() const noexcept override;
        [[nodiscard]] IBindlessHeap   &BindlessHeap() noexcept override;

        [[nodiscard]] BufferHandle CreateBuffer(const BufferDesc &d) override;
        void                       DestroyBuffer(BufferHandle h) override;
        [[nodiscard]] GpuAddress   BufferAddress(BufferHandle h) const override;
        [[nodiscard]] BufferInfo   GetBufferInfo(BufferHandle h) const override;
        [[nodiscard]] MappedBuffer MapBuffer(BufferHandle h) override;
        void                       UnmapBuffer(BufferHandle h) override;

        [[nodiscard]] TextureHandle CreateTexture(const TextureDesc &d) override;
        void                        DestroyTexture(TextureHandle h) override;
        [[nodiscard]] GpuAddress    TextureAddress(TextureHandle h) const override;

        [[nodiscard]] SamplerHandle CreateSampler(const SamplerDesc &d) override;
        void                        DestroySampler(SamplerHandle h) override;
        [[nodiscard]] GpuAddress    SamplerAddress(SamplerHandle h) const override;

        [[nodiscard]] PipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &d) override;
        [[nodiscard]] PipelineHandle CreateComputePipeline(const ComputePipelineDesc &d) override;
        void                         DestroyPipeline(PipelineHandle h) override;

        // ---- Ray tracing ----
        [[nodiscard]] bool SupportsRayTracing() const noexcept override;
        [[nodiscard]] AccelerationStructureBuildSizes
        QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &d) const override;
        [[nodiscard]] AccelerationStructureHandle
                                 CreateAccelerationStructure(const AccelerationStructureDesc &d) override;
        void                     DestroyAccelerationStructure(AccelerationStructureHandle h) override;
        [[nodiscard]] GpuAddress AccelerationStructureAddress(AccelerationStructureHandle h) const override;

        [[nodiscard]] std::unique_ptr<ICommandList> CreateCommandList(QueueType        q    = QueueType::Graphics,
                                                                      std::string_view name = {}) override;

        void BeginCaptureScope(std::string_view name) override;
        void EndCaptureScope() override;

        [[nodiscard]] FenceHandle Submit(ICommandList &cmd, const SubmitDesc &d = {}) override;
        void                      WaitForFence(FenceHandle f) override;
        [[nodiscard]] bool        IsFenceComplete(FenceHandle f) override;
        void                      WaitIdle() override;

        void UploadBuffer(BufferHandle dst, const void *data, uint64_t size, uint64_t dstOffset = 0) override;
        void UploadTexture(TextureHandle dst, const void *data, uint64_t rowPitch, uint64_t slicePitch,
                           const TextureCopyRegion &region) override;

        // ---- Accessors for MetalCommandList ----
        [[nodiscard]] MTL::Device                *MtlDevice() const noexcept;
        [[nodiscard]] MTL4::CommandQueue         *Mtl4Queue() const noexcept;
        [[nodiscard]] bool                        DebugCaptureEnabled() const noexcept;
        [[nodiscard]] MetalBuffer                &Buffer(BufferHandle h);
        [[nodiscard]] MetalTexture               &Texture(TextureHandle h);
        [[nodiscard]] MetalSampler               &Sampler(SamplerHandle h);
        [[nodiscard]] MetalPipeline              &Pipeline(PipelineHandle h);
        [[nodiscard]] MetalAccelerationStructure &AccelStruct(AccelerationStructureHandle h);
        [[nodiscard]] FenceHandle                 NextFence() noexcept;
        [[nodiscard]] MTL::SharedEvent           *TimelineEvent() const noexcept;
        [[nodiscard]] MTL::CaptureScope          *SubmissionCaptureScope(std::string_view name);
        [[nodiscard]] MetalCommandResources       AcquireCommandResources();
        void                                      RecycleCommandResources(MetalCommandResources &&resources);

        // Acceleration structures deliberately stay on the CLASSIC (non-MTL4)
        // Metal raytracing API — everything else in this backend targets
        // MTL4 exclusively (see this class's header comment and
        // API_GUIDELINES.md), but MTL4's
        // MTL4::ComputeCommandEncoder::buildAccelerationStructure requires
        // real RT hardware and throws "Metal 4 does not support raytracing
        // with software emulation" on GPUs without it (discovered at runtime
        // on this project's Apple M1 Pro dev machine, which has no hardware
        // RT units — classic Metal quietly built BVHs via a software
        // emulation path, MTLGPUBVHBuilder, that MTL4 removes entirely).
        // Since this project must run on non-RT-hardware Macs, acceleration
        // structures are the one deliberate, scoped exception to "always
        // latest API": MakeAccelerationStructureDescriptor (classic types)
        // is used for BOTH sizing (QueryAccelerationStructureBuildSizes) and
        // the actual build (MetalCommandList::BuildAccelerationStructure,
        // which submits its own small classic MTL::CommandQueue — see
        // LegacyQueue() — synchronously, since classic
        // MTL4::CommandBuffer has no accelerationStructureCommandEncoder()).
        [[nodiscard]] NS::SharedPtr<MTL::AccelerationStructureDescriptor>
        MakeAccelerationStructureDescriptor(const AccelerationStructureDesc        &d,
                                            std::vector<NS::SharedPtr<NS::Object>> &keepAlive);

        // Classic MTL::CommandQueue used only for the acceleration-structure
        // build exception above.
        [[nodiscard]] MTL::CommandQueue *LegacyQueue() const noexcept;

        // BuildAccelerationStructure submits synchronously (commit +
        // waitUntilCompleted) on LegacyQueue() — a queue the active capture
        // scope (if any; see BeginCaptureScope) was never created from. That
        // blocking cross-queue wait happening mid-scope, on the same thread
        // Xcode's capture is recording, is a plausible source of capture
        // hangs, so BuildAccelerationStructure suspends the active scope
        // around its own submission and resumes it afterward. No-ops when no
        // scope is active.
        void SuspendActiveCaptureScope() noexcept;
        void ResumeActiveCaptureScope() noexcept;

      private:
        NS::SharedPtr<MTL::Device>        _device;
        NS::SharedPtr<MTL4::CommandQueue> _queue;
        NS::SharedPtr<MTL::CommandQueue> _legacyQueue; // acceleration structures only — see LegacyQueue()'s doc comment
        NS::SharedPtr<MTL4::Compiler>    _compiler;
        NS::SharedPtr<MTL::ResidencySet> _residencySet; // replaces classic per-encoder useResources()
        NS::SharedPtr<MTL::SharedEvent>  _timelineEvent;
        std::unordered_map<std::string, NS::SharedPtr<MTL::CaptureScope>> _submissionCaptureScopes;
        std::vector<MetalCommandResources>                                _commandResourcePool;
        // BeginCaptureScope's scope, by contrast, is NOT cached/reused by
        // name — see its doc comment for why a per-call scope is required.
        NS::SharedPtr<MTL::CaptureScope> _frameCaptureScope;
        std::mutex                       _captureScopeMutex;
        std::mutex                       _commandResourcePoolMutex;
        MTL::CaptureScope               *_activeCaptureScope{nullptr};
        uint64_t                         _timelineValue{0};
        bool                             _raytracingSupported{false};
        bool                             _debugCaptureEnabled{false};
        bool _gpuValidationEnabled{false}; // DeviceDesc::EnableGpuValidation — see _configurePipelineForDebugging

        std::string _adapterName;
        uint64_t    _videoMemoryBytes{0};

        SlotPool<MetalBuffer>                _buffers{kMaxBuffers};
        SlotPool<MetalTexture>               _textures{kMaxTextures};
        SlotPool<MetalSampler>               _samplers{kMaxSamplers};
        SlotPool<MetalPipeline>              _pipelines{kMaxPipelines};
        SlotPool<MetalAccelerationStructure> _accelStructs{kMaxAccelerationStructures};

        // Reverse lookup: base GPU address of a live buffer -> its slot index.
        // Needed because AccelerationStructureDesc addresses vertex/index data
        // by GpuAddress (BDA), but Metal's geometry descriptors take an
        // MTL::Buffer* + byte offset, not a raw pointer.
        std::unordered_map<uint64_t, uint32_t> _bufferAddrToIndex;
        mutable std::mutex                     _bufferAddrMutex;

        // Adds/removes a resource from the persistent residency set — call
        // right after MTL::Device::new{Buffer,Texture,AccelerationStructure}
        // and right before destroying one. Every resource that might be
        // referenced indirectly (BDA pointer, gpuResourceID, or an argument
        // table entry) must be resident; MTL4 has no per-encoder residency
        // declaration, only this queue-attached, incrementally-committed set.
        void _addResident(MTL::Allocation *res);
        void _removeResident(MTL::Allocation *res);

        [[nodiscard]] MTL::Buffer *_bufferAndOffsetFromAddress(GpuAddress addr, uint64_t &outOffset) const;

        // Conversion helpers (static — no state needed)
        [[nodiscard]] static MTL::ResourceOptions         _toOptions(MemoryType m) noexcept;
        [[nodiscard]] static MTL::TextureType             _toTexType(TextureDimension d) noexcept;
        [[nodiscard]] static MTL::PixelFormat             _toPixFmt(Format f) noexcept;
        [[nodiscard]] static MTL::TextureUsage            _toTexUsage(TextureUsage u) noexcept;
        [[nodiscard]] static MTL::SamplerMinMagFilter     _toMinMag(SamplerFilter f) noexcept;
        [[nodiscard]] static MTL::SamplerMipFilter        _toMipFlt(SamplerMipMode m) noexcept;
        [[nodiscard]] static MTL::SamplerAddressMode      _toAddrMode(SamplerAddressMode m) noexcept;
        [[nodiscard]] static MTL::CompareFunction         _toCompare(CompareOp op) noexcept;
        [[nodiscard]] static MTL::BlendFactor             _toBlendF(BlendFactor f) noexcept;
        [[nodiscard]] static MTL::BlendOperation          _toBlendOp(BlendOp op) noexcept;
        [[nodiscard]] static MTL::PrimitiveTopologyClass  _toTopology(PrimitiveTopology t) noexcept;
        [[nodiscard]] static MTL::Winding                 _toWinding(FrontFace f) noexcept;
        [[nodiscard]] static MTL::CullMode                _toCull(CullMode m) noexcept;
        [[nodiscard]] static MTL::DepthStencilDescriptor *_makeDepthStencilDesc(const DepthStencilState &ds,
                                                                                MTL::Device             *device);

        // Shader loading helper (returns retained MTL::Library*)
        [[nodiscard]] MTL::Library *_loadLibrary(const ShaderDesc &sd);
        void                        _configurePipelineForDebugging(MTL4::PipelineDescriptor *descriptor) const;

        MetalBindlessHeap _heap;
    };

} // namespace rhi::metal
