/**
 * {file} metal_internal.h
 * {brief} Declares private types shared by the Metal device and command-list implementations.
 */
//
// Include AFTER "import rhi;" in the including TU.
// Included by both metal_device.cpp and metal_command_list.cpp.
//
// Targets Metal 4 (MTL4CommandQueue/MTL4CommandBuffer/MTL4Compiler/
// MTL4ArgumentTable/MTLResidencySet) exclusively — see docs/API_GUIDELINES.md's
// "Target the latest platform API version — always" section. This requires
// macOS 26 / iOS 26 as the minimum deployment target; there is no classic-
// Metal fallback path.

#ifndef LIGHTRHI_BACKEND_METAL_METAL_INTERNAL_H
#define LIGHTRHI_BACKEND_METAL_METAL_INTERNAL_H

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
// Every standard header this file needs must already be included by each including TU's
// global module fragment (see the note above): <expected> and <map> are pulled in there.
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rhi::metal
{

    // ============================================================================
    // Metal object creation
    //
    // A Metal `new...` call returns a retained object, or null with an optional NS::Error when it
    // refuses. These turn that null into a reason once, at the call, so everything past it holds a
    // non-null object and never checks again.
    // ============================================================================

    /** Describes why Metal refused a request; Metal leaves `error` null when it gives no reason. */
    template <typename Error> [[nodiscard]] std::string ErrorText(Error *error)
    {
        return error != nullptr ? std::string{error->localizedDescription()->utf8String()} : std::string{"unknown"};
    }

    /**
     * Takes ownership of the object a Metal `new...` call returned. When Metal returned none, reports
     * `what` with Metal's reason and returns `refusal` instead.
     */
    template <typename T>
    [[nodiscard]] std::expected<NS::SharedPtr<T>, DeviceError> AdoptCreated(T *created, std::string_view what,
                                                                            NS::Error  *error   = nullptr,
                                                                            DeviceError refusal = DeviceError::Refused)
    {
        if (created == nullptr)
        {
            ReportRefusal(error != nullptr ? std::string{what} + ": " + ErrorText(error) : std::string{what});
            return std::unexpected{refusal};
        }
        return NS::TransferPtr(created);
    }

    /**
     * Takes ownership of an object a Metal `alloc()->init()` call returned, where the caller has no
     * way to report a refusal. Metal returns none only when memory is exhausted, which leaves
     * nothing sensible to do, so that stops the process with `what` as the reason.
     */
    template <typename T> [[nodiscard]] NS::SharedPtr<T> AdoptRequired(T *created, std::string_view what)
    {
        if (created == nullptr)
        {
            FailContract(what);
        }
        return NS::TransferPtr(created);
    }

    // ============================================================================
    // MakeLabel — a Metal label from a view that need not be null-terminated.
    //
    // NS::String::string reads a C string up to its terminator, and a std::string_view promises
    // no terminator: a caller may hand over a window into a larger buffer, and reading past the
    // length it gave is then out of bounds. The bytes are copied through a terminated temporary,
    // so the length the caller gave is the length that is read. NS::String copies them again, so
    // the temporary dying here is fine.
    // ============================================================================
    [[nodiscard]] inline NS::String *MakeLabel(const std::string_view text)
    {
        const std::string terminated{text};
        return NS::String::string(terminated.c_str(), NS::UTF8StringEncoding);
    }

    // ============================================================================
    // SlotPool
    // ============================================================================

    template <typename T> class SlotPool
    {
        std::vector<T>        _slots{};
        std::vector<uint32_t> _freeList{};
        mutable std::mutex    _mutex{};

      public:
        explicit SlotPool(uint32_t capacity)
        {
            _slots.resize(capacity);
            _freeList.reserve(capacity);
            for (uint32_t i{capacity}; i-- > 0;)
            {
                _freeList.push_back(i);
            }
        }
        /** Takes a free slot, or kInvalidIndex when the pool has none left. */
        [[nodiscard]] uint32_t Alloc()
        {
            const std::scoped_lock lk{_mutex};
            if (_freeList.empty())
            {
                return kInvalidIndex;
            }
            const uint32_t idx{_freeList.back()};
            _freeList.pop_back();
            return idx;
        }
        void Free(uint32_t idx)
        {
            const std::scoped_lock lk{_mutex};
            assert(idx < _slots.size());
            _slots[idx] = {};
            _freeList.push_back(idx);
        }
        [[nodiscard]] T &Get(uint32_t idx) noexcept
        {
            assert(idx < _slots.size());
            return _slots[idx];
        }
        [[nodiscard]] const T &Get(uint32_t idx) const noexcept
        {
            assert(idx < _slots.size());
            return _slots[idx];
        }
    };

    // ============================================================================
    // Internal resource records
    // ============================================================================

    struct MetalBuffer
    {
        NS::SharedPtr<MTL::Buffer> Buffer{};
        uint64_t                   Size{0};
        BufferUsage                Usage{};
        // Invalid for a buffer that owns its allocation. A placed buffer keeps its heap and offset
        // so destruction can release the range without touching the heap's residency.
        MemoryHeapHandle Heap{};
        uint64_t         Offset{0};
    };

    struct MetalTexture
    {
        NS::SharedPtr<MTL::Texture> Texture{};
        TextureDesc                 Desc{};
        // Invalid for a texture that owns its allocation. A placed texture keeps its heap and
        // offset so destruction can release the range without touching the heap's residency.
        MemoryHeapHandle Heap{};
        uint64_t         Offset{0};
        // True for a surface texture borrowed from a drawable: the layer owns it, the layer's
        // residency set covers it, and destroying the handle must not touch either.
        bool Borrowed{false};
    };

    // Memory the application places textures into: one MTLHeap of type placement, plus the live
    // ranges it currently holds. The ranges exist to refuse an overlapping placement, not to
    // choose one — offsets always come from the caller.
    struct MetalMemoryHeap
    {
        NS::SharedPtr<MTL::Heap>     Heap{};
        uint64_t                     Size{0};
        MemoryType                   Memory{MemoryType::GpuOnly};
        std::map<uint64_t, uint64_t> LiveRanges{}; // placement offset -> end offset
    };

    struct MetalSampler
    {
        NS::SharedPtr<MTL::SamplerState> State{};
    };

    struct MetalTimestampQueryPool
    {
        NS::SharedPtr<MTL4::CounterHeap> Heap{};
        uint32_t                         Count{0};
    };

    struct MetalAccelerationStructure
    {
        NS::SharedPtr<MTL::AccelerationStructure> As{};
        AccelerationStructureType                 Type{AccelerationStructureType::BottomLevel};
        uint64_t                                  Size{0};
    };

    struct MetalPipeline
    {
        NS::SharedPtr<MTL::RenderPipelineState>  RenderPso{};
        NS::SharedPtr<MTL::ComputePipelineState> ComputePso{};
        // Rasterizer/depth state applied to the encoder when this pipeline is set
        NS::SharedPtr<MTL::DepthStencilState> DepthStencilState{};
        MTL::Winding                          Winding{MTL::WindingCounterClockwise};
        MTL::CullMode                         CullMode{MTL::CullModeNone};
        MTL::TriangleFillMode                 FillMode{MTL::TriangleFillModeFill};
        float                                 DepthBiasConstant{0.F};
        float                                 DepthBiasSlope{0.F};
        bool                                  IsCompute{false};
        uint32_t                              ThreadGroupSizeX{1};
        uint32_t                              ThreadGroupSizeY{1};
        uint32_t                              ThreadGroupSizeZ{1};
    };

    struct MetalCommandResources
    {
        NS::SharedPtr<MTL4::CommandAllocator> Allocator{};
        NS::SharedPtr<MTL4::CommandBuffer>    CommandBuffer{};
        FenceHandle                           CompletionFence{};
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
    //     fixed kPushConstantSlot (30), matching
    //     LIGHTRHI_PUSH_CONSTANT_REGISTER in the shader ABI header.
    // ============================================================================

    // ============================================================================
    // IFramePresenter — the one thing a recorded present needs of whatever it named.
    //
    // A provider that serves a presentable surface can do more than hand textures out: it holds
    // the frame it handed over and can show it once the work that drew it is committed. Naming
    // that ability on its own is what lets the device below store and resolve a present target
    // without knowing which concrete provider it is.
    // ============================================================================
    class IFramePresenter
    {
      public:
        IFramePresenter(const IFramePresenter &)            = delete;
        IFramePresenter(IFramePresenter &&)                 = delete;
        IFramePresenter &operator=(const IFramePresenter &) = delete;
        IFramePresenter &operator=(IFramePresenter &&)      = delete;

        virtual ~IFramePresenter();

        /** Shows the frame this object is holding, after the submission that drew it. */
        virtual void PresentHeldFrame() = 0;

      protected:
        IFramePresenter();
    };

    class MetalDevice final : public IDevice
    {
      public:
        static constexpr uint32_t kMaxBuffers{1U << 20U};
        static constexpr uint32_t kMaxMemoryHeaps{4096};
        static constexpr uint32_t kMaxTextures{1U << 20U};
        static constexpr uint32_t kMaxSamplers{2048};
        static constexpr uint32_t kMaxPipelines{65536};
        static constexpr uint32_t kMaxAccelerationStructures{65536};
        static constexpr uint32_t kMaxTimestampQueryPools{1024};
        /// One per window an application shows at once; far more than any of them opens.
        static constexpr uint32_t kMaxExternalTextureProviders{64};

      private:
        /** Restricts construction to Create(), while still letting it use std::make_unique. */
        struct ConstructionToken
        {
            explicit ConstructionToken();
        };

        /** The Metal objects a device cannot work without, each one non-null. */
        struct RequiredObjects
        {
            NS::SharedPtr<MTL::Device>        Device{};
            NS::SharedPtr<MTL4::CommandQueue> Queue{};
            NS::SharedPtr<MTL::CommandQueue>  LegacyQueue{};
            NS::SharedPtr<MTL4::Compiler>     Compiler{};
            NS::SharedPtr<MTL::ResidencySet>  ResidencySet{};
            NS::SharedPtr<MTL::SharedEvent>   TimelineEvent{};
        };

        /** Creates every required object, stopping at the first one Metal refuses. */
        [[nodiscard]] static std::expected<RequiredObjects, DeviceError> _createRequiredObjects(const DeviceDesc &desc);

      public:
        /**
         * Creates the device, or reports why this system cannot provide one. A device that exists
         * holds every required Metal object, so no member function checks for a missing one.
         */
        [[nodiscard]] static std::expected<std::unique_ptr<MetalDevice>, DeviceError> Create(const DeviceDesc &desc);

        MetalDevice(ConstructionToken /*unused*/, RequiredObjects objects, const DeviceDesc &desc);

        // Defined out-of-line in metal_device.cpp for the reason given on MetalBindlessHeap above:
        // this header reaches more than one translation unit of the same named module.
        MetalDevice(const MetalDevice &)            = delete;
        MetalDevice(MetalDevice &&)                 = delete;
        MetalDevice &operator=(const MetalDevice &) = delete;
        MetalDevice &operator=(MetalDevice &&)      = delete;
        ~MetalDevice() override;

        // IDevice
        [[nodiscard]] std::string_view AdapterName() const noexcept override;
        [[nodiscard]] uint64_t         VideoMemoryBytes() const noexcept override;
        [[nodiscard]] IBindlessHeap   &BindlessHeap() noexcept override;

        [[nodiscard]] std::expected<MemoryHeapHandle, DeviceError>
             CreateMemoryHeap(const MemoryHeapDesc &desc) override;
        void DestroyMemoryHeap(MemoryHeapHandle h) override;
        [[nodiscard]] std::expected<PlacementRequirements, PlacementError>
        TexturePlacementRequirements(const TextureDesc &desc, MemoryType memory) const override;

        [[nodiscard]] std::expected<PlacementRequirements, PlacementError>
        BufferPlacementRequirements(const BufferDesc &desc) const override;

        [[nodiscard]] std::expected<BufferHandle, DeviceError>    CreateBuffer(const BufferDesc &desc) override;
        [[nodiscard]] std::expected<BufferHandle, PlacementError> CreateBuffer(const BufferDesc    &desc,
                                                                               const HeapPlacement &placement) override;
        void                                                      DestroyBuffer(BufferHandle h) override;
        [[nodiscard]] GpuAddress                                  BufferAddress(BufferHandle h) const override;
        [[nodiscard]] BufferInfo                                  GetBufferInfo(BufferHandle h) const override;
        [[nodiscard]] TextureInfo                                 GetTextureInfo(TextureHandle h) const override;
        [[nodiscard]] MappedBuffer                                MapBuffer(BufferHandle h) override;
        void                                                      UnmapBuffer(BufferHandle h) override;

        [[nodiscard]] std::expected<TextureHandle, DeviceError> CreateTexture(const TextureDesc &desc) override;
        [[nodiscard]] std::expected<TextureHandle, PlacementError>
                                 CreateTexture(const TextureDesc &desc, const HeapPlacement &placement) override;
        void                     DestroyTexture(TextureHandle h) override;
        [[nodiscard]] GpuAddress TextureAddress(TextureHandle h) const override;

        [[nodiscard]] std::expected<SamplerHandle, DeviceError> CreateSampler(const SamplerDesc &desc) override;
        void                                                    DestroySampler(SamplerHandle h) override;
        [[nodiscard]] GpuAddress                                SamplerAddress(SamplerHandle h) const override;

        [[nodiscard]] bool   SupportsComputeTimestamps() const noexcept override;
        [[nodiscard]] double TimestampPeriodNanoseconds() const noexcept override;
        [[nodiscard]] std::expected<TimestampQueryPoolHandle, DeviceError>
             CreateTimestampQueryPool(uint32_t count) override;
        void DestroyTimestampQueryPool(TimestampQueryPoolHandle pool) override;
        void ResetTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, uint32_t count) override;
        void ReadTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, std::span<uint64_t> results) override;

        [[nodiscard]] std::expected<PipelineHandle, DeviceError>
        CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) override;
        [[nodiscard]] std::expected<PipelineHandle, DeviceError>
             CreateComputePipeline(const ComputePipelineDesc &desc) override;
        void DestroyPipeline(PipelineHandle h) override;

        // ---- Ray tracing ----
        [[nodiscard]] bool SupportsRayTracing() const noexcept override;
        [[nodiscard]] AccelerationStructureBuildSizes
        QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &desc) const override;
        [[nodiscard]] std::expected<AccelerationStructureHandle, DeviceError>
                                 CreateAccelerationStructure(const AccelerationStructureDesc &desc) override;
        void                     DestroyAccelerationStructure(AccelerationStructureHandle h) override;
        [[nodiscard]] GpuAddress AccelerationStructureAddress(AccelerationStructureHandle h) const override;

        [[nodiscard]] std::expected<std::unique_ptr<ICommandList>, DeviceError>
        CreateCommandList(QueueType q = QueueType::Graphics, std::string_view name = {}) override;

        void BeginCaptureScope(std::string_view name) override;
        void EndCaptureScope() override;

        [[nodiscard]] FenceHandle Submit(ICommandList &cmdList, const SubmitDesc &desc = {}) override;
        void                      WaitForFence(FenceHandle fence) override;
        [[nodiscard]] bool        IsFenceComplete(FenceHandle fence) override;
        void                      WaitIdle() override;

        void UploadBuffer(BufferHandle dst, std::span<const std::byte> data, uint64_t dstOffset = 0) override;
        void UploadTexture(TextureHandle dst, std::span<const std::byte> data, uint64_t rowPitch, uint64_t slicePitch,
                           const TextureCopyRegion &region) override;

        // ---- Accessors for MetalCommandList ----
        [[nodiscard]] MTL::Device                      &MtlDevice() const noexcept;
        [[nodiscard]] MTL4::CommandQueue               &Mtl4Queue() const noexcept;
        [[nodiscard]] bool                              DebugCaptureEnabled() const noexcept;
        [[nodiscard]] MetalBuffer                      &Buffer(BufferHandle h);
        [[nodiscard]] MetalTexture                     &Texture(TextureHandle h);
        [[nodiscard]] MetalSampler                     &Sampler(SamplerHandle h);
        [[nodiscard]] MetalTimestampQueryPool          &TimestampQueryPool(TimestampQueryPoolHandle h);
        [[nodiscard]] MetalPipeline                    &Pipeline(PipelineHandle h);
        [[nodiscard]] MetalAccelerationStructure       &AccelStruct(AccelerationStructureHandle h);
        [[nodiscard]] const MetalAccelerationStructure &AccelStruct(AccelerationStructureHandle h) const;
        [[nodiscard]] FenceHandle                       NextFence() noexcept;
        [[nodiscard]] MTL::SharedEvent                 &TimelineEvent() const noexcept;
        /** The capture scope for a submission named `name`, or null while capture metadata is off. */
        [[nodiscard]] MTL::CaptureScope *SubmissionCaptureScope(std::string_view name);
        /** Whether the hardware is gone, which a surface reports instead of handing out a texture. */
        [[nodiscard]] bool DeviceLost() const noexcept;
        /** The Metal pixel format a surface layer must carry to hand out textures of `f`. */
        [[nodiscard]] static MTL::PixelFormat ExternalTexturePixelFormat(Format f) noexcept;
        /** Names a drawable's texture for one frame; released by DestroyTexture() once presented. */
        [[nodiscard]] std::expected<TextureHandle, DeviceError> AdoptExternalTexture(NS::SharedPtr<MTL::Texture> tex,
                                                                                     const TextureDesc          &desc);
        /** Reuses resources a completed submission released, or creates new ones. */
        [[nodiscard]] std::expected<MetalCommandResources, DeviceError> AcquireCommandResources();
        void RecycleCommandResources(MetalCommandResources &&resources);

        /**
         * Registers a present target so a recorded present can name it by handle.
         * Returns an invalid handle when this device already holds kMaxExternalTextureProviders of them.
         * {note} Takes the ability, not the provider: this device never needs to know which kind of
         * provider is behind a present, only that the frame it named can be shown.
         */
        [[nodiscard]] ExternalTextureProviderHandle RegisterExternalTextureProvider(IFramePresenter &target);
        void UnregisterExternalTextureProvider(ExternalTextureProviderHandle handle) noexcept;
        /** Resolves a recorded present's handle, or null when it names nothing on this device. */
        [[nodiscard]] IFramePresenter *
        ResolveExternalTextureProvider(ExternalTextureProviderHandle handle) const noexcept;

        // Acceleration structures deliberately stay on the CLASSIC (non-MTL4)
        // Metal raytracing API — everything else in this backend targets
        // MTL4 exclusively (see this class's header comment and
        // docs/API_GUIDELINES.md), but MTL4's
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
        MakeAccelerationStructureDescriptor(const AccelerationStructureDesc        &desc,
                                            std::vector<NS::SharedPtr<NS::Object>> &keepAlive) const;

        // Classic MTL::CommandQueue used only for the acceleration-structure
        // build exception above.
        [[nodiscard]] MTL::CommandQueue &LegacyQueue() const noexcept;

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
        // Non-null for the device's whole life: Create() builds no device without every one of them.
        NS::SharedPtr<MTL::Device>        _device{};
        NS::SharedPtr<MTL4::CommandQueue> _queue{};
        NS::SharedPtr<MTL::CommandQueue>
                                      _legacyQueue{}; // acceleration structures only — see LegacyQueue()'s doc comment
        NS::SharedPtr<MTL4::Compiler> _compiler{};
        NS::SharedPtr<MTL::ResidencySet> _residencySet{}; // replaces classic per-encoder useResources()
        NS::SharedPtr<MTL::SharedEvent>  _timelineEvent{};
        std::unordered_map<std::string, NS::SharedPtr<MTL::CaptureScope>> _submissionCaptureScopes{};
        std::vector<MetalCommandResources>                                _commandResourcePool{};
        // BeginCaptureScope's scope, by contrast, is NOT cached/reused by
        // name — see its doc comment for why a per-call scope is required.
        NS::SharedPtr<MTL::CaptureScope> _frameCaptureScope{};
        std::mutex                       _captureScopeMutex{};
        std::mutex                       _commandResourcePoolMutex{};
        MTL::CaptureScope               *_activeCaptureScope{nullptr};
        uint64_t                         _timelineValue{0};
        bool                             _raytracingSupported{false};
        bool                             _debugCaptureEnabled{false};
        bool _gpuValidationEnabled{false}; // DeviceDesc::EnableGpuValidation — see _configurePipelineForDebugging

        std::string _adapterName{};
        uint64_t    _videoMemoryBytes{0};

        // A device that has lost its hardware answers every later call the same way; a surface it
        // draws to reads this to report itself lost rather than handing out a texture.
        bool _deviceLost{false};

        SlotPool<MetalMemoryHeap>            _memoryHeaps{kMaxMemoryHeaps};
        SlotPool<MetalBuffer>                _buffers{kMaxBuffers};
        SlotPool<MetalTexture>               _textures{kMaxTextures};
        SlotPool<MetalSampler>               _samplers{kMaxSamplers};
        SlotPool<MetalPipeline>              _pipelines{kMaxPipelines};
        SlotPool<MetalAccelerationStructure> _accelStructs{kMaxAccelerationStructures};
        SlotPool<MetalTimestampQueryPool>    _timestampQueryPools{kMaxTimestampQueryPools};
        SlotPool<IFramePresenter *>          _externalTextureProviders{kMaxExternalTextureProviders};

        // Reverse lookup: base GPU address of a live buffer -> its slot index.
        // Needed because AccelerationStructureDesc addresses vertex/index data
        // by GpuAddress (BDA), but Metal's geometry descriptors take an
        // MTL::Buffer* + byte offset, not a raw pointer.
        std::unordered_map<uint64_t, uint32_t> _bufferAddrToIndex{};
        mutable std::mutex                     _bufferAddrMutex{};

        // Adds/removes a resource from the persistent residency set — call
        // right after MTL::Device::new{Buffer,Texture,AccelerationStructure}
        // and right before destroying one. Every resource that might be
        // referenced indirectly (BDA pointer, gpuResourceID, or an argument
        // table entry) must be resident; MTL4 has no per-encoder residency
        // declaration, only this queue-attached, incrementally-committed set.
        void _addResident(MTL::Allocation &res);
        void _removeResident(MTL::Allocation &res);

        /**
         * The live buffer whose base address is `addr`. An address that names none is a broken
         * contract, reported with `role` naming which address it was.
         */
        [[nodiscard]] MTL::Buffer &_bufferAtAddress(GpuAddress addr, std::string_view role) const;

        /** Builds the MTLTextureDescriptor for `desc`; shared by owned and placed creation. */
        [[nodiscard]] static NS::SharedPtr<MTL::TextureDescriptor> _makeTextureDescriptor(const TextureDesc &desc,
                                                                                          MTL::StorageMode   storage);

        /**
         * Checks a placement against the heap it names and reserves its range on success.
         * Buffers and textures answer to the same rules, so both come through here.
         */
        [[nodiscard]] std::expected<std::reference_wrapper<MetalMemoryHeap>, PlacementError>
        _reservePlacement(const HeapPlacement &placement, MemoryType memory, uint64_t size, uint64_t alignment);

        /** Registers the finished buffer in a slot, indexes its address and reports its handle. */
        [[nodiscard]] std::expected<BufferHandle, DeviceError>
        _adoptBuffer(NS::SharedPtr<MTL::Buffer> buf, const BufferDesc &desc, MemoryHeapHandle heap, uint64_t offset);

        /** Registers the finished texture in a slot and reports its handle. */
        [[nodiscard]] std::expected<TextureHandle, DeviceError> _adoptTexture(NS::SharedPtr<MTL::Texture> tex,
                                                                              const TextureDesc          &desc,
                                                                              MemoryHeapHandle heap, uint64_t offset,
                                                                              bool borrowed = false);

        // Conversion helpers (static — no state needed)
        [[nodiscard]] static MTL::ResourceOptions        _toOptions(MemoryType m) noexcept;
        [[nodiscard]] static MTL::StorageMode            _toStorageMode(MemoryType m) noexcept;
        [[nodiscard]] static MTL::TextureType            _toTexType(TextureDimension d) noexcept;
        [[nodiscard]] static MTL::PixelFormat            _toPixFmt(Format f) noexcept;
        [[nodiscard]] static MTL::TextureUsage           _toTexUsage(TextureUsage u) noexcept;
        [[nodiscard]] static MTL::SamplerMinMagFilter    _toMinMag(SamplerFilter f) noexcept;
        [[nodiscard]] static MTL::SamplerMipFilter       _toMipFlt(SamplerMipMode m) noexcept;
        [[nodiscard]] static MTL::SamplerAddressMode     _toAddrMode(SamplerAddressMode m) noexcept;
        [[nodiscard]] static MTL::CompareFunction        _toCompare(CompareOp op) noexcept;
        [[nodiscard]] static MTL::BlendFactor            _toBlendF(BlendFactor f) noexcept;
        [[nodiscard]] static MTL::BlendOperation         _toBlendOp(BlendOp op) noexcept;
        [[nodiscard]] static MTL::PrimitiveTopologyClass _toTopology(PrimitiveTopology t) noexcept;
        [[nodiscard]] static MTL::Winding                _toWinding(FrontFace f) noexcept;
        [[nodiscard]] static MTL::CullMode               _toCull(CullMode m) noexcept;
        [[nodiscard]] static NS::SharedPtr<MTL::DepthStencilDescriptor>
        _makeDepthStencilDesc(const DepthStencilState &ds);

        // Shader loading helper
        [[nodiscard]] NS::SharedPtr<MTL::Library> _loadLibrary(const ShaderDesc &sd);
        void _configurePipelineForDebugging(MTL4::PipelineDescriptor &descriptor) const;

        MetalBindlessHeap _heap{};
    };

    // ============================================================================
    // MetalPresentingTextureProvider — one CAMetalLayer, seen as somewhere to draw.
    //
    // The layer belongs to the application, which sizes it; this takes a drawable when one is free,
    // names its texture for the frame, and shows it once the submission that drew it is committed.
    // Metal 4 orders that on the queue rather than on the command buffer: wait for the drawable
    // before the work, signal it after, then present.
    // ============================================================================
    class MetalPresentingTextureProvider final : public IExternalTextureProvider, public IFramePresenter
    {
      public:
        /**
         * {param paced} Take drawables from the layer's display link, one per refresh, instead of
         * asking the layer; ignored where CAMetalDisplayLink does not exist.
         */
        MetalPresentingTextureProvider(MetalDevice &device, NS::SharedPtr<CA::MetalLayer> layer, Format format,
                                       bool paced) noexcept;
        MetalPresentingTextureProvider(const MetalPresentingTextureProvider &)            = delete;
        MetalPresentingTextureProvider(MetalPresentingTextureProvider &&)                 = delete;
        MetalPresentingTextureProvider &operator=(const MetalPresentingTextureProvider &) = delete;
        MetalPresentingTextureProvider &operator=(MetalPresentingTextureProvider &&)      = delete;
        ~MetalPresentingTextureProvider() override;

        // IExternalTextureProvider
        [[nodiscard]] Format                                             TextureFormat() const noexcept override;
        [[nodiscard]] std::expected<TextureHandle, ExternalTextureError> NextTexture() override;
        [[nodiscard]] ExternalTextureProviderHandle                      Handle() const noexcept override;
#if METRICS_ENABLED
        [[nodiscard]] std::size_t TakeShownFrames(std::span<ShownFrame> frames) noexcept override;
#endif

        // IFramePresenter
        void PresentHeldFrame() override;

#if METRICS_ENABLED
        /**
         * {brief} Fates the platform reported, waiting to be taken; shared with the presented
         * handlers, which may run after this provider is gone.
         */
        struct ShownFrames
        {
            std::mutex                 Lock;
            std::array<ShownFrame, 64> Fates{};
            std::size_t                First{};
            std::size_t                Count{};
        };
#endif // METRICS_ENABLED

      private:
        /** Waits for the held drawable on the queue and names its texture for this frame. */
        [[nodiscard]] std::expected<TextureHandle, ExternalTextureError> _adoptHeldDrawable(Extent2D extent);
        /** The layer's current size, which is what a new drawable will be. */
        [[nodiscard]] Extent2D _layerExtent() const noexcept;

        MetalDevice                     &_device;
        NS::SharedPtr<CA::MetalLayer>    _layer{};
        NS::SharedPtr<CA::MetalDrawable> _drawable{}; ///< Held between NextTexture() and the present.
        /// Frames presented and not yet on screen. Shared with the presented handlers, which may run
        /// after this provider is gone.
        std::shared_ptr<std::atomic<uint32_t>> _framesWaiting{std::make_shared<std::atomic<uint32_t>>(0U)};
#if METRICS_ENABLED
        std::shared_ptr<ShownFrames> _shown{std::make_shared<ShownFrames>()};
        uint64_t                     _presents{}; ///< Presents made so far.
#endif
        /// The layer's display link, which hands over a drawable once per refresh; 0 when unpaced.
        std::uintptr_t                _pacer{};
        TextureHandle                 _texture{}; ///< The drawable's texture, named for this frame only.
        Format                        _format{Format::Undefined};
        ExternalTextureProviderHandle _handle{}; ///< What a recorded present names this target by.
    };

} // namespace rhi::metal

#endif // LIGHTRHI_BACKEND_METAL_METAL_INTERNAL_H
