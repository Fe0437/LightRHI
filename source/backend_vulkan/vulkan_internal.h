/**
 * {file} vulkan_internal.h
 * {brief} Declares private types shared by the Vulkan backend implementation units.
 */
//
// IMPORTANT: in the including .cpp file, do `import rhi;` BEFORE including
// this header. The structs below use rhi:: types that become available via
// the module import; the header itself does not re-import.
//
// Vulkan entry points come through volk, not a linked loader: volk.h defines
// VK_NO_PROTOTYPES and turns every vk* symbol into a function pointer that
// volkInitialize()/volkLoadInstance() fill in at runtime from the driver's
// loader. This is why the backend links volk::volk + Vulkan::Headers instead
// of an SDK import library. Extension functions (VK_EXT_descriptor_buffer)
// are still loaded explicitly via vkGetDeviceProcAddr after device creation.

#pragma once

#include "vulkan_platform.h" // also included by each TU's global module fragment

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rhi::vulkan
{


    // ============================================================================
    // Error checking
    // ============================================================================

    /**
     * Stops on a Vulkan call that failed where nothing can be returned.
     *
     * VK_CHECK guards the calls whose failure leaves no usable state behind - a device that will
     * not create its queues, a submission the driver rejected. Where a caller can act on the
     * outcome instead, the code checks the VkResult itself and reports a refusal.
     */
    inline void vkFailOnError(VkResult r, const char *expr)
    {
        if (r != VK_SUCCESS)
        {
            FailContract(std::string("Vulkan: ") + expr + " failed (VkResult=" + std::to_string(static_cast<int>(r)) +
                         ')');
        }
    }
#define VK_CHECK(expr) ::rhi::vulkan::vkFailOnError((expr), #expr)

    // ============================================================================
    // SlotPool — free-list handle allocator
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
        [[nodiscard]] uint32_t alloc()
        {
            std::scoped_lock lk{_mutex};
            if (_freeList.empty())
            {
                return kInvalidIndex;
            }
            uint32_t idx{_freeList.back()};
            _freeList.pop_back();
            return idx;
        }
        void free(uint32_t idx)
        {
            std::scoped_lock lk{_mutex};
            assert(idx < _slots.size());
            _slots[idx] = {};
            _freeList.push_back(idx);
        }
        [[nodiscard]] T &get(uint32_t idx) noexcept
        {
            assert(idx < _slots.size());
            return _slots[idx];
        }
        [[nodiscard]] const T &get(uint32_t idx) const noexcept
        {
            assert(idx < _slots.size());
            return _slots[idx];
        }
    };

    // ============================================================================
    // Internal resource records
    // ============================================================================

    struct VkBuffer_
    {
        VkBuffer        buffer{VK_NULL_HANDLE};
        VmaAllocation   alloc{}; // null for a placed buffer, which owns no allocation
        uint64_t        size{0};
        BufferUsage     usage{};
        VkDeviceAddress bda{0}; // buffer device address (0 if unused)
        // Invalid for a buffer that owns its allocation. A placed buffer keeps its heap and offset
        // so destruction can release the range without freeing the heap.
        MemoryHeapHandle heap{};
        uint64_t         offset{0};
    };

    // Memory the application places textures into: one VkDeviceMemory allocation plus the live
    // ranges it currently holds. The ranges exist to refuse an overlapping placement, not to choose
    // one — offsets always come from the caller.
    struct VkMemoryHeap_
    {
        VkDeviceMemory               memory{VK_NULL_HANDLE};
        uint64_t                     size{0};
        MemoryType                   type{MemoryType::GpuOnly};
        uint32_t                     memoryTypeIndex{0};
        std::map<uint64_t, uint64_t> liveRanges{}; // placement offset -> end offset
    };

    struct VkTexture_
    {
        VkImage       image{VK_NULL_HANDLE};
        VkImageView   view{VK_NULL_HANDLE}; // full-resource default view
        VmaAllocation alloc{};              // null for a placed texture, which owns no allocation
        TextureDesc   desc{};
        // Invalid for a texture that owns its allocation. A placed texture keeps its heap and
        // offset so destruction can release the range without freeing the heap.
        MemoryHeapHandle heap{};
        uint64_t         offset{0};
        // True for a swapchain image borrowed for one frame: the swapchain owns the image and its
        // view, so destroying the handle releases neither.
        bool borrowed{false};
        // Current image layout, tracked so the command list can insert the
        // layout transitions Vulkan requires without the caller issuing explicit
        // barriers. Starts UNDEFINED (matches VkImageCreateInfo::initialLayout).
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    };

    struct VkSampler_
    {
        VkSampler sampler{VK_NULL_HANDLE};
    };

    struct VkTimestampQueryPool_
    {
        VkQueryPool pool{VK_NULL_HANDLE};
        uint32_t    count{0};
    };

    struct VkPipeline_
    {
        VkPipeline pipeline{VK_NULL_HANDLE};
        bool       isCompute{false};
        uint32_t   pushConstantBytes{128};
    };

    struct VkAccelStruct_
    {
        VkAccelerationStructureKHR as{VK_NULL_HANDLE};
        VkBuffer                   buffer{VK_NULL_HANDLE}; // backing storage, sized to build-size query
        VmaAllocation              alloc{};
        uint64_t                   size{0};
        VkDeviceAddress            deviceAddress{0}; // vkGetAccelerationStructureDeviceAddressKHR, set at create
    };

    // ============================================================================
    // Functions declared in vulkan_resources.cpp
    // ============================================================================

    VkFormat            toVkFormat(Format f) noexcept;
    VkSamplerCreateInfo toVkSamplerCreateInfo(const SamplerDesc &d) noexcept;

    // ============================================================================
    // VulkanBindlessHeap — VK_EXT_descriptor_buffer based
    // ============================================================================

    class VulkanBindlessHeap final : public IBindlessHeap
    {
      public:
        static constexpr uint32_t kMaxBuffers{1u << 20};
        static constexpr uint32_t kMaxTextures{1u << 20};
        static constexpr uint32_t kMaxSamplers{2048};

        VulkanBindlessHeap() = default;
        inline ~VulkanBindlessHeap() override
        { /* destroy() must be called explicitly */
        }

        void Init(VkDevice vkDev, VmaAllocator allocator,
                  const VkPhysicalDeviceDescriptorBufferPropertiesEXT &properties, PFN_vkGetDescriptorEXT getDescriptor,
                  PFN_vkGetDescriptorSetLayoutSizeEXT          getLayoutSize,
                  PFN_vkGetDescriptorSetLayoutBindingOffsetEXT getBindingOffset);
        void Destroy(VkDevice vkDev);

        // IBindlessHeap
        [[nodiscard]] inline uint32_t MaxBuffers() const noexcept override
        {
            return kMaxBuffers;
        }
        [[nodiscard]] inline uint32_t MaxTextures() const noexcept override
        {
            return kMaxTextures;
        }
        [[nodiscard]] inline uint32_t MaxSamplers() const noexcept override
        {
            return kMaxSamplers;
        }
        [[nodiscard]] inline GpuAddress HeapAddress() const noexcept override
        {
            return _heapBDA;
        }
        [[nodiscard]] inline uint32_t UsedBuffers() const noexcept override
        {
            return _usedBufs;
        }
        [[nodiscard]] inline uint32_t UsedTextures() const noexcept override
        {
            return _usedTexs;
        }
        [[nodiscard]] inline uint32_t UsedSamplers() const noexcept override
        {
            return _usedSmps;
        }

        // Internal: called by VulkanDevice after resource creation
        void RegisterBuffer(uint32_t slot, VkBuffer buf, uint64_t size, VkDeviceAddress bda);
        void RegisterTexture(uint32_t slot, VkImageView view);
        void RegisterSampler(uint32_t slot, VkSampler smp);

        // Clear the corresponding descriptor-array entry on destruction.
        void UnregisterBuffer(uint32_t slot);
        void UnregisterTexture(uint32_t slot);
        void UnregisterSampler(uint32_t slot);

        // For command list binding (vkCmdBindDescriptorBuffersEXT)
        [[nodiscard]] inline VkDeviceAddress DescriptorBufferAddress() const noexcept
        {
            return _heapBDA.Address;
        }
        [[nodiscard]] inline VkDescriptorSetLayout DescriptorSetLayout() const noexcept
        {
            return _layout;
        }
        [[nodiscard]] inline VkDescriptorSetLayout EmptyDescriptorSetLayout() const noexcept
        {
            return _emptyLayout;
        }

      private:
        VkDevice                                      _vkDev{VK_NULL_HANDLE};
        VmaAllocator                                  _allocator{};
        VkPhysicalDeviceDescriptorBufferPropertiesEXT _properties{};
        PFN_vkGetDescriptorEXT                        _getDescriptor{};
        PFN_vkGetDescriptorSetLayoutSizeEXT           _getLayoutSize{};
        PFN_vkGetDescriptorSetLayoutBindingOffsetEXT  _getBindingOffset{};

        VkDescriptorSetLayout _emptyLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout _layout{VK_NULL_HANDLE};
        VkBuffer              _descBuf{VK_NULL_HANDLE};
        VmaAllocation         _descAlloc{};
        void                 *_descMapped{nullptr};
        GpuAddress            _heapBDA{};

        // Slang DescriptorHandle heap bindings in system bindless space 1:
        // binding 0 = sampler array, binding 2 = sampled/resource image array.
        // Binding 3 preserves LightRHI's slot-to-BDA buffer-address table.
        VkDeviceSize  _smpBindOff{0}, _texBindOff{0}, _bufBindOff{0};
        VkBuffer      _bufTable{VK_NULL_HANDLE};
        VmaAllocation _bufTableAlloc{};
        uint64_t     *_bufTableMapped{nullptr};

        uint32_t _usedBufs{0}, _usedTexs{0}, _usedSmps{0};
    };

    // ============================================================================
    // LiveDeviceSlot
    // ============================================================================

    /**
     * The process's one live-device slot, held for as long as this object exists.
     *
     * volk dispatches every vk* call through process-global function pointers bound to one instance,
     * so a second live device would silently redirect the first one's calls. Claiming a slot that is
     * already held breaks that contract.
     */
    class LiveDeviceSlot
    {
      public:
        LiveDeviceSlot();
        ~LiveDeviceSlot();
        LiveDeviceSlot(const LiveDeviceSlot &)            = delete;
        LiveDeviceSlot &operator=(const LiveDeviceSlot &) = delete;
    };

    // ============================================================================
    // VulkanDevice
    // ============================================================================

    // ============================================================================
    // IFramePresenter — the one thing a recorded present needs of whatever it named.
    //
    // A provider that serves a presentable surface can do more than hand textures out: it holds
    // the frame it handed over and can show it as part of the submission that drew it. Naming that
    // ability on its own is what lets the device below store and resolve a present target without
    // knowing which concrete provider it is.
    // ============================================================================
    class IFramePresenter
    {
      public:
        IFramePresenter(const IFramePresenter &)            = delete;
        IFramePresenter(IFramePresenter &&)                 = delete;
        IFramePresenter &operator=(const IFramePresenter &) = delete;
        IFramePresenter &operator=(IFramePresenter &&)      = delete;

        virtual ~IFramePresenter() = default;

        /** Submits `cmd` and shows the frame this object is holding, as one queue operation. */
        [[nodiscard]] virtual FenceHandle SubmitAndPresentHeldFrame(ICommandList &cmd, const SubmitDesc &desc) = 0;

      protected:
        IFramePresenter() = default;
    };

    class VulkanDevice final : public IDevice
    {
        /** Restricts construction to Create(), while still letting it use std::make_unique. */
        struct ConstructionToken
        {
            explicit ConstructionToken() = default;
        };

      public:
        static constexpr uint32_t kMaxMemoryHeaps{4096};
        static constexpr uint32_t kMaxBuffers{1u << 20};
        static constexpr uint32_t kMaxTextures{1u << 20};
        static constexpr uint32_t kMaxSamplers{2048};
        static constexpr uint32_t kMaxPipelines{65536};
        static constexpr uint32_t kMaxAccelerationStructures{65536};
        static constexpr uint32_t kMaxTimestampQueryPools{1024};
        /// One per window an application shows at once; far more than any of them opens.
        static constexpr uint32_t kMaxExternalTextureProviders{64};

        /**
         * Creates the device, or reports why this system cannot provide one. A device that exists
         * holds every required Vulkan object, so no member function checks for a missing one.
         */
        [[nodiscard]] static std::expected<std::unique_ptr<VulkanDevice>, DeviceError> Create(const DeviceDesc &desc);

        explicit VulkanDevice(ConstructionToken);
        ~VulkanDevice() override;

        // ---- IDevice ----
        [[nodiscard]] inline std::string_view AdapterName() const noexcept override
        {
            return _adapterName;
        }
        [[nodiscard]] inline uint64_t VideoMemoryBytes() const noexcept override
        {
            return _videoMemoryBytes;
        }
        [[nodiscard]] inline IBindlessHeap &BindlessHeap() noexcept override
        {
            return _heap;
        }

        [[nodiscard]] std::expected<BufferHandle, DeviceError> CreateBuffer(const BufferDesc &d) override;
        void                                                   DestroyBuffer(BufferHandle h) override;
        [[nodiscard]] GpuAddress                               BufferAddress(BufferHandle h) const override;
        [[nodiscard]] BufferInfo                               GetBufferInfo(BufferHandle h) const override;
        [[nodiscard]] TextureInfo                              GetTextureInfo(TextureHandle h) const override;
        [[nodiscard]] MappedBuffer                             MapBuffer(BufferHandle h) override;
        void                                                   UnmapBuffer(BufferHandle h) override;

        [[nodiscard]] std::expected<PlacementRequirements, PlacementError>
        BufferPlacementRequirements(const BufferDesc &d) const override;
        [[nodiscard]] std::expected<BufferHandle, PlacementError> CreateBuffer(const BufferDesc    &d,
                                                                               const HeapPlacement &placement) override;

        [[nodiscard]] std::expected<MemoryHeapHandle, DeviceError> CreateMemoryHeap(const MemoryHeapDesc &d) override;
        void                                                       DestroyMemoryHeap(MemoryHeapHandle h) override;
        [[nodiscard]] std::expected<PlacementRequirements, PlacementError>
        TexturePlacementRequirements(const TextureDesc &d, MemoryType memory) const override;

        [[nodiscard]] std::expected<TextureHandle, DeviceError> CreateTexture(const TextureDesc &d) override;
        [[nodiscard]] std::expected<TextureHandle, PlacementError>
                                 CreateTexture(const TextureDesc &d, const HeapPlacement &placement) override;
        void                     DestroyTexture(TextureHandle h) override;
        [[nodiscard]] GpuAddress TextureAddress(TextureHandle h) const override;

        [[nodiscard]] std::expected<SamplerHandle, DeviceError> CreateSampler(const SamplerDesc &d) override;
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
        CreateGraphicsPipeline(const GraphicsPipelineDesc &d) override;
        [[nodiscard]] std::expected<PipelineHandle, DeviceError>
             CreateComputePipeline(const ComputePipelineDesc &d) override;
        void DestroyPipeline(PipelineHandle h) override;

        // ---- Ray tracing ----
        [[nodiscard]] bool SupportsRayTracing() const noexcept override;
        [[nodiscard]] AccelerationStructureBuildSizes
        QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &d) const override;
        [[nodiscard]] std::expected<AccelerationStructureHandle, DeviceError>
                                 CreateAccelerationStructure(const AccelerationStructureDesc &d) override;
        void                     DestroyAccelerationStructure(AccelerationStructureHandle h) override;
        [[nodiscard]] GpuAddress AccelerationStructureAddress(AccelerationStructureHandle h) const override;

        [[nodiscard]] std::expected<std::unique_ptr<ICommandList>, DeviceError>
        CreateCommandList(QueueType q = QueueType::Graphics, std::string_view name = {}) override;

        void BeginCaptureScope(std::string_view name) override;
        void EndCaptureScope() override;

        [[nodiscard]] FenceHandle Submit(ICommandList &cmd, const SubmitDesc &d = {}) override;
        void                      WaitForFence(FenceHandle f) override;
        [[nodiscard]] bool        IsFenceComplete(FenceHandle f) override;
        void                      WaitIdle() override;

        void UploadBuffer(BufferHandle dst, std::span<const std::byte> data, uint64_t dstOffset = 0) override;
        void UploadTexture(TextureHandle dst, std::span<const std::byte> data, uint64_t rowPitch, uint64_t slicePitch,
                           const TextureCopyRegion &region) override;

        // ---- Accessors for VulkanCommandList ----
        /**
         * Registers an external texture provider so a recorded present can name it by handle.
         * Returns an invalid handle when this device already holds kMaxExternalTextureProviders of them.
         */
        [[nodiscard]] ExternalTextureProviderHandle RegisterExternalTextureProvider(IFramePresenter &target);
        void                               UnregisterExternalTextureProvider(ExternalTextureProviderHandle handle) noexcept;
        /** Resolves a recorded present's handle, or null when it names nothing on this device. */
        [[nodiscard]] IFramePresenter *ResolveExternalTextureProvider(
            ExternalTextureProviderHandle handle) const noexcept;

        [[nodiscard]] inline VkDevice NativeDevice() const noexcept
        {
            return _device;
        }
        [[nodiscard]] inline VkPipelineLayout GlobalPipelineLayout() const noexcept
        {
            return _globalLayout;
        }
        [[nodiscard]] inline VmaAllocator Allocator() const noexcept
        {
            return _allocator;
        }
        [[nodiscard]] inline VkPhysicalDevice NativePhysicalDevice() const noexcept
        {
            return _physDev;
        }
        [[nodiscard]] inline uint32_t GraphicsQueueFamily() const noexcept
        {
            return gfxQ.family;
        }

        // ---- Accessors for VulkanPresentingTextureProvider ----
        /** Whether the hardware is gone, which a surface reports instead of handing out a texture. */
        [[nodiscard]] inline bool DeviceLost() const noexcept
        {
            return _deviceLost;
        }
        /** Records that the hardware is gone, so every later call answers the same way. */
        inline void MarkDeviceLost() noexcept
        {
            _deviceLost = true;
        }
        /** Names a swapchain image for one frame; released by DestroyTexture() once presented. */
        [[nodiscard]] std::expected<TextureHandle, DeviceError> AdoptExternalTexture(VkImage image, VkImageView view,
                                                                                    const TextureDesc &desc);
        /** Submits `cmd` around a surface's acquire and present semaphores. */
        [[nodiscard]] FenceHandle SubmitForPresent(ICommandList &cmd, const SubmitDesc &desc, VkSemaphore waitBinary,
                                                   VkSemaphore signalBinary);
        /** Shows `image` of `swapchain` once `waitBinary` signals. */
        [[nodiscard]] VkResult PresentImage(VkSwapchainKHR swapchain, uint32_t image, VkSemaphore waitBinary);
        /** The instance every surface this device draws to was made from. */
        [[nodiscard]] inline VkInstance NativeInstance() const noexcept
        {
            return _instance;
        }

        [[nodiscard]] inline VkBuffer_ &Buffer(BufferHandle h) noexcept
        {
            assert(h.Valid());
            VkBuffer_ &buffer{_buffers.get(h.Index)};
            assert(buffer.buffer != VK_NULL_HANDLE);
            return buffer;
        }
        [[nodiscard]] inline VkTexture_ &Texture(TextureHandle h) noexcept
        {
            assert(h.Valid());
            VkTexture_ &texture{_textures.get(h.Index)};
            assert(texture.image != VK_NULL_HANDLE);
            return texture;
        }
        [[nodiscard]] inline VkPipeline_ &Pipeline(PipelineHandle h) noexcept
        {
            assert(h.Valid());
            VkPipeline_ &pipeline{_pipelines.get(h.Index)};
            assert(pipeline.pipeline != VK_NULL_HANDLE);
            return pipeline;
        }
        [[nodiscard]] inline VkTimestampQueryPool_ &TimestampQueryPool(TimestampQueryPoolHandle h) noexcept
        {
            assert(h.Valid());
            VkTimestampQueryPool_ &pool{_timestampQueryPools.get(h.Index)};
            assert(pool.pool != VK_NULL_HANDLE);
            return pool;
        }
        [[nodiscard]] inline VulkanBindlessHeap &Heap() noexcept
        {
            return _heap;
        }
        [[nodiscard]] inline VkAccelStruct_ &AccelStruct(AccelerationStructureHandle h) noexcept
        {
            assert(h.Valid());
            VkAccelStruct_ &accelerationStructure{_accelStructs.get(h.Index)};
            assert(accelerationStructure.as != VK_NULL_HANDLE);
            return accelerationStructure;
        }

        // Populates a build-geometry-info + geometry for `desc`, ready to pass
        // to vkGetAccelerationStructureBuildSizesKHR (size query — pass
        // instanceBufferAddress=0, since the build-size query never
        // dereferences geometry data) or vkCmdBuildAccelerationStructuresKHR
        // (a real build — instanceBufferAddress must point at an uploaded
        // VkAccelerationStructureInstanceKHR[] array for TopLevel builds).
        // Out-params (not a return-by-value struct) because buildInfo holds a
        // pointer into `geometry` that must stay valid in the caller's frame.
        void BuildAccelGeometryInfo(const AccelerationStructureDesc &desc, VkDeviceAddress instanceBufferAddress,
                                    VkAccelerationStructureGeometryKHR          &outGeometry,
                                    VkAccelerationStructureBuildGeometryInfoKHR &outBuildInfo,
                                    uint32_t                                    &outPrimitiveCount) const;

        // Physical device descriptor buffer properties (set in _pickPhysicalDevice)
        VkPhysicalDeviceDescriptorBufferPropertiesEXT descBufProps{};

        // Extension function pointers — loaded after device creation
        PFN_vkGetDescriptorEXT                       pfn_GetDescriptorEXT{};
        PFN_vkGetDescriptorSetLayoutSizeEXT          pfn_GetLayoutSize{};
        PFN_vkGetDescriptorSetLayoutBindingOffsetEXT pfn_GetBindingOffset{};
        PFN_vkCmdBindDescriptorBuffersEXT            pfn_CmdBindDescriptorBuffers{};
        PFN_vkCmdSetDescriptorBufferOffsetsEXT       pfn_CmdSetDescriptorBufferOffsets{};
        PFN_vkCmdBeginDebugUtilsLabelEXT             pfn_CmdBeginDebugUtilsLabel{};
        PFN_vkCmdEndDebugUtilsLabelEXT               pfn_CmdEndDebugUtilsLabel{};
        PFN_vkCmdInsertDebugUtilsLabelEXT            pfn_CmdInsertDebugUtilsLabel{};

        // Ray tracing extension function pointers — null if !SupportsRayTracing().
        // Device-side build only (vkCmdBuildAccelerationStructuresKHR); the
        // host-side vkBuildAccelerationStructuresKHR entry point is
        // deliberately never loaded or used — Khronos has deprecated
        // host-side AS builds in favor of device-side builds.
        PFN_vkCreateAccelerationStructureKHR           pfn_CreateAccelerationStructure{};
        PFN_vkDestroyAccelerationStructureKHR          pfn_DestroyAccelerationStructure{};
        PFN_vkGetAccelerationStructureBuildSizesKHR    pfn_GetAccelerationStructureBuildSizes{};
        PFN_vkCmdBuildAccelerationStructuresKHR        pfn_CmdBuildAccelerationStructures{};
        PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_GetAccelerationStructureDeviceAddress{};

        [[nodiscard]] inline bool RaytracingSupported() const noexcept
        {
            return _raytracingSupported;
        }

        // Queue info (read by VulkanCommandList for Submit)
        struct QueueInfo
        {
            VkQueue    q{VK_NULL_HANDLE};
            uint32_t   family{~0u};
            std::mutex mtx{};
        };
        QueueInfo gfxQ{}, computeQ{}, transferQ{};

        /** The queue that work of `type` is submitted to. */
        [[nodiscard]] QueueInfo &QueueFor(QueueType type) noexcept;

        // Command pools (one per queue type)
        VkCommandPool gfxPool{VK_NULL_HANDLE};
        VkCommandPool computePool{VK_NULL_HANDLE};
        VkCommandPool xferPool{VK_NULL_HANDLE};
        std::mutex    gfxPoolMtx{}, computePoolMtx{}, xferPoolMtx{};

        /** A command pool with the mutex that serializes its use. */
        struct CommandPool
        {
            VkCommandPool pool{};
            std::mutex   &mutex;
        };

        /** The command pool that command buffers for `type` are allocated from. */
        [[nodiscard]] CommandPool CommandPoolFor(QueueType type) noexcept;

        // Timeline semaphore
        VkSemaphore _timeline{VK_NULL_HANDLE};
        uint64_t    _timelineValue{0};
        std::mutex  _timelineMtx{};

      private:
        // Helpers of this class only: named with a leading underscore, and now in the section
        // that makes that true rather than only implied.

        /**
         * Fills the image creation info for `d`. `queueFamilies` is borrowed by the returned
         * structure, so it must outlive every use of that structure.
         */
        [[nodiscard]] VkImageCreateInfo _imageCreateInfo(const TextureDesc &d, std::array<uint32_t, 3> &queueFamilies,
                                                         VkImageViewType    &outViewType,
                                                         VkImageAspectFlags &outAspect) const;

        /** Creates the default view, registers bindless slots and names a freshly bound image. */
        void _finishTexture(uint32_t idx, const TextureDesc &d, VkImageViewType viewType, VkFormat format,
                            VkImageAspectFlags aspect);

        /** Reports the memory type index matching `memory`, or the device's best device-local one. */
        [[nodiscard]] uint32_t _memoryTypeIndexFor(MemoryType memory) const;

        /**
         * How this device can place `d` in `memory`. Optimal tiling is tried first and linear
         * tiling second, because a discrete adapter usually refuses optimal images host-visible
         * memory while accepting linear ones. `Viable` is false when neither layout fits.
         */
        struct PlacementPlan
        {
            VkImageTiling tiling{VK_IMAGE_TILING_OPTIMAL};
            uint32_t      memoryTypeIndex{0};
            uint64_t      size{0};
            uint64_t      alignment{0};
            bool          viable{false};
        };

        [[nodiscard]] PlacementPlan _planPlacement(const TextureDesc &d, MemoryType memory) const;

        /**
         * Checks a placement against the heap it names and reserves its range on success.
         * Buffers and textures answer to the same rules, so both come through here.
         */
        [[nodiscard]] std::expected<std::reference_wrapper<VkMemoryHeap_>, PlacementError>
        _reservePlacement(const HeapPlacement &placement, uint32_t memoryTypeIndex, uint64_t size, uint64_t alignment);

        /** Fills the buffer creation info for `d`; shared by owned and placed creation. */
        [[nodiscard]] VkBufferCreateInfo _bufferCreateInfo(const BufferDesc        &d,
                                                           std::array<uint32_t, 3> &queueFamilies) const;

        /** Takes the buffer's address, registers bindless storage and names it. */
        void _finishBuffer(uint32_t idx, const BufferDesc &d);

        /** Submits `cmd`, additionally waiting on and signalling the given binary semaphores. */
        [[nodiscard]] FenceHandle _submit(ICommandList &cmd, const SubmitDesc &desc, VkSemaphore waitBinary,
                                          VkSemaphore signalBinary);
        // Declared first so it is released last, after every Vulkan object this device destroys.
        LiveDeviceSlot _liveDeviceSlot{};

        VkInstance       _instance{VK_NULL_HANDLE};
        VkPhysicalDevice _physDev{VK_NULL_HANDLE};
        VkDevice         _device{VK_NULL_HANDLE};
        VmaAllocator     _allocator{};

        VkDebugUtilsMessengerEXT _debugMessenger{VK_NULL_HANDLE};
        VkPipelineLayout         _globalLayout{VK_NULL_HANDLE};
        uint32_t                 _maxPushConstantBytes{128};

        // A device that has lost its hardware answers every later call the same way; a surface it
        // draws to reads this to report itself lost rather than handing out a texture.
        bool _deviceLost{false};

        SlotPool<VkMemoryHeap_>         _memoryHeaps{kMaxMemoryHeaps};
        SlotPool<VkBuffer_>             _buffers{kMaxBuffers};
        SlotPool<VkTexture_>            _textures{kMaxTextures};
        SlotPool<VkSampler_>            _samplers{kMaxSamplers};
        SlotPool<VkPipeline_>           _pipelines{kMaxPipelines};
        SlotPool<VkAccelStruct_>        _accelStructs{kMaxAccelerationStructures};
        SlotPool<VkTimestampQueryPool_> _timestampQueryPools{kMaxTimestampQueryPools};
        SlotPool<IFramePresenter *>     _externalTextureProviders{kMaxExternalTextureProviders};
        VulkanBindlessHeap              _heap{};

        // Set once in _pickPhysicalDevice; reflects whether
        // VK_KHR_acceleration_structure + VK_KHR_ray_query + their required
        // features are actually available on the selected physical device —
        // both are optional extensions here (unlike VK_EXT_descriptor_buffer,
        // which device selection requires), so a driver/device lacking them
        // still creates a working (non-ray-tracing) VulkanDevice.
        bool   _raytracingSupported{false};
        bool   _computeTimestampsSupported{false};
        double _timestampPeriodNanoseconds{0.0};

        std::string _adapterName{};
        uint64_t    _videoMemoryBytes{0};

        // Init steps
        /** Creates the required Vulkan objects in dependency order; stops at the first refusal. */
        [[nodiscard]] bool _createRequiredDeviceObjects(const DeviceDesc &desc);
        /** Creates the instance; false, with the reason reported, when there is no Vulkan loader. */
        [[nodiscard]] bool _createInstance(const DeviceDesc &desc);
        /** Selects the GPU and its queue families; false, with the reason reported, when none fits. */
        [[nodiscard]] bool _pickPhysicalDevice();
        void               _createLogicalDevice();
        void               _loadExtensionFunctions();
        void               _createAllocator();
        void               _createCommandPools();
        void               _createTimeline();
        void               _createGlobalLayout();

        // Helpers
        [[nodiscard]] VkImageAspectFlags _aspectMask(Format f) const noexcept;
        [[nodiscard]] VkImageUsageFlags  _toImageUsage(TextureUsage u) const noexcept;
        [[nodiscard]] VkBufferUsageFlags _toBufUsage(BufferUsage u) const noexcept;
        [[nodiscard]] VmaMemoryUsage     _toVmaUsage(MemoryType m) const noexcept;
        [[nodiscard]] VkShaderModule     _makeShaderModule(std::span<const uint32_t> spirv);
        void                             _setDebugName(VkObjectType type, uint64_t handle, std::string_view name);
    };

    // ============================================================================
    // VulkanPresentingTextureProvider — one VkSurfaceKHR, seen as somewhere to draw.
    //
    // The surface belongs to the application; the swapchain with its per-image views and semaphores
    // belongs to this target and is rebuilt whenever the surface changes size. Presentation is
    // ordered by the submission that drew the frame: that submission waits on the acquisition
    // semaphore and signals the one the present waits on, which is why a recorded present costs no
    // extra submission.
    // ============================================================================
    class VulkanPresentingTextureProvider final : public IExternalTextureProvider, public IFramePresenter
    {
      public:
        VulkanPresentingTextureProvider(VulkanDevice &device, VkInstance instance, VkSurfaceKHR surface,
                                        Format requestedFormat, PresentTiming timing) noexcept;
        ~VulkanPresentingTextureProvider() override;

        // IExternalTextureProvider
        [[nodiscard]] Format                                     TextureFormat() const noexcept override;
        [[nodiscard]] std::expected<TextureHandle, ExternalTextureError> NextTexture() override;
        [[nodiscard]] ExternalTextureProviderHandle                        Handle() const noexcept override;
#if METRICS_ENABLED
        /** Vulkan reports when a frame reached the screen only through extensions this does not use. */
        [[nodiscard]] std::size_t TakeShownFrames(std::span<ShownFrame> frames) noexcept override;
#endif

        // IFramePresenter
        [[nodiscard]] FenceHandle SubmitAndPresentHeldFrame(ICommandList &cmd, const SubmitDesc &desc) override;

      private:
        /** The surface's current size, which is what a new chain will be built at. */
        [[nodiscard]] Extent2D _surfaceExtent() const noexcept;
        /** The mode `_timing` asks for when the surface offers it, otherwise FIFO. */
        [[nodiscard]] VkPresentModeKHR _presentMode() const;
        /** Builds a chain for the surface's current size, retiring the previous one. */
        [[nodiscard]] bool _createSwapchain();
        /** Destroys the per-image views and semaphores, leaving the chain handle to its retirer. */
        void _destroySwapchain();
        /** Frees the texture slot holding the current swapchain image, if one is held. */
        void _releaseSurfaceTexture();

        VulkanDevice            &_device;
        VkInstance               _instance{VK_NULL_HANDLE};
        VkSurfaceKHR             _surface{VK_NULL_HANDLE};
        VkSwapchainKHR           _swapchain{VK_NULL_HANDLE};
        VkFormat                 _swapchainFormat{VK_FORMAT_UNDEFINED};
        VkExtent2D               _swapchainExtent{};
        std::vector<VkImage>     _swapchainImages{};
        std::vector<VkImageView> _swapchainViews{};
        std::vector<VkSemaphore> _acquireSemaphores{};
        std::vector<VkSemaphore> _presentSemaphores{};
        uint32_t                 _frameSlot{0};
        uint32_t                 _currentImage{UINT32_MAX};
        TextureHandle            _texture{};
        Format                   _format{Format::Undefined};
        Format                   _requestedFormat{Format::Undefined};
        PresentTiming                      _timing{PresentTiming::OnRefresh};
        ExternalTextureProviderHandle      _handle{}; ///< What a recorded present names this target by.
    };

} // namespace rhi::vulkan
