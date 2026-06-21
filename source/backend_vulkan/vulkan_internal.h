// vulkan_internal.h — shared internal types for the LightRHI Vulkan backend.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <volk.h>         // defines VK_NO_PROTOTYPES + pulls in <vulkan/vulkan.h>
#include <vk_mem_alloc.h> // must follow volk so VMA sees VK_NO_PROTOTYPES

namespace rhi::vulkan
{

    // ============================================================================
    // Error checking
    // ============================================================================

    inline void vkThrowOnFail(VkResult r, const char *expr)
    {
        if (r != VK_SUCCESS)
        {
            throw std::runtime_error(std::string("[LightRHI::Vulkan] ") + expr +
                                     " failed (VkResult=" + std::to_string(static_cast<int>(r)) + ')');
        }
    }
#define VK_CHECK(expr) ::rhi::vulkan::vkThrowOnFail((expr), #expr)

    // ============================================================================
    // SlotPool — free-list handle allocator
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

    struct VkBuffer_
    {
        VkBuffer        buffer{VK_NULL_HANDLE};
        VmaAllocation   alloc{};
        uint64_t        size{0};
        BufferUsage     usage{};
        VkDeviceAddress bda{0}; // buffer device address (0 if unused)
    };

    struct VkTexture_
    {
        VkImage       image{VK_NULL_HANDLE};
        VkImageView   view{VK_NULL_HANDLE}; // full-resource default view
        VmaAllocation alloc{};
        TextureDesc   desc{};
        // Current image layout, tracked so the command list can insert the
        // layout transitions Vulkan requires without the caller issuing explicit
        // barriers. Starts UNDEFINED (matches VkImageCreateInfo::initialLayout).
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    };

    struct VkSampler_
    {
        VkSampler sampler{VK_NULL_HANDLE};
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
                  const VkPhysicalDeviceDescriptorBufferPropertiesEXT &properties,
                  PFN_vkGetDescriptorEXT getDescriptor,
                  PFN_vkGetDescriptorSetLayoutSizeEXT getLayoutSize,
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
    // VulkanDevice
    // ============================================================================

    class VulkanDevice final : public IDevice
    {
      public:
        static constexpr uint32_t kMaxBuffers{1u << 20};
        static constexpr uint32_t kMaxTextures{1u << 20};
        static constexpr uint32_t kMaxSamplers{2048};
        static constexpr uint32_t kMaxPipelines{65536};
        static constexpr uint32_t kMaxAccelerationStructures{4096};

        explicit VulkanDevice(const DeviceDesc &desc);
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

        // ---- Accessors for VulkanCommandList ----
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

        [[nodiscard]] inline VkBuffer_ &Buffer(BufferHandle h) noexcept
        {
            return _buffers.get(h.Index);
        }
        [[nodiscard]] inline VkTexture_ &Texture(TextureHandle h) noexcept
        {
            return _textures.get(h.Index);
        }
        [[nodiscard]] inline VkPipeline_ &Pipeline(PipelineHandle h) noexcept
        {
            return _pipelines.get(h.Index);
        }
        [[nodiscard]] inline VulkanBindlessHeap &Heap() noexcept
        {
            return _heap;
        }
        [[nodiscard]] inline VkAccelStruct_ &AccelStruct(AccelerationStructureHandle h) noexcept
        {
            return _accelStructs.get(h.Index);
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
            std::mutex mtx;
        };
        QueueInfo gfxQ, computeQ, transferQ;

        // Command pools (one per queue type)
        VkCommandPool gfxPool{VK_NULL_HANDLE};
        VkCommandPool computePool{VK_NULL_HANDLE};
        VkCommandPool xferPool{VK_NULL_HANDLE};
        std::mutex    gfxPoolMtx, computePoolMtx, xferPoolMtx;

        // Timeline semaphore
        VkSemaphore _timeline{VK_NULL_HANDLE};
        uint64_t    _timelineValue{0};
        std::mutex  _timelineMtx;

      private:
        VkInstance       _instance{VK_NULL_HANDLE};
        VkPhysicalDevice _physDev{VK_NULL_HANDLE};
        VkDevice         _device{VK_NULL_HANDLE};
        VmaAllocator     _allocator{};

        VkDebugUtilsMessengerEXT _debugMessenger{VK_NULL_HANDLE};
        VkPipelineLayout         _globalLayout{VK_NULL_HANDLE};
        uint32_t                 _maxPushConstantBytes{128};

        SlotPool<VkBuffer_>      _buffers{kMaxBuffers};
        SlotPool<VkTexture_>     _textures{kMaxTextures};
        SlotPool<VkSampler_>     _samplers{kMaxSamplers};
        SlotPool<VkPipeline_>    _pipelines{kMaxPipelines};
        SlotPool<VkAccelStruct_> _accelStructs{kMaxAccelerationStructures};
        VulkanBindlessHeap       _heap;

        // Set once in _pickPhysicalDevice; reflects whether
        // VK_KHR_acceleration_structure + VK_KHR_ray_query + their required
        // features are actually available on the selected physical device —
        // both are optional extensions here (unlike VK_EXT_descriptor_buffer,
        // which device selection requires), so a driver/device lacking them
        // still creates a working (non-ray-tracing) VulkanDevice.
        bool _raytracingSupported{false};

        std::string _adapterName;
        uint64_t    _videoMemoryBytes{0};

        // Init steps
        void _createInstance(const DeviceDesc &desc);
        void _pickPhysicalDevice();
        void _createLogicalDevice();
        void _loadExtensionFunctions();
        void _createAllocator();
        void _createCommandPools();
        void _createTimeline();
        void _createGlobalLayout();

        // Helpers
        [[nodiscard]] VkImageAspectFlags _aspectMask(Format f) const noexcept;
        [[nodiscard]] VkImageUsageFlags  _toImageUsage(TextureUsage u) const noexcept;
        [[nodiscard]] VkBufferUsageFlags _toBufUsage(BufferUsage u) const noexcept;
        [[nodiscard]] VmaMemoryUsage     _toVmaUsage(MemoryType m) const noexcept;
        [[nodiscard]] VkShaderModule     _makeShaderModule(std::span<const uint32_t> spirv);
        void                             _setDebugName(VkObjectType type, uint64_t handle, std::string_view name);
    };

} // namespace rhi::vulkan
