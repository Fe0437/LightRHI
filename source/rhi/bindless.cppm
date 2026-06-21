module;
#include <cstdint>

export module rhi:bindless;
import :types;
import :handles;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // BindlessHeap
    //
    // The global descriptor table — all bindless resource access flows through here.
    // Resources created via IDevice are automatically registered on creation and
    // deregistered on destruction. The slot index is the handle's .Index field.
    //
    // Backend mapping:
    //   Vulkan  → VK_EXT_descriptor_buffer, one large buffer of descriptors.
    //             Bound once per frame; the GPU walks it via descriptor indexing.
    //   Metal   → native GPU addresses/resource IDs in root or scene data;
    //             all indirectly reachable allocations live in one residency
    //             set attached to the MTL4 command queue.
    //
    // Vulkan exposes the descriptor buffer's address for backend plumbing.
    // Metal returns zero because native resource IDs need no descriptor table:
    //
    //   struct RootConstants { GpuAddress heap; uint32_t drawIdx; };
    //   cmd->SetPushConstants(RootConstants{ device.bindlessHeapAddress(), i });
    //
    // Slang shader example:
    //   struct PC { DescriptorHandle<Texture2D> texture; };
    //   Texture2D texture = pc.texture;
    //
    // Acceleration structures don't occupy a slot in this heap: they are
    // bindless through their own 8-byte GPU handle instead
    // (IDevice::AccelerationStructureAddress() → a
    // DescriptorHandle<RaytracingAccelerationStructure> in push constants,
    // used exactly like a buffer BDA pointer — see raytracing.cppm).
    // ---------------------------------------------------------------------------

    struct BindlessLimits
    {
        uint32_t MaxBuffers{1 << 20};  // 1M buffer slots
        uint32_t MaxTextures{1 << 20}; // 1M texture slots
        uint32_t MaxSamplers{2048};    // samplers are precious; keep small
    };

    class IBindlessHeap
    {
      public:
        virtual ~IBindlessHeap() = default;

        // Capacity
        [[nodiscard]] virtual uint32_t MaxBuffers() const noexcept  = 0;
        [[nodiscard]] virtual uint32_t MaxTextures() const noexcept = 0;
        [[nodiscard]] virtual uint32_t MaxSamplers() const noexcept = 0;

        // Vulkan descriptor-buffer address. Zero on Metal, where native
        // addresses/resource IDs plus the queue residency set are the heap.
        [[nodiscard]] virtual GpuAddress HeapAddress() const noexcept = 0;

        // Occupancy helpers (for debugging / profiling)
        [[nodiscard]] virtual uint32_t UsedBuffers() const noexcept  = 0;
        [[nodiscard]] virtual uint32_t UsedTextures() const noexcept = 0;
        [[nodiscard]] virtual uint32_t UsedSamplers() const noexcept = 0;
    };

} // namespace rhi
