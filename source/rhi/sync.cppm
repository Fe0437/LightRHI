module;
#include <cstdint>

export module rhi:sync;
import :types;
import :handles;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // Subresource range — used in barriers to scope mip/layer access.
    // ---------------------------------------------------------------------------

    struct SubresourceRange
    {
        uint32_t BaseMip{0};
        uint32_t MipCount{~0U}; // ~0u = all remaining mips
        uint32_t BaseLayer{0};
        uint32_t LayerCount{~0U}; // ~0u = all remaining layers
    };

    // ---------------------------------------------------------------------------
    // Explicit resource barriers.
    //
    // Design note: we deliberately expose transitions. Callers are expected to
    // know what state their resources are in. The backend translates to:
    //   Vulkan  → VkImageMemoryBarrier2 / VkBufferMemoryBarrier2 (sync2)
    //   Metal   → MTLFence / MTLEvent or useResource:usage:
    //
    // We do NOT silently insert VK_IMAGE_LAYOUT_GENERAL to avoid thinking.
    // If you want a read-only texture, transition to ResourceState::ShaderRead.
    // If you want compute write, transition to ResourceState::UnorderedAccess.
    // ---------------------------------------------------------------------------

    struct TextureBarrier
    {
        TextureHandle    Texture;
        ResourceState    Before;
        ResourceState    After;
        SubresourceRange Range{}; // defaults to whole resource
    };

    struct BufferBarrier
    {
        BufferHandle  Buffer;
        ResourceState Before;
        ResourceState After;
        uint64_t      Offset{0};
        uint64_t      Size{~0ULL}; // ~0ull = whole buffer
    };

    // Global memory barrier — no specific resource; flushes all caches.
    struct MemoryBarrier
    {
        ResourceState Before;
        ResourceState After;
    };

    // ---------------------------------------------------------------------------
    // Timeline semaphore point.
    // IDevice::Submit() can signal/wait on these for GPU-GPU and CPU-GPU sync
    // without needing the caller to manage raw semaphore objects.
    // ---------------------------------------------------------------------------

    struct TimelinePoint
    {
        uint64_t                     Value{0};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Value != 0;
        }
    };

} // namespace rhi
