module;
#include <cstdint>

export module rhi:handles;

export namespace rhi
{

    inline constexpr uint32_t kInvalidIndex{~0U};

    // ---------------------------------------------------------------------------
    // Opaque, index-based resource handles.
    // The index is the slot in the backend's bindless heap / resource pool.
    // Shaders address resources through these indices directly.
    //
    // Example Slang shader usage:
    //   RWTexture2D<float4> textures[];
    //   textures[pc.outputTexture.Index][coord] = value;
    // ---------------------------------------------------------------------------

    struct BufferHandle
    {
        uint32_t                     Index{kInvalidIndex};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        [[nodiscard]] constexpr bool operator==(const BufferHandle &) const noexcept = default;
    };

    struct TextureHandle
    {
        uint32_t                     Index{kInvalidIndex};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        [[nodiscard]] constexpr bool operator==(const TextureHandle &) const noexcept = default;
    };

    struct SamplerHandle
    {
        uint32_t                     Index{kInvalidIndex};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        [[nodiscard]] constexpr bool operator==(const SamplerHandle &) const noexcept = default;
    };

    struct PipelineHandle
    {
        uint32_t                     Index{kInvalidIndex};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        [[nodiscard]] constexpr bool operator==(const PipelineHandle &) const noexcept = default;
    };

    // ---------------------------------------------------------------------------
    // GPU virtual address — buffer device address (Vulkan BDA / Metal GPU ptr).
    // Used to pass buffer pointers directly in push constants / root constants,
    // enabling GPU-driven rendering without binding per-draw.
    //
    //   struct DrawPacket {
    //       GpuAddress  vertices;
    //       GpuAddress  instances;
    //       TextureHandle albedo;
    //       SamplerHandle sampler;
    //       uint32_t      materialId;
    //   };
    // ---------------------------------------------------------------------------

    struct GpuAddress
    {
        uint64_t                     Address{0};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Address != 0;
        }
        [[nodiscard]] constexpr GpuAddress Offset(uint64_t bytes) const noexcept
        {
            return {.Address = Address + bytes};
        }
        [[nodiscard]] constexpr bool operator==(const GpuAddress &) const noexcept = default;
    };

    // ---------------------------------------------------------------------------
    // CPU-side fence handle.
    // Returned from IDevice::Submit(); poll or block with IDevice::WaitForFence().
    // ---------------------------------------------------------------------------

    struct FenceHandle
    {
        uint64_t                     Id{0};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Id != 0;
        }
        [[nodiscard]] constexpr bool operator==(const FenceHandle &) const noexcept = default;
    };

} // namespace rhi
