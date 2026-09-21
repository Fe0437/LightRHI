/**
 * {file} handles.cppm
 * {brief} Defines opaque value handles for GPU resources and submissions.
 */
module;
#include <cstdint>

export module rhi:handles;

export namespace rhi
{
    // Every handle below is one public `uint32_t Index`, and the check below would have it made
    // private behind an accessor. It is refused here, once, for all of them.
    //
    // A struct keeps its public data members to stay an *aggregate*, which is what allows
    //
    //     return BufferHandle{.Index = idx};
    //
    // the form each backend uses to mint a handle - 11 sites across the two of them. A private
    // data member makes the type a non-aggregate and every one of those stops compiling.
    //
    // There is also nothing for an accessor to protect. The only rule a handle has is "is this
    // slot real", and `Valid()` already answers it; the index is meaningless outside the device
    // that issued it, which the member's own comment says. Hiding it would buy ceremony, not
    // an invariant.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes,cppcoreguidelines-non-private-member-variables-in-classes)

    /** {brief} Sentinel stored by default-constructed opaque resource handles. */
    inline constexpr uint32_t kInvalidIndex{~0U};

    // ---- Resource handles ----

    /**
     * {brief} Opaque handle to a heap of device memory owned by an IDevice.
     *
     * A memory heap hands out bytes; the bindless heap reported by IBindlessHeap hands out
     * shader-visible descriptor slots. Both are pools the caller draws from, so both are heaps,
     * and the qualifier is the whole difference.
     */
    struct MemoryHeapHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this handle names a memory heap. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares heap identity. */
        [[nodiscard]] constexpr bool operator==(const MemoryHeapHandle &) const noexcept = default;
    };

    /** {brief} Opaque handle to a buffer owned by an IDevice. */
    struct BufferHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this value names a resource slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const BufferHandle &) const noexcept = default;
    };

    /** {brief} Opaque handle to a texture owned by an IDevice. */
    struct TextureHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this value names a resource slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const TextureHandle &) const noexcept = default;
    };

    /**
     * {brief} Opaque handle to an external texture provider registered with an IDevice.
     *
     * Presenting names its provider the way drawing names a buffer or a texture: by an opaque slot
     * the device resolves. A recorded present therefore carries a value, not a pointer the backend
     * has to know the real type of.
     */
    struct ExternalTextureProviderHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this value names an external texture provider slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const ExternalTextureProviderHandle &) const noexcept = default;
    };

    /** {brief} Opaque handle to a sampler owned by an IDevice. */
    struct SamplerHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this value names a resource slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const SamplerHandle &) const noexcept = default;
    };

    /** {brief} Opaque handle to a graphics or compute pipeline owned by an IDevice. */
    struct PipelineHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this value names a resource slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const PipelineHandle &) const noexcept = default;
    };

    /** {brief} Opaque handle to a timestamp-query pool owned by an IDevice. */
    struct TimestampQueryPoolHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether this value names a query-pool slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const TimestampQueryPoolHandle &) const noexcept = default;
    };

    // ---- Shader-visible addresses ----

    /**
     * {brief} Shader-visible address or bindless resource value returned by IDevice.
     *
     * Pass addresses through push constants or GPU buffers without exposing a
     * backend object to application code.
     *
     * ```cpp
     * struct DrawPacket
     * {
     *     GpuAddress Vertices;
     *     GpuAddress Instances;
     *     GpuAddress Albedo;
     *     uint32_t MaterialIndex;
     * };
     *
     * DrawPacket packet{
     *     .Vertices = device->BufferAddress(vertices),
     *     .Instances = device->BufferAddress(instances),
     *     .Albedo = device->TextureAddress(albedo),
     * };
     * ```
     */
    struct GpuAddress
    {
        uint64_t Address{0}; ///< Opaque shader-visible value; zero is invalid.

        /** {brief} Reports whether this address may be passed to GPU code. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Address != 0;
        }
        /**
         * {brief} Returns an address `bytes` after this buffer address.
         * {pre} This is a valid address and the offset stays within the same live allocation.
         */
        [[nodiscard]] constexpr GpuAddress Offset(uint64_t bytes) const noexcept
        {
            return {.Address = Address + bytes};
        }
        /** {brief} Compares address values. */
        [[nodiscard]] constexpr bool operator==(const GpuAddress &) const noexcept = default;
    };

    // ---- Submission fences ----

    /** {brief} Identifies completion of one IDevice submission. */
    struct FenceHandle
    {
        uint64_t Id{0}; ///< Device-selected timeline value; zero is invalid.

        /** {brief} Reports whether this value can be passed to a fence operation. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Id != 0;
        }
        /** {brief} Compares submission identity. */
        [[nodiscard]] constexpr bool operator==(const FenceHandle &) const noexcept = default;
    };

    // NOLINTEND(misc-non-private-member-variables-in-classes,cppcoreguidelines-non-private-member-variables-in-classes)
} // namespace rhi
