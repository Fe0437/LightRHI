module;
#include <cstdint>
#include <string_view>
#include <utility> // std::to_underlying
#include <vector>

export module rhi:raytracing;
import :types;
import :handles;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // AccelerationStructureHandle — opaque, index-based, same shape as
    // BufferHandle/TextureHandle. Backends allocate it from their own slot pool.
    // ---------------------------------------------------------------------------

    struct AccelerationStructureHandle
    {
        uint32_t                     Index{kInvalidIndex};
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        [[nodiscard]] constexpr bool operator==(const AccelerationStructureHandle &) const noexcept = default;
    };

    // ---------------------------------------------------------------------------
    // Acceleration structure construction.
    //
    // Two levels, one Desc type discriminated by Type:
    //   BottomLevel (BLAS) — geometry built directly from a triangle buffer,
    //     addressed by BDA (GpuAddress) exactly like everywhere else in LightRHI.
    //     Positions are tightly packed float3 at VertexStride-byte intervals;
    //     IndexBufferAddress.Valid() == false means non-indexed.
    //   TopLevel (TLAS) — built from a list of instances, each referencing an
    //     already-built BLAS with a per-instance transform.
    // ---------------------------------------------------------------------------

    enum class AccelerationStructureType : uint8_t
    {
        BottomLevel,
        TopLevel,
    };

    // Per-instance flags. Bit-compatible with VkGeometryInstanceFlagBitsKHR /
    // MTL::AccelerationStructureInstanceOptions (both use the same bit
    // positions for these four flags), so backends pass this straight
    // through with a single static_cast — no per-backend translation table.
    enum class AccelerationStructureInstanceFlags : uint32_t
    {
        None                          = 0,
        TriangleCullDisable           = 1 << 0,
        TriangleFrontCounterClockwise = 1 << 1,
        ForceOpaque                   = 1 << 2,
        ForceNonOpaque                = 1 << 3,
    };

    [[nodiscard]] constexpr AccelerationStructureInstanceFlags operator|(AccelerationStructureInstanceFlags a,
                                                                         AccelerationStructureInstanceFlags b) noexcept
    {
        return static_cast<AccelerationStructureInstanceFlags>(std::to_underlying(a) | std::to_underlying(b));
    }

    // Row-major 3x4 world transform (Vulkan VkTransformMatrixKHR / Metal MTL::PackedFloat4x3 layout).
    struct AccelerationStructureInstance
    {
        AccelerationStructureHandle Blas;
        float                       Transform[3][4]{
            {1.F, 0.F, 0.F, 0.F},
            {0.F, 1.F, 0.F, 0.F},
            {0.F, 0.F, 1.F, 0.F},
        };
        uint32_t                           InstanceCustomIndex{0};
        uint32_t                           InstanceMask{0xFF};
        AccelerationStructureInstanceFlags Flags{AccelerationStructureInstanceFlags::None};
    };

    struct AccelerationStructureDesc
    {
        AccelerationStructureType Type{AccelerationStructureType::BottomLevel};

        // ---- BLAS fields (Type == BottomLevel) ----
        GpuAddress VertexBufferAddress{};
        uint32_t   VertexStride{12}; // bytes; default = tightly packed float3
        uint32_t   VertexCount{0};
        GpuAddress IndexBufferAddress{}; // optional; invalid() => non-indexed triangle list
        uint32_t   IndexCount{0};
        IndexType  IndexType{IndexType::Uint32};

        // ---- TLAS fields (Type == TopLevel) ----
        std::vector<AccelerationStructureInstance> Instances;

        bool             PreferFastTrace{true}; // build-quality hint (vs. PreferFastBuild)
        std::string_view DebugName;
    };

    // Convenience constructors, in the style of Texture2D() / LinearRepeat().

    [[nodiscard]] inline AccelerationStructureDesc
    BlasFromTriangleBuffer(GpuAddress vertexBufferAddress, uint32_t vertexStride, uint32_t vertexCount,
                           GpuAddress indexBufferAddress = {}, uint32_t indexCount = 0,
                           IndexType indexType = IndexType::Uint32, std::string_view name = {})
    {
        return AccelerationStructureDesc{
            .Type                = AccelerationStructureType::BottomLevel,
            .VertexBufferAddress = vertexBufferAddress,
            .VertexStride        = vertexStride,
            .VertexCount         = vertexCount,
            .IndexBufferAddress  = indexBufferAddress,
            .IndexCount          = indexCount,
            .IndexType           = indexType,
            .DebugName           = name,
        };
    }

    [[nodiscard]] inline AccelerationStructureDesc
    TlasFromInstances(std::vector<AccelerationStructureInstance> &&instances, std::string_view name = {})
    {
        return AccelerationStructureDesc{
            .Type      = AccelerationStructureType::TopLevel,
            .Instances = std::move(instances),
            .DebugName = name,
        };
    }

    // ---------------------------------------------------------------------------
    // Build-size query result. Mirrors vkGetAccelerationStructureBuildSizesKHR:
    // the caller sizes both the AS backing Buffer (implicitly, via
    // CreateAccelerationStructure) and an explicit scratch buffer before
    // recording a build.
    // ---------------------------------------------------------------------------

    struct AccelerationStructureBuildSizes
    {
        uint64_t AccelerationStructureSize{0}; // bytes required for the AS itself
        uint64_t BuildScratchSize{0};          // scratch bytes required for a from-scratch build
        uint64_t UpdateScratchSize{0};         // scratch bytes required for an in-place update (0 if unsupported)
    };

    // ---------------------------------------------------------------------------
    // Barrier — same shape as TextureBarrier/BufferBarrier (see sync.cppm),
    // scoped to a single acceleration structure. Use with
    // ICommandList::Transition() before the AS is consumed by a later
    // ray-query compute dispatch.
    // ---------------------------------------------------------------------------

    struct AccelerationStructureBarrier
    {
        AccelerationStructureHandle AccelerationStructure;
        ResourceState               Before;
        ResourceState               After;
    };

} // namespace rhi
