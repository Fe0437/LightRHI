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

    // ---- Acceleration-structure handles ----

    /** {brief} Opaque handle to a bottom-level or top-level acceleration structure. */
    struct AccelerationStructureHandle
    {
        uint32_t Index{kInvalidIndex}; ///< Opaque device-owned slot; do not manufacture indices.

        /** {brief} Reports whether the handle names a resource slot. */
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Index != kInvalidIndex;
        }
        /** {brief} Compares opaque handle identity. */
        [[nodiscard]] constexpr bool operator==(const AccelerationStructureHandle &) const noexcept = default;
    };

    // ---- Acceleration-structure construction ----

    /** {brief} Selects whether an acceleration structure contains triangles or instances. */
    enum class AccelerationStructureType : uint8_t
    {
        BottomLevel,
        TopLevel,
    };

    /** {brief} Modifies culling and opacity for one top-level acceleration-structure instance. */
    enum class AccelerationStructureInstanceFlags : uint32_t
    {
        None                          = 0,      ///< Uses the geometry's default culling and opacity.
        TriangleCullDisable           = 1 << 0, ///< Makes both triangle faces visible to ray queries.
        TriangleFrontCounterClockwise = 1 << 1, ///< Treats counter-clockwise triangles as front-facing.
        ForceOpaque                   = 1 << 2, ///< Treats all geometry in the instance as opaque.
        ForceNonOpaque                = 1 << 3, ///< Treats all geometry in the instance as non-opaque.
    };

    /** {brief} Combines acceleration-structure instance flags. */
    [[nodiscard]] constexpr AccelerationStructureInstanceFlags operator|(AccelerationStructureInstanceFlags a,
                                                                         AccelerationStructureInstanceFlags b) noexcept
    {
        return static_cast<AccelerationStructureInstanceFlags>(std::to_underlying(a) | std::to_underlying(b));
    }

    /** {brief} Places one already-built BLAS in a TLAS. */
    struct AccelerationStructureInstance
    {
        AccelerationStructureHandle Blas; ///< Live BLAS referenced by this instance.
        /** {brief} Row-major 3x4 object-to-world transform. */
        float Transform[3][4]{
            {1.F, 0.F, 0.F, 0.F},
            {0.F, 1.F, 0.F, 0.F},
            {0.F, 0.F, 1.F, 0.F},
        };
        uint32_t                           InstanceCustomIndex{0}; ///< Application value returned by ray queries.
        uint32_t                           InstanceMask{0xFF};     ///< Eight-bit visibility mask tested by ray queries.
        AccelerationStructureInstanceFlags Flags{
            AccelerationStructureInstanceFlags::None}; ///< Per-instance traversal flags.
    };

    /**
     * {brief} Describes the input used to size, allocate, and build one acceleration structure.
     * {note} Reuse the same descriptor values for the size query, creation, and recorded build.
     *
     * A bottom-level acceleration structure (BLAS) contains triangles read from
     * GPU-addressable vertex and optional index buffers. A top-level acceleration
     * structure (TLAS) contains transformed instances of already-built BLAS handles.
     *
     * \par Build sequence
     * ```cpp
     * auto desc = BlasFromTriangleBuffer(
     *     device->BufferAddress(vertices), sizeof(Vertex), vertexCount);
     * const auto sizes = device->QueryAccelerationStructureBuildSizes(desc);
     * const auto accelerationStructure = device->CreateAccelerationStructure(desc);
     * const auto scratch = device->CreateBuffer({
     *     .Size = sizes.BuildScratchSize,
     *     .Usage = BufferUsage::Storage | BufferUsage::DeviceAddress,
     * });
     *
     * cmd->BuildAccelerationStructure(accelerationStructure, desc, scratch);
     * cmd->Transition(accelerationStructure,
     *                 ResourceState::AccelerationStructureWrite,
     *                 ResourceState::AccelerationStructureRead);
     * cmd->FlushBarriers();
     * ```
     */
    struct AccelerationStructureDesc
    {
        AccelerationStructureType Type{
            AccelerationStructureType::BottomLevel}; ///< Selects the active field group below.

        // ---- BLAS fields (Type == BottomLevel) ----
        GpuAddress VertexBufferAddress{};        ///< Address of position data beginning with the first vertex.
        uint32_t   VertexStride{12};             ///< Byte distance between consecutive float3 positions.
        uint32_t   VertexCount{0};               ///< Number of accessible positions.
        GpuAddress IndexBufferAddress{};         ///< Optional index data; invalid selects non-indexed triangles.
        uint32_t   IndexCount{0};                ///< Number of accessible indices; ignored for non-indexed geometry.
        IndexType  IndexType{IndexType::Uint32}; ///< Width of each index.

        // ---- TLAS fields (Type == TopLevel) ----
        std::vector<AccelerationStructureInstance> Instances; ///< Instances copied for TLAS construction.

        bool             PreferFastTrace{true}; ///< Favors traversal speed over build speed when true.
        std::string_view DebugName;             ///< Optional diagnostic name, copied during creation.
    };

    /**
     * {brief} Returns a BLAS descriptor for a triangle list stored in GPU-addressable buffers.
     * {param indexBufferAddress} Leave invalid for non-indexed triangles.
     * {pre} Vertex and optional index buffers remain live until the build completes.
     */
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

    /**
     * {brief} Returns a TLAS descriptor that takes ownership of `instances`.
     * {pre} Every instance references an already-built BLAS that remains live through the TLAS build and use.
     */
    [[nodiscard]] inline AccelerationStructureDesc
    TlasFromInstances(std::vector<AccelerationStructureInstance> &&instances, std::string_view name = {})
    {
        return AccelerationStructureDesc{
            .Type      = AccelerationStructureType::TopLevel,
            .Instances = std::move(instances),
            .DebugName = name,
        };
    }

    // ---- Build requirements ----

    /** {brief} Storage requirements returned before an acceleration-structure build. */
    struct AccelerationStructureBuildSizes
    {
        uint64_t AccelerationStructureSize{0}; ///< Bytes reserved by CreateAccelerationStructure().
        uint64_t BuildScratchSize{0};          ///< Scratch bytes required for a from-scratch build.
        uint64_t UpdateScratchSize{0};         ///< Scratch bytes for an update; zero when updates are unavailable.
    };

    // ---- Barriers ----

    /** {brief} Orders access to one acceleration structure between build and traversal commands. */
    struct AccelerationStructureBarrier
    {
        AccelerationStructureHandle AccelerationStructure; ///< Resource whose access is ordered.
        ResourceState               Before;                ///< State of preceding access.
        ResourceState               After;                 ///< State required by following access.
    };

} // namespace rhi
