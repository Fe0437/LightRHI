/**
 * {file} types.cppm
 * {brief} Defines backend-neutral GPU value types, flags, and errors.
 */
module;
#include <cstdint>
#include <utility> // std::to_underlying

export module rhi:types;

export namespace rhi
{

    // ---- Pixel and texture formats ----

    /**
     * {brief} Pixel, texture, attachment, and typed-buffer formats accepted by descriptors.
     *
     * Names encode channel order, bits per channel, numeric interpretation, and
     * optional sRGB transfer. Use Undefined only where an optional format is allowed.
     */
    // Part of this library's published interface: the width is visible in every struct that
    // carries one, so narrowing it would change layouts a consumer has already compiled against.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class Format : uint32_t
    {
        Undefined = 0,
        // 8-bit normalised
        R8Unorm,
        R8Snorm,
        R8Uint,
        R8Sint,
        RG8Unorm,
        RG8Snorm,
        RG8Uint,
        RG8Sint,
        RGBA8Unorm,
        RGBA8Snorm,
        RGBA8Uint,
        RGBA8Sint,
        RGBA8Srgb,
        BGRA8Unorm,
        BGRA8Srgb,
        // 16-bit float / int
        R16Float,
        RG16Float,
        RGBA16Float,
        R16Uint,
        R16Sint,
        RG16Uint,
        RG16Sint,
        // 32-bit float / int
        R32Float,
        RG32Float,
        RGBA32Float,
        R32Uint,
        R32Sint,
        RG32Uint,
        RG32Sint,
        RGBA32Uint,
        RGBA32Sint,
        // Packed
        RGB10A2Unorm,
        RGB10A2Uint,
        R11G11B10Float,
        RGB9E5Float,
        // Depth / stencil
        D16Unorm,
        D32Float,
        D24UnormS8Uint,
        D32FloatS8Uint,
        // Block-compressed
        BC1Unorm,
        BC1Srgb,
        BC2Unorm,
        BC2Srgb,
        BC3Unorm,
        BC3Srgb,
        BC4Unorm,
        BC4Snorm,
        BC5Unorm,
        BC5Snorm,
        BC6HUfloat,
        BC6HSfloat,
        BC7Unorm,
        BC7Srgb,
    };

    /** {brief} Reports whether `format` contains a depth component. */
    [[nodiscard]] constexpr bool IsDepthFormat(Format f) noexcept
    {
        return f == Format::D16Unorm || f == Format::D32Float || f == Format::D24UnormS8Uint ||
               f == Format::D32FloatS8Uint;
    }

    /** {brief} Reports whether `format` contains a stencil component. */
    [[nodiscard]] constexpr bool IsStencilFormat(Format f) noexcept
    {
        return f == Format::D24UnormS8Uint || f == Format::D32FloatS8Uint;
    }

    // ---- Texture dimensionality ----

    /** {brief} Selects the coordinate and array shape exposed for a texture. */
    enum class TextureDimension : uint8_t
    {
        Tex1D,
        Tex2D,
        Tex3D,
        TexCube,
        Tex1DArray,
        Tex2DArray,
        TexCubeArray,
    };

    // ---- Resource states ----

    /**
     * {brief} Describes the kind of GPU access immediately before or after an explicit transition.
     * {note} Callers track states and record ICommandList::Transition() when access changes.
     */
    // Bit flags. The width is chosen for the flags this may still gain, not for the ones it
    // has today, so shrinking it to fit the current set would cap the type.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class ResourceState : uint32_t
    {
        Undefined                  = 0,           ///< Contents and previous access need not be preserved.
        VertexBuffer               = 1U << 0U,      ///< Read as vertex input.
        IndexBuffer                = 1U << 1U,      ///< Read as index input.
        ConstantBuffer             = 1U << 2U,      ///< Read as constant or uniform data.
        ShaderRead                 = 1U << 3U,      ///< Read-only shader access or texture sampling.
        UnorderedAccess            = 1U << 4U,      ///< Shader read/write access.
        RenderTarget               = 1U << 5U,      ///< Color-attachment write access.
        DepthRead                  = 1U << 6U,      ///< Read-only depth/stencil attachment access.
        DepthWrite                 = 1U << 7U,      ///< Writable depth/stencil attachment access.
        TransferSrc                = 1U << 8U,      ///< Source of a copy or blit operation.
        TransferDst                = 1U << 9U,      ///< Destination of a copy, blit, clear, or fill operation.
        Present                    = 1U << 10U,     ///< Ready for presentation by a presentation owner.
        IndirectArgument           = 1U << 11U,     ///< Read as draw or dispatch arguments.
        AccelerationStructureWrite = 1U << 12U,     ///< Written by an acceleration-structure build.
        AccelerationStructureRead  = 1U << 13U,     ///< Traversed by a shader ray query or TLAS build.
        CopySrc                    = TransferSrc, ///< Alias for TransferSrc.
        CopyDst                    = TransferDst, ///< Alias for TransferDst.
    };

    /** {brief} Combines resource-state flags for accesses that are valid concurrently. */
    [[nodiscard]] constexpr ResourceState operator|(ResourceState a, ResourceState b) noexcept
    {
        return static_cast<ResourceState>(std::to_underlying(a) | std::to_underlying(b));
    }
    /** {brief} Intersects two resource-state flag sets. */
    [[nodiscard]] constexpr ResourceState operator&(ResourceState a, ResourceState b) noexcept
    {
        return static_cast<ResourceState>(std::to_underlying(a) & std::to_underlying(b));
    }
    /** {brief} Reports whether all bits in `flag` are present in `mask`. */
    [[nodiscard]] constexpr bool HasState(ResourceState mask, ResourceState flag) noexcept
    {
        return (mask & flag) == flag;
    }

    // ---- Buffer usage ----

    /** {brief} Declares every operation a buffer may participate in during its lifetime. */
    // Bit flags. The width is chosen for the flags this may still gain, not for the ones it
    // has today, so shrinking it to fit the current set would cap the type.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class BufferUsage : uint32_t
    {
        None          = 0,      ///< No GPU operation is declared.
        Vertex        = 1U << 0U, ///< May be bound as vertex input.
        Index         = 1U << 1U, ///< May be bound as index input.
        Constant      = 1U << 2U, ///< May be read as constant or uniform data.
        Storage       = 1U << 3U, ///< May be read or written as shader storage.
        IndirectArgs  = 1U << 4U, ///< May provide draw or dispatch arguments.
        TransferSrc   = 1U << 5U, ///< May be the source of copy operations.
        TransferDst   = 1U << 6U, ///< May be the destination of copy or fill operations.
        DeviceAddress = 1U << 7U, ///< May expose a shader-visible address through BufferAddress().
    };

    /** {brief} Combines buffer-usage flags for a descriptor. */
    [[nodiscard]] constexpr BufferUsage operator|(BufferUsage a, BufferUsage b) noexcept
    {
        return static_cast<BufferUsage>(std::to_underlying(a) | std::to_underlying(b));
    }
    /** {brief} Intersects two buffer-usage flag sets. */
    [[nodiscard]] constexpr BufferUsage operator&(BufferUsage a, BufferUsage b) noexcept
    {
        return static_cast<BufferUsage>(std::to_underlying(a) & std::to_underlying(b));
    }
    /** {brief} Reports whether all bits in `flag` are present in `mask`. */
    [[nodiscard]] constexpr bool HasUsage(BufferUsage mask, BufferUsage flag) noexcept
    {
        return (mask & flag) == flag;
    }

    // ---- Texture usage ----

    /** {brief} Declares every operation a texture may participate in during its lifetime. */
    // Bit flags. The width is chosen for the flags this may still gain, not for the ones it
    // has today, so shrinking it to fit the current set would cap the type.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class TextureUsage : uint32_t
    {
        None    = 0,      ///< No GPU operation is declared.
        Sampled = 1U << 0U, ///< May be sampled or read by shaders.
        // Declares the intent on the resource, which every backend honours. Reaching a texture as
        // writable storage from a shader has no path in this API yet: shaders address buffers by
        // GPU address and textures as sampled bindless handles, so a compute kernel that produces
        // pixels writes them to a buffer. Do not read this flag as a promise of a write path.
        Storage      = 1U << 1U, ///< May be read or written as shader storage.
        RenderTarget = 1U << 2U, ///< May be used as a color attachment.
        DepthStencil = 1U << 3U, ///< May be used as a depth/stencil attachment.
        TransferSrc  = 1U << 4U, ///< May be the source of copy operations.
        TransferDst  = 1U << 5U, ///< May be the destination of copy, upload, or clear operations.
    };

    /** {brief} Combines texture-usage flags for a descriptor. */
    [[nodiscard]] constexpr TextureUsage operator|(TextureUsage a, TextureUsage b) noexcept
    {
        return static_cast<TextureUsage>(std::to_underlying(a) | std::to_underlying(b));
    }
    /** {brief} Intersects two texture-usage flag sets. */
    [[nodiscard]] constexpr TextureUsage operator&(TextureUsage a, TextureUsage b) noexcept
    {
        return static_cast<TextureUsage>(std::to_underlying(a) & std::to_underlying(b));
    }
    /** {brief} Reports whether all bits in `flag` are present in `mask`. */
    [[nodiscard]] constexpr bool HasUsage(TextureUsage mask, TextureUsage flag) noexcept
    {
        return (mask & flag) == flag;
    }

    // ---- Memory placement ----

    /** {brief} Selects the CPU/GPU visibility required from a buffer allocation. */
    enum class MemoryType : uint8_t
    {
        GpuOnly,  ///< GPU access only; update through copy commands or upload helpers.
        CpuToGpu, ///< CPU-writable memory for staging or frequently updated data.
        GpuToCpu, ///< CPU-readable memory for results after GPU completion.
    };

    // ---- Queues ----

    /** {brief} Selects the capabilities required by a command list or submission. */
    enum class QueueType : uint8_t
    {
        Graphics, ///< Supports drawing, compute, copies, and barriers.
        Compute,  ///< Supports compute, copies, and barriers.
        Transfer, ///< Supports copies and transfer-related barriers.
    };

    // ---- Primitive assembly and rasterization ----

    /** {brief} Selects the integer width of indices read from an index buffer. */
    enum class IndexType : uint8_t
    {
        Uint16,
        Uint32
    };

    /** {brief} Selects how vertex or index sequences form primitives. */
    enum class PrimitiveTopology : uint8_t
    {
        TriangleList,
        TriangleStrip,
        LineList,
        LineStrip,
        PointList,
    };

    /** {brief} Selects which triangle faces are discarded before rasterization. */
    enum class CullMode : uint8_t
    {
        None,
        Front,
        Back
    };
    /** {brief} Selects solid or edge-only triangle rasterization. */
    enum class FillMode : uint8_t
    {
        Solid,
        Wireframe
    };
    /** {brief} Selects the winding order considered front-facing. */
    enum class FrontFace : uint8_t
    {
        CounterClockwise,
        Clockwise
    };

    // ---- Comparison and blending ----

    /** {brief} Comparison function used by depth tests and comparison samplers. */
    enum class CompareOp : uint8_t
    {
        Never,
        Less,
        Equal,
        LessEqual,
        Greater,
        NotEqual,
        GreaterEqual,
        Always,
    };

    /** {brief} Selects a source or destination term in color blending. */
    enum class BlendFactor : uint8_t
    {
        Zero,
        One,
        SrcColor,
        OneMinusSrcColor,
        DstColor,
        OneMinusDstColor,
        SrcAlpha,
        OneMinusSrcAlpha,
        DstAlpha,
        OneMinusDstAlpha,
        ConstantColor,
        OneMinusConstantColor,
        SrcAlphaSaturate,
    };

    /** {brief} Operation combining the factored source and destination blend values. */
    enum class BlendOp : uint8_t
    {
        Add,
        Subtract,
        ReverseSubtract,
        Min,
        Max
    };

    // ---- Attachment operations ----

    /** {brief} Selects how an attachment's previous contents are treated at BeginRendering(). */
    enum class LoadOp : uint8_t
    {
        Load,    ///< Preserves and exposes existing contents.
        Clear,   ///< Initializes the attachment from its clear value.
        DontCare ///< Existing contents need not be preserved.
    };

    /** {brief} Selects whether attachment contents are preserved after EndRendering(). */
    enum class StoreOp : uint8_t
    {
        Store,   ///< Preserves rendered contents for later use.
        DontCare ///< Contents need not be preserved.
    };

    // ---- Sampling ----

    /** {brief} Selects nearest-neighbor or linear sampling within a mip level. */
    enum class SamplerFilter : uint8_t
    {
        Nearest,
        Linear
    };
    /** {brief} Selects nearest-neighbor or linear selection between mip levels. */
    enum class SamplerMipMode : uint8_t
    {
        Nearest,
        Linear
    };
    /** {brief} Selects how normalized texture coordinates outside `[0, 1]` are resolved. */
    enum class SamplerAddressMode : uint8_t
    {
        Repeat,
        MirroredRepeat,
        ClampToEdge,
        ClampToBorder
    };
    /** {brief} Selects the fixed value returned by ClampToBorder addressing. */
    enum class BorderColor : uint8_t
    {
        TransparentBlack,
        OpaqueBlack,
        OpaqueWhite
    };

    // ---- Shader stages and artifacts ----

    /** {brief} Identifies shader stages for artifact metadata and visibility. */
    // Bit flags. The width is chosen for the flags this may still gain, not for the ones it
    // has today, so shrinking it to fit the current set would cap the type.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class ShaderStage : uint32_t
    {
        None     = 0,
        Vertex   = 1U << 0U,
        Fragment = 1U << 1U,
        Compute  = 1U << 2U,
        All      = Vertex | Fragment | Compute,
    };

    /** {brief} Combines shader-stage flags. */
    [[nodiscard]] constexpr ShaderStage operator|(ShaderStage a, ShaderStage b) noexcept
    {
        return static_cast<ShaderStage>(std::to_underlying(a) | std::to_underlying(b));
    }

    /** {brief} Identifies the encoding of an already-compiled shader artifact. */
    enum class ShaderFormat : uint8_t
    {
        Spirv,     ///< SPIR-V words for a Vulkan device.
        MslSource, ///< Metal Shading Language source accepted by a Metal device.
        MetalLib,  ///< Precompiled Metal library bytes accepted by a Metal device.
    };

    // ---- Geometry and clear values ----

    /** {brief} Unsigned three-dimensional extent used for textures, copies, and thread groups. */
    struct Extent3D
    {
        uint32_t Width{1};  ///< Extent along X.
        uint32_t Height{1}; ///< Extent along Y.
        uint32_t Depth{1};  ///< Extent along Z.
    };

    /** {brief} Unsigned two-dimensional extent used for render areas. */
    struct Extent2D
    {
        uint32_t Width{1};  ///< Extent along X.
        uint32_t Height{1}; ///< Extent along Y.
    };

    /** {brief} Signed three-dimensional offset used as a texture-copy origin. */
    struct Offset3D
    {
        int32_t X{0}; ///< Offset along X.
        int32_t Y{0}; ///< Offset along Y.
        int32_t Z{0}; ///< Offset along Z.
    };

    /** {brief} Defines the rasterized rectangle and post-projection depth range. */
    struct Viewport
    {
        float X{0.F};        ///< Left origin in framebuffer coordinates.
        float Y{0.F};        ///< Top origin in framebuffer coordinates.
        float Width{};       ///< Width in pixels.
        float Height{};      ///< Height in pixels.
        float MinDepth{0.F}; ///< Minimum mapped depth.
        float MaxDepth{1.F}; ///< Maximum mapped depth.
    };

    /** {brief} Defines the integer framebuffer rectangle that may be rasterized. */
    struct Scissor
    {
        int32_t  X{0};     ///< Left origin in pixels.
        int32_t  Y{0};     ///< Top origin in pixels.
        uint32_t Width{};  ///< Width in pixels.
        uint32_t Height{}; ///< Height in pixels.
    };

    /** {brief} Floating-point RGBA value used to clear color textures and attachments. */
    struct ClearColor
    {
        float R{0.F}; ///< Red component.
        float G{0.F}; ///< Green component.
        float B{0.F}; ///< Blue component.
        float A{1.F}; ///< Alpha component.
    };

    /** {brief} Depth and stencil values used to clear depth/stencil textures and attachments. */
    struct ClearDepthStencil
    {
        float   Depth{1.F}; ///< Depth clear value.
        uint8_t Stencil{0}; ///< Stencil clear value.
    };

} // namespace rhi
