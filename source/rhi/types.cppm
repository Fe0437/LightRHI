module;
#include <cstdint>
#include <utility> // std::to_underlying

export module rhi:types;

export namespace rhi
{

    // ---- Pixel / texture formats ----

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

    [[nodiscard]] constexpr bool IsDepthFormat(Format f) noexcept
    {
        return f == Format::D16Unorm || f == Format::D32Float || f == Format::D24UnormS8Uint ||
               f == Format::D32FloatS8Uint;
    }

    [[nodiscard]] constexpr bool IsStencilFormat(Format f) noexcept
    {
        return f == Format::D24UnormS8Uint || f == Format::D32FloatS8Uint;
    }

    // ---- Texture dimensionality ----

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

    // ---- Resource state / layout ----
    // Explicit — callers must track and transition; we do NOT hide this.
    // Maps to VkImageLayout + VkAccessFlags on Vulkan, MTLResourceUsage on Metal.

    enum class ResourceState : uint32_t
    {
        Undefined                  = 0,
        VertexBuffer               = 1 << 0,
        IndexBuffer                = 1 << 1,
        ConstantBuffer             = 1 << 2,
        ShaderRead                 = 1 << 3, // SRV: sampled / read in shader
        UnorderedAccess            = 1 << 4, // UAV: read+write in shader
        RenderTarget               = 1 << 5,
        DepthRead                  = 1 << 6,
        DepthWrite                 = 1 << 7,
        TransferSrc                = 1 << 8,
        TransferDst                = 1 << 9,
        Present                    = 1 << 10,
        IndirectArgument           = 1 << 11,
        AccelerationStructureWrite = 1 << 12, // AS build/update output
        AccelerationStructureRead  = 1 << 13, // AS consumed by RayQuery in a shader
        // Aliases
        CopySrc = TransferSrc,
        CopyDst = TransferDst,
    };

    [[nodiscard]] constexpr ResourceState operator|(ResourceState a, ResourceState b) noexcept
    {
        return static_cast<ResourceState>(std::to_underlying(a) | std::to_underlying(b));
    }
    [[nodiscard]] constexpr ResourceState operator&(ResourceState a, ResourceState b) noexcept
    {
        return static_cast<ResourceState>(std::to_underlying(a) & std::to_underlying(b));
    }
    [[nodiscard]] constexpr bool HasState(ResourceState mask, ResourceState flag) noexcept
    {
        return (mask & flag) == flag;
    }

    // ---- Buffer usage flags ----

    enum class BufferUsage : uint32_t
    {
        None          = 0,
        Vertex        = 1 << 0,
        Index         = 1 << 1,
        Constant      = 1 << 2, // uniform / constant buffer
        Storage       = 1 << 3, // structured / SSBO
        IndirectArgs  = 1 << 4,
        TransferSrc   = 1 << 5,
        TransferDst   = 1 << 6,
        DeviceAddress = 1 << 7, // buffer device address (BDA) — GPU-driven
    };

    [[nodiscard]] constexpr BufferUsage operator|(BufferUsage a, BufferUsage b) noexcept
    {
        return static_cast<BufferUsage>(std::to_underlying(a) | std::to_underlying(b));
    }
    [[nodiscard]] constexpr BufferUsage operator&(BufferUsage a, BufferUsage b) noexcept
    {
        return static_cast<BufferUsage>(std::to_underlying(a) & std::to_underlying(b));
    }
    [[nodiscard]] constexpr bool HasUsage(BufferUsage mask, BufferUsage flag) noexcept
    {
        return (mask & flag) == flag;
    }

    // ---- Texture usage flags ----

    enum class TextureUsage : uint32_t
    {
        None         = 0,
        Sampled      = 1 << 0,
        Storage      = 1 << 1, // UAV / read-write image
        RenderTarget = 1 << 2,
        DepthStencil = 1 << 3,
        TransferSrc  = 1 << 4,
        TransferDst  = 1 << 5,
    };

    [[nodiscard]] constexpr TextureUsage operator|(TextureUsage a, TextureUsage b) noexcept
    {
        return static_cast<TextureUsage>(std::to_underlying(a) | std::to_underlying(b));
    }
    [[nodiscard]] constexpr TextureUsage operator&(TextureUsage a, TextureUsage b) noexcept
    {
        return static_cast<TextureUsage>(std::to_underlying(a) & std::to_underlying(b));
    }
    [[nodiscard]] constexpr bool HasUsage(TextureUsage mask, TextureUsage flag) noexcept
    {
        return (mask & flag) == flag;
    }

    // ---- Memory placement ----

    enum class MemoryType : uint8_t
    {
        GpuOnly,  // device-local, no CPU access
        CpuToGpu, // upload / staging — write-combined
        GpuToCpu, // readback — CPU read after GPU write
    };

    // ---- Queue types ----

    enum class QueueType : uint8_t
    {
        Graphics,
        Compute,
        Transfer,
    };

    // ---- Primitive / raster ----

    enum class IndexType : uint8_t
    {
        Uint16,
        Uint32
    };

    enum class PrimitiveTopology : uint8_t
    {
        TriangleList,
        TriangleStrip,
        LineList,
        LineStrip,
        PointList,
    };

    enum class CullMode : uint8_t
    {
        None,
        Front,
        Back
    };
    enum class FillMode : uint8_t
    {
        Solid,
        Wireframe
    };
    enum class FrontFace : uint8_t
    {
        CounterClockwise,
        Clockwise
    };

    // ---- Compare / blend ----

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

    enum class BlendOp : uint8_t
    {
        Add,
        Subtract,
        ReverseSubtract,
        Min,
        Max
    };

    // ---- Attachment ops ----

    enum class LoadOp : uint8_t
    {
        Load,
        Clear,
        DontCare
    };
    enum class StoreOp : uint8_t
    {
        Store,
        DontCare
    };

    // ---- Sampler ----

    enum class SamplerFilter : uint8_t
    {
        Nearest,
        Linear
    };
    enum class SamplerMipMode : uint8_t
    {
        Nearest,
        Linear
    };
    enum class SamplerAddressMode : uint8_t
    {
        Repeat,
        MirroredRepeat,
        ClampToEdge,
        ClampToBorder
    };
    enum class BorderColor : uint8_t
    {
        TransparentBlack,
        OpaqueBlack,
        OpaqueWhite
    };

    // ---- Shader stage flags ----

    enum class ShaderStage : uint32_t
    {
        None     = 0,
        Vertex   = 1 << 0,
        Fragment = 1 << 1,
        Compute  = 1 << 2,
        All      = Vertex | Fragment | Compute,
    };

    [[nodiscard]] constexpr ShaderStage operator|(ShaderStage a, ShaderStage b) noexcept
    {
        return static_cast<ShaderStage>(std::to_underlying(a) | std::to_underlying(b));
    }

    // ---- Shader artifact format ----
    // Identifies which backend-specific encoding a compiled shader artifact is
    // in. Slang is a tooling-layer concern (see tools/compile_shaders.py) —
    // the RHI only ever sees these already-compiled forms.

    enum class ShaderFormat : uint8_t
    {
        Spirv,     // Vulkan: SPIR-V words
        MslSource, // Metal:  MSL source text, compiled at runtime (dev builds)
        MetalLib,  // Metal:  pre-compiled .metallib bytes (shipping builds; not produced yet)
    };

    // ---- Geometry structs ----

    struct Extent3D
    {
        uint32_t Width{1};
        uint32_t Height{1};
        uint32_t Depth{1};
    };
    struct Extent2D
    {
        uint32_t Width{1};
        uint32_t Height{1};
    };
    struct Offset3D
    {
        int32_t X{0};
        int32_t Y{0};
        int32_t Z{0};
    };

    struct Viewport
    {
        float X{0.F};
        float Y{0.F};
        float Width{};
        float Height{};
        float MinDepth{0.F};
        float MaxDepth{1.F};
    };

    struct Scissor
    {
        int32_t  X{0};
        int32_t  Y{0};
        uint32_t Width{};
        uint32_t Height{};
    };

    struct ClearColor
    {
        float R{0.F};
        float G{0.F};
        float B{0.F};
        float A{1.F};
    };
    struct ClearDepthStencil
    {
        float   Depth{1.F};
        uint8_t Stencil{0};
    };

} // namespace rhi
