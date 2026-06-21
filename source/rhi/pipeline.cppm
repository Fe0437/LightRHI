module;
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

export module rhi:pipeline;
import :types;
import :handles;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // Shader bytecode alternatives — exactly one is active in ShaderDesc.
    //
    // Vulkan  — SpirvBytecode     (Slang → SPIR-V)
    // Metal   — MetalLibBytecode  (Slang → MSL → xcrun metal/metallib → .metallib)
    //           MslSource         (compiled at runtime; dev builds / tests)
    //
    // std::monostate = "no shader" (e.g. optional VS/FS stages, or default-
    // constructed ShaderDesc when the caller has not yet assigned a source).
    // ---------------------------------------------------------------------------

    struct SpirvBytecode
    {
        std::span<const uint32_t> Words;
    }; // Vulkan: SPIR-V words
    struct MetalLibBytecode
    {
        std::span<const uint8_t> Bytes;
    }; // Metal:  pre-compiled .metallib
    struct MslSource
    {
        std::string_view Source;
    }; // Metal:  MSL source text

    /// Discriminated union of all supported shader bytecode formats.
    /// Default-constructed value is std::monostate ("not present").
    using ShaderBytecode = std::variant<std::monostate, SpirvBytecode, MetalLibBytecode, MslSource>;

    struct ShaderDesc
    {
        ShaderBytecode   Bytecode; // std::monostate = not present
        std::string_view EntryPoint{"main"};
    };

    // ---------------------------------------------------------------------------
    // Backend-neutral view onto a compiled shader artifact (a .spv or .metal
    // file produced by tools/compile_shaders.py, or any other loader).
    //
    // The RHI core never invokes Slang or any shader compiler — it only knows
    // how to turn already-compiled bytes tagged with a ShaderFormat into the
    // backend-specific ShaderDesc fields above. Ownership of Data stays with
    // the caller for the lifetime of the resulting ShaderDesc.
    // ---------------------------------------------------------------------------
    struct ShaderArtifactView
    {
        ShaderFormat               Format{ShaderFormat::Spirv};
        ShaderStage                Stage{ShaderStage::None};
        std::string_view           EntryPoint{"main"};
        std::span<const std::byte> Data{};
    };

    // Errors ToShaderDesc can report. No exceptions in public API — see
    // API_GUIDELINES.md.
    enum class ShaderArtifactError
    {
        SpirvSizeNotWordAligned, // Data.size() is not a multiple of 4
    };

    [[nodiscard]] inline std::expected<ShaderDesc, ShaderArtifactError> ToShaderDesc(const ShaderArtifactView &artifact)
    {
        ShaderDesc desc{.EntryPoint = artifact.EntryPoint};
        switch (artifact.Format)
        {
            case ShaderFormat::Spirv:
                if (artifact.Data.size() % 4 != 0)
                {
                    return std::unexpected(ShaderArtifactError::SpirvSizeNotWordAligned);
                }
                desc.Bytecode = SpirvBytecode{
                    .Words = std::span<const uint32_t>{reinterpret_cast<const uint32_t *>(artifact.Data.data()),
                                                       artifact.Data.size() / 4}};
                break;
            case ShaderFormat::MslSource:
                desc.Bytecode =
                    MslSource{.Source = std::string_view{reinterpret_cast<const char *>(artifact.Data.data()),
                                                         artifact.Data.size()}};
                break;
            case ShaderFormat::MetalLib:
                desc.Bytecode = MetalLibBytecode{
                    .Bytes = std::span<const uint8_t>{reinterpret_cast<const uint8_t *>(artifact.Data.data()),
                                                      artifact.Data.size()}};
                break;
        }
        return desc;
    }

    // ---------------------------------------------------------------------------
    // Per-attachment blend state
    // ---------------------------------------------------------------------------

    struct BlendState
    {
        bool        Enable{false};
        BlendFactor SrcColor{BlendFactor::One};
        BlendFactor DstColor{BlendFactor::Zero};
        BlendOp     ColorOp{BlendOp::Add};
        BlendFactor SrcAlpha{BlendFactor::One};
        BlendFactor DstAlpha{BlendFactor::Zero};
        BlendOp     AlphaOp{BlendOp::Add};
        uint8_t     WriteMask{0xF}; // RGBA channel mask
    };

    // Common blend presets
    [[nodiscard]] inline BlendState BlendDisabled()
    {
        return BlendState{.Enable = false};
    }

    [[nodiscard]] inline BlendState BlendAlphaPremultiplied()
    {
        return BlendState{
            .Enable   = true,
            .SrcColor = BlendFactor::One,
            .DstColor = BlendFactor::OneMinusSrcAlpha,
            .SrcAlpha = BlendFactor::One,
            .DstAlpha = BlendFactor::OneMinusSrcAlpha,
        };
    }

    [[nodiscard]] inline BlendState BlendAlphaTraditional()
    {
        return BlendState{
            .Enable   = true,
            .SrcColor = BlendFactor::SrcAlpha,
            .DstColor = BlendFactor::OneMinusSrcAlpha,
            .SrcAlpha = BlendFactor::One,
            .DstAlpha = BlendFactor::Zero,
        };
    }

    // ---------------------------------------------------------------------------
    // Depth / stencil state
    // ---------------------------------------------------------------------------

    struct DepthStencilState
    {
        bool      DepthTest{true};
        bool      DepthWrite{true};
        CompareOp DepthOp{CompareOp::Less};
        bool      StencilTest{false};
        // Stencil state intentionally minimal until needed (YAGNI)
    };

    [[nodiscard]] inline DepthStencilState DepthReadWrite(CompareOp op = CompareOp::Less)
    {
        return DepthStencilState{.DepthTest = true, .DepthWrite = true, .DepthOp = op};
    }

    [[nodiscard]] inline DepthStencilState DepthReadOnly(CompareOp op = CompareOp::Less)
    {
        return DepthStencilState{.DepthTest = true, .DepthWrite = false, .DepthOp = op};
    }

    [[nodiscard]] inline DepthStencilState DepthDisabled()
    {
        return DepthStencilState{.DepthTest = false, .DepthWrite = false};
    }

    // ---------------------------------------------------------------------------
    // Rasterizer state
    // ---------------------------------------------------------------------------

    struct RasterizerState
    {
        CullMode  CullMode{CullMode::Back};
        FillMode  FillMode{FillMode::Solid};
        FrontFace FrontFace{FrontFace::CounterClockwise};
        float     DepthBiasConstant{0.F};
        float     DepthBiasSlope{0.F};
        bool      DepthClamp{false};
        bool      ConservativeRaster{false}; // future use
    };

    // ---------------------------------------------------------------------------
    // Graphics pipeline descriptor
    // ---------------------------------------------------------------------------

    struct GraphicsPipelineDesc
    {
        ShaderDesc VertexShader;
        ShaderDesc FragmentShader;

        PrimitiveTopology Topology{PrimitiveTopology::TriangleList};
        RasterizerState   Rasterizer{};
        DepthStencilState DepthStencil{};

        // One entry per color attachment; empty = no color output
        std::vector<BlendState> ColorBlend{};

        // Attachment formats — must match the RenderingDesc used at draw time.
        // Required by VK_KHR_dynamic_rendering / MTLRenderPipelineDescriptor.
        std::vector<Format> ColorFormats{};
        Format              DepthFormat{Format::Undefined};
        uint32_t            SampleCount{1};

        // Push constant range visible to all stages.
        // LightRHI uses a single global push constant block for all bindless root data.
        uint32_t PushConstantBytes{128}; // safe minimum; Vulkan guarantees ≥128

        std::string_view DebugName{};
    };

    // ---------------------------------------------------------------------------
    // Compute pipeline descriptor
    // ---------------------------------------------------------------------------

    struct ComputePipelineDesc
    {
        ShaderDesc Shader;
        uint32_t   PushConstantBytes{128};

        // Explicit numthreads(X, Y, Z) shape, for backends that need it at
        // Vulkan ignores this field — local_size_x/y/z is embedded in the
        // SPIR-V module and read automatically by the driver at dispatch time.
        Extent3D ThreadGroupSize{.Width = 0, .Height = 0, .Depth = 0};

        std::string_view DebugName{};
    };

    // ---------------------------------------------------------------------------
    // Dynamic rendering — replaces render passes entirely.
    // Maps to VK_KHR_dynamic_rendering (core in Vulkan 1.3)
    // and MTLRenderCommandEncoder on Metal.
    // ---------------------------------------------------------------------------

    struct ColorAttachment
    {
        TextureHandle Texture;
        TextureHandle ResolveTexture; // optional MSAA resolve target
        LoadOp        LoadOp{LoadOp::Clear};
        StoreOp       StoreOp{StoreOp::Store};
        ClearColor    ClearValue{};
    };

    struct DepthAttachment
    {
        TextureHandle     Texture;
        LoadOp            LoadOp{LoadOp::Clear};
        StoreOp           StoreOp{StoreOp::DontCare};
        ClearDepthStencil ClearValue{};
    };

    struct RenderingDesc
    {
        std::vector<ColorAttachment> Color;
        DepthAttachment              Depth{}; // Texture.Valid() == false → no depth
        Extent2D                     RenderArea{};
        uint32_t                     LayerCount{1};
    };

} // namespace rhi
