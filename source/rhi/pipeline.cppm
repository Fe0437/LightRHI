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

    // ---- Shader artifacts ----

    /// \brief Non-owning view of SPIR-V words supplied to pipeline creation.
    struct SpirvBytecode
    {
        std::span<const uint32_t> Words; ///< Complete SPIR-V module, aligned as 32-bit words.
    };

    /// \brief Non-owning view of a precompiled Metal library supplied to pipeline creation.
    struct MetalLibBytecode
    {
        std::span<const uint8_t> Bytes; ///< Complete `.metallib` file contents.
    };

    /// \brief Non-owning view of Metal Shading Language source supplied to pipeline creation.
    struct MslSource
    {
        std::string_view Source; ///< Complete source text containing the requested entry point.
    };

    /// \brief Discriminated union of the shader artifact encodings accepted by pipeline creation.
    /// \note A default-constructed value is `std::monostate`, meaning no shader was supplied.
    ///
    /// Select SpirvBytecode for a Vulkan device and MslSource or
    /// MetalLibBytecode for a Metal device. Applications that load tagged shader
    /// assets can use ShaderArtifactView and ToShaderDesc() instead of constructing
    /// the alternative directly.
    using ShaderBytecode = std::variant<std::monostate, SpirvBytecode, MetalLibBytecode, MslSource>;

    /// \brief Selects one compiled shader entry point for pipeline creation.
    /// \note The bytecode storage must remain valid until CreateGraphicsPipeline() or
    /// CreateComputePipeline() returns.
    struct ShaderDesc
    {
        ShaderBytecode   Bytecode;           ///< Compiled artifact accepted by the active backend.
        std::string_view EntryPoint{"main"}; ///< Function name to use from the artifact.
    };

    /// \brief Describes a non-owning, already-compiled shader artifact.
    ///
    /// Use ToShaderDesc() to adapt loaded artifact bytes to a pipeline descriptor.
    /// The caller owns `Data` and must retain it while the converted ShaderDesc is used.
    struct ShaderArtifactView
    {
        ShaderFormat               Format{ShaderFormat::Spirv}; ///< Encoding of `Data`.
        ShaderStage                Stage{ShaderStage::None};    ///< Pipeline stage for validation and selection.
        std::string_view           EntryPoint{"main"};          ///< Entry-point name in the artifact.
        std::span<const std::byte> Data{};                      ///< Complete artifact bytes.
    };

    /// \brief Recoverable validation errors returned by ToShaderDesc().
    enum class ShaderArtifactError
    {
        SpirvSizeNotWordAligned, ///< SPIR-V data size is not a multiple of one 32-bit word.
    };

    /// \brief Creates a ShaderDesc that views the storage in `artifact`.
    /// \return A descriptor on success, or a validation error for malformed input.
    /// \note The returned descriptor does not own the artifact bytes.
    ///
    /// \code{.cpp}
    /// const auto shader = ToShaderDesc(artifact);
    /// if (!shader)
    /// {
    ///     return shader.error();
    /// }
    /// const PipelineHandle pipeline = device->CreateComputePipeline({
    ///     .Shader = *shader,
    ///     .ThreadGroupSize = {.Width = 8, .Height = 8, .Depth = 1},
    /// });
    /// \endcode
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

    // ---- Blend state ----

    /// \brief Selects blending and color writes for one color attachment.
    struct BlendState
    {
        bool        Enable{false};               ///< Enables source/destination blending.
        BlendFactor SrcColor{BlendFactor::One};  ///< Source factor for RGB channels.
        BlendFactor DstColor{BlendFactor::Zero}; ///< Destination factor for RGB channels.
        BlendOp     ColorOp{BlendOp::Add};       ///< Operation combining factored RGB values.
        BlendFactor SrcAlpha{BlendFactor::One};  ///< Source factor for alpha.
        BlendFactor DstAlpha{BlendFactor::Zero}; ///< Destination factor for alpha.
        BlendOp     AlphaOp{BlendOp::Add};       ///< Operation combining factored alpha values.
        uint8_t     WriteMask{0xF};              ///< RGBA bit mask; bit zero controls red.
    };

    /// \brief Returns opaque replacement blending with all color channels writable.
    [[nodiscard]] inline BlendState BlendDisabled()
    {
        return BlendState{.Enable = false};
    }

    /// \brief Returns conventional source-over blending for premultiplied-alpha colors.
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

    /// \brief Returns conventional source-over blending for straight-alpha colors.
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

    // ---- Depth and stencil state ----

    /// \brief Selects depth testing, depth writes, and optional stencil testing.
    struct DepthStencilState
    {
        bool      DepthTest{true};          ///< Enables comparison against the depth attachment.
        bool      DepthWrite{true};         ///< Writes passing fragment depths.
        CompareOp DepthOp{CompareOp::Less}; ///< Depth comparison function.
        bool      StencilTest{false};       ///< Enables stencil testing with the API's default stencil behavior.
    };

    /// \brief Returns a depth state that tests and writes passing fragments.
    [[nodiscard]] inline DepthStencilState DepthReadWrite(CompareOp op = CompareOp::Less)
    {
        return DepthStencilState{.DepthTest = true, .DepthWrite = true, .DepthOp = op};
    }

    /// \brief Returns a depth state that tests without modifying the depth attachment.
    [[nodiscard]] inline DepthStencilState DepthReadOnly(CompareOp op = CompareOp::Less)
    {
        return DepthStencilState{.DepthTest = true, .DepthWrite = false, .DepthOp = op};
    }

    /// \brief Returns a state with depth testing and writing disabled.
    [[nodiscard]] inline DepthStencilState DepthDisabled()
    {
        return DepthStencilState{.DepthTest = false, .DepthWrite = false};
    }

    // ---- Rasterizer state ----

    /// \brief Selects triangle culling, fill, winding, and depth-bias behavior.
    struct RasterizerState
    {
        CullMode  CullMode{CullMode::Back};               ///< Triangle faces discarded before rasterization.
        FillMode  FillMode{FillMode::Solid};              ///< Solid or wireframe triangle fill.
        FrontFace FrontFace{FrontFace::CounterClockwise}; ///< Winding considered front-facing.
        float     DepthBiasConstant{0.F};                 ///< Constant depth offset applied during rasterization.
        float     DepthBiasSlope{0.F};                    ///< Slope-scaled depth offset.
        bool      DepthClamp{false};                      ///< Clamps depth instead of clipping outside the depth range.
        bool      ConservativeRaster{false};              ///< Requests conservative coverage when supported.
    };

    // ---- Graphics pipelines ----

    /// \brief Completely describes a graphics pipeline for dynamic rendering.
    /// \note `ColorFormats`, `DepthFormat`, and `SampleCount` must match every
    /// RenderingDesc used with the resulting pipeline.
    struct GraphicsPipelineDesc
    {
        ShaderDesc VertexShader;   ///< Required vertex-stage entry point.
        ShaderDesc FragmentShader; ///< Fragment-stage entry point; omit bytecode for depth-only use.

        PrimitiveTopology Topology{PrimitiveTopology::TriangleList}; ///< Primitive assembly topology.
        RasterizerState   Rasterizer{};                              ///< Rasterization behavior.
        DepthStencilState DepthStencil{};                            ///< Depth and stencil behavior.

        std::vector<BlendState> ColorBlend{}; ///< One state per color format; empty for no color output.

        std::vector<Format> ColorFormats{};                 ///< Ordered formats of rendering color attachments.
        Format              DepthFormat{Format::Undefined}; ///< Depth attachment format, or Undefined when absent.
        uint32_t            SampleCount{1};                 ///< Sample count shared by all attachments.

        uint32_t PushConstantBytes{128}; ///< Bytes available to SetPushConstants() across graphics stages.

        std::string_view DebugName{}; ///< Optional diagnostic name, copied during creation.
    };

    // ---- Compute pipelines ----

    /// \brief Completely describes a compute pipeline.
    struct ComputePipelineDesc
    {
        ShaderDesc Shader;                 ///< Required compute-stage entry point.
        uint32_t   PushConstantBytes{128}; ///< Bytes available to SetPushConstants().

        /// \brief Thread-group shape declared by the shader; zero components request
        /// discovery from the compiled artifact when supported.
        Extent3D ThreadGroupSize{.Width = 0, .Height = 0, .Depth = 0};

        std::string_view DebugName{}; ///< Optional diagnostic name, copied during creation.
    };

    // ---- Dynamic rendering ----

    /// \brief Describes one color attachment used by BeginRendering().
    struct ColorAttachment
    {
        TextureHandle Texture;                 ///< Render target for this attachment slot.
        TextureHandle ResolveTexture;          ///< Optional single-sample resolve target.
        LoadOp        LoadOp{LoadOp::Clear};   ///< Treatment of existing attachment contents.
        StoreOp       StoreOp{StoreOp::Store}; ///< Treatment of rendered contents at EndRendering().
        ClearColor    ClearValue{};            ///< Value used when LoadOp is Clear.
    };

    /// \brief Describes the optional depth/stencil attachment used by BeginRendering().
    struct DepthAttachment
    {
        TextureHandle     Texture;                    ///< Depth texture; invalid disables the attachment.
        LoadOp            LoadOp{LoadOp::Clear};      ///< Treatment of existing depth/stencil contents.
        StoreOp           StoreOp{StoreOp::DontCare}; ///< Treatment of contents at EndRendering().
        ClearDepthStencil ClearValue{};               ///< Value used when LoadOp is Clear.
    };

    /// \brief Selects attachments and bounds for one dynamic rendering region.
    struct RenderingDesc
    {
        std::vector<ColorAttachment> Color;         ///< Ordered color attachments, matching pipeline formats.
        DepthAttachment              Depth{};       ///< Optional depth attachment.
        Extent2D                     RenderArea{};  ///< Width and height affected by rendering.
        uint32_t                     LayerCount{1}; ///< Number of array layers rendered.
    };

} // namespace rhi
