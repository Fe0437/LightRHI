module;
#include <cstdint>
#include <optional>
#include <string_view>

export module rhi:descriptors;
import :types;
import :handles;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // Device
    // ---------------------------------------------------------------------------

    struct DeviceDesc
    {
        bool        EnableValidation{true};
        bool        EnableGpuValidation{false}; // slow; off by default
        const char *AppName{"LightRHI"};
    };

    // ---------------------------------------------------------------------------
    // Buffer
    // ---------------------------------------------------------------------------

    struct BufferDesc
    {
        uint64_t         Size{};
        BufferUsage      Usage{BufferUsage::Storage};
        MemoryType       MemoryType{MemoryType::GpuOnly};
        std::string_view DebugName{};
    };

    // ---------------------------------------------------------------------------
    // Texture
    // ---------------------------------------------------------------------------

    struct TextureDesc
    {
        TextureDimension Dimension{TextureDimension::Tex2D};
        Format           Format{Format::RGBA8Unorm};
        Extent3D         Extent{.Width = 1, .Height = 1, .Depth = 1};
        uint32_t         MipLevels{1};
        uint32_t         ArrayLayers{1};
        uint32_t         SampleCount{1};
        TextureUsage     Usage{TextureUsage::Sampled | TextureUsage::TransferDst};
        ResourceState    InitialState{ResourceState::Undefined};
        std::string_view DebugName{};
    };

    // Convenience constructors

    [[nodiscard]] inline TextureDesc Texture2D(uint32_t width, uint32_t height, Format fmt = Format::RGBA8Unorm,
                                               TextureUsage     use = TextureUsage::Sampled | TextureUsage::TransferDst,
                                               std::string_view name = {})
    {
        return TextureDesc{
            .Dimension = TextureDimension::Tex2D,
            .Format    = fmt,
            .Extent    = {.Width = width, .Height = height, .Depth = 1},
            .Usage     = use,
            .DebugName = name,
        };
    }

    [[nodiscard]] inline TextureDesc RenderTarget2D(uint32_t width, uint32_t height, Format fmt = Format::RGBA8Unorm,
                                                    std::string_view name = {})
    {
        return TextureDesc{
            .Dimension = TextureDimension::Tex2D,
            .Format    = fmt,
            .Extent    = {.Width = width, .Height = height, .Depth = 1},
            .Usage     = TextureUsage::RenderTarget | TextureUsage::Sampled,
            .DebugName = name,
        };
    }

    [[nodiscard]] inline TextureDesc DepthTarget2D(uint32_t width, uint32_t height, Format fmt = Format::D32Float,
                                                   std::string_view name = {})
    {
        return TextureDesc{
            .Dimension = TextureDimension::Tex2D,
            .Format    = fmt,
            .Extent    = {.Width = width, .Height = height, .Depth = 1},
            .Usage     = TextureUsage::DepthStencil | TextureUsage::Sampled,
            .DebugName = name,
        };
    }

    // ---------------------------------------------------------------------------
    // Sampler
    // ---------------------------------------------------------------------------

    struct SamplerDesc
    {
        SamplerFilter      MinFilter{SamplerFilter::Linear};
        SamplerFilter      MagFilter{SamplerFilter::Linear};
        SamplerMipMode     MipMode{SamplerMipMode::Linear};
        SamplerAddressMode AddressU{SamplerAddressMode::Repeat};
        SamplerAddressMode AddressV{SamplerAddressMode::Repeat};
        SamplerAddressMode AddressW{SamplerAddressMode::Repeat};
        float              MipLodBias{0.F};
        float              MinLod{0.F};
        float              MaxLod{1000.F};
        bool               Anisotropy{false};
        float              MaxAniso{1.F};
        bool               CompareEnable{false};
        CompareOp          CompareOp{CompareOp::Always};
        BorderColor        BorderColor{BorderColor::OpaqueBlack};
        std::string_view   DebugName{};
    };

    // Common presets
    [[nodiscard]] inline SamplerDesc LinearRepeat()
    {
        return SamplerDesc{
            .MinFilter = SamplerFilter::Linear,
            .MagFilter = SamplerFilter::Linear,
            .MipMode   = SamplerMipMode::Linear,
        };
    }

    [[nodiscard]] inline SamplerDesc NearestClamp()
    {
        return SamplerDesc{
            .MinFilter = SamplerFilter::Nearest,
            .MagFilter = SamplerFilter::Nearest,
            .MipMode   = SamplerMipMode::Nearest,
            .AddressU  = SamplerAddressMode::ClampToEdge,
            .AddressV  = SamplerAddressMode::ClampToEdge,
            .AddressW  = SamplerAddressMode::ClampToEdge,
        };
    }

    [[nodiscard]] inline SamplerDesc ShadowSampler()
    {
        return SamplerDesc{
            .MinFilter     = SamplerFilter::Linear,
            .MagFilter     = SamplerFilter::Linear,
            .MipMode       = SamplerMipMode::Nearest,
            .AddressU      = SamplerAddressMode::ClampToEdge,
            .AddressV      = SamplerAddressMode::ClampToEdge,
            .AddressW      = SamplerAddressMode::ClampToEdge,
            .CompareEnable = true,
            .CompareOp     = CompareOp::LessEqual,
        };
    }

    // ---------------------------------------------------------------------------
    // Copy / upload regions
    // ---------------------------------------------------------------------------

    struct BufferCopyRegion
    {
        uint64_t SrcOffset{0};
        uint64_t DstOffset{0};
        uint64_t Size{0};
    };

    struct TextureCopyRegion
    {
        uint32_t MipLevel{0};
        uint32_t ArrayLayer{0};
        Offset3D DstOffset{};
        Extent3D Extent{};
    };

    // ---------------------------------------------------------------------------
    // Mapped Buffer (returned by IDevice::MapBuffer / UnmapBuffer)
    // ---------------------------------------------------------------------------

    struct MappedBuffer
    {
        void    *Data{nullptr};
        uint64_t Size{0};

        template <typename T> [[nodiscard]] T *As() const noexcept
        {
            return static_cast<T *>(Data);
        }

        [[nodiscard]] bool Valid() const noexcept
        {
            return Data != nullptr;
        }
    };

} // namespace rhi
