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

    /** {brief} Selects validation behavior and the application name when creating a device. */
    struct DeviceDesc
    {
        bool        EnableValidation{true};     ///< Enables API validation when the backend provides it.
        bool        EnableGpuValidation{false}; ///< Enables expensive GPU-assisted validation.
        const char *AppName{"LightRHI"};        ///< Null-terminated name shown by diagnostics and tooling.
    };

    // ---------------------------------------------------------------------------
    // Buffer
    // ---------------------------------------------------------------------------

    /** {brief} Describes the storage, access modes, and memory placement of a buffer. */
    struct BufferDesc
    {
        uint64_t         Size{};                          ///< Allocation size in bytes; must be non-zero.
        BufferUsage      Usage{BufferUsage::Storage};     ///< All operations for which the buffer may be used.
        MemoryType       MemoryType{MemoryType::GpuOnly}; ///< CPU/GPU visibility required by the caller.
        std::string_view DebugName{};                     ///< Optional diagnostic name, copied during creation.
    };

    // ---------------------------------------------------------------------------
    // Texture
    // ---------------------------------------------------------------------------

    /** {brief} Describes a texture and all of its subresources. */
    struct TextureDesc
    {
        TextureDimension Dimension{TextureDimension::Tex2D};          ///< Dimensionality visible to shaders.
        Format           Format{Format::RGBA8Unorm};                  ///< Texel or depth/stencil format.
        Extent3D         Extent{.Width = 1, .Height = 1, .Depth = 1}; ///< Base-mip dimensions.
        uint32_t         MipLevels{1};                                ///< Number of allocated mip levels.
        uint32_t         ArrayLayers{1};                              ///< Number of array layers or cube faces.
        uint32_t         SampleCount{1};                              ///< Samples per texel for multisampling.
        TextureUsage     Usage{TextureUsage::Sampled | TextureUsage::TransferDst}; ///< All intended texture operations.
        ResourceState    InitialState{ResourceState::Undefined}; ///< State tracked immediately after creation.
        std::string_view DebugName{};                            ///< Optional diagnostic name, copied during creation.
    };

    /**
     * {brief} Returns a descriptor for a sampled two-dimensional texture.
     * {note} Add storage, render-target, or transfer-source usage before creation when needed.
     */
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

    /** {brief} Returns a descriptor for a single-sample color render target that can also be sampled. */
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

    /**
     * {brief} Returns a descriptor for a single-sample depth target that can also be sampled.
     * {pre} `fmt` is a depth format.
     */
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

    /** {brief} Describes filtering, addressing, level-of-detail, and comparison sampling. */
    struct SamplerDesc
    {
        SamplerFilter      MinFilter{SamplerFilter::Linear};      ///< Filter used when reducing the image.
        SamplerFilter      MagFilter{SamplerFilter::Linear};      ///< Filter used when enlarging the image.
        SamplerMipMode     MipMode{SamplerMipMode::Linear};       ///< Filter between adjacent mip levels.
        SamplerAddressMode AddressU{SamplerAddressMode::Repeat};  ///< Addressing outside the U range.
        SamplerAddressMode AddressV{SamplerAddressMode::Repeat};  ///< Addressing outside the V range.
        SamplerAddressMode AddressW{SamplerAddressMode::Repeat};  ///< Addressing outside the W range.
        float              MipLodBias{0.F};                       ///< Bias added to the shader-selected mip level.
        float              MinLod{0.F};                           ///< Lowest accessible mip level.
        float              MaxLod{1000.F};                        ///< Highest accessible mip level.
        bool               Anisotropy{false};                     ///< Enables anisotropic filtering.
        float              MaxAniso{1.F};                         ///< Requested anisotropy when enabled.
        bool               CompareEnable{false};                  ///< Enables depth-reference comparison sampling.
        CompareOp          CompareOp{CompareOp::Always};          ///< Comparison applied when comparison is enabled.
        BorderColor        BorderColor{BorderColor::OpaqueBlack}; ///< Value returned by ClampToBorder addressing.
        std::string_view   DebugName{};                           ///< Optional diagnostic name, copied during creation.
    };

    /** {brief} Returns a trilinear sampler that repeats in every coordinate. */
    [[nodiscard]] inline SamplerDesc LinearRepeat()
    {
        return SamplerDesc{
            .MinFilter = SamplerFilter::Linear,
            .MagFilter = SamplerFilter::Linear,
            .MipMode   = SamplerMipMode::Linear,
        };
    }

    /** {brief} Returns a nearest-neighbor sampler clamped to the edge in every coordinate. */
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

    /** {brief} Returns a comparison sampler suitable for conventional less-equal shadow maps. */
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

    /** {brief} Selects a contiguous byte range for a buffer-to-buffer copy. */
    struct BufferCopyRegion
    {
        uint64_t SrcOffset{0}; ///< First source byte.
        uint64_t DstOffset{0}; ///< First destination byte.
        uint64_t Size{0};      ///< Number of bytes to copy.
    };

    /** {brief} Selects one mip/layer and a three-dimensional region for a texture copy. */
    struct TextureCopyRegion
    {
        uint32_t MipLevel{0};   ///< Mip level to copy.
        uint32_t ArrayLayer{0}; ///< Array layer or cube face to copy.
        Offset3D DstOffset{};   ///< Destination origin within the selected subresource.
        Extent3D Extent{};      ///< Width, height, and depth of the copied region.
    };

    // ---------------------------------------------------------------------------
    // Mapped Buffer (returned by IDevice::MapBuffer / UnmapBuffer)
    // ---------------------------------------------------------------------------

    /**
     * {brief} Non-owning view of a buffer mapped into CPU address space.
     * {note} The view and pointers obtained from it become invalid at UnmapBuffer().
     */
    struct MappedBuffer
    {
        void    *Data{nullptr}; ///< First mapped byte, or null when invalid.
        uint64_t Size{0};       ///< Number of mapped bytes.

        /**
         * {brief} Interprets the first mapped byte as `T` without changing ownership.
         * {pre} The mapping is valid, suitably aligned, and large enough for `T`.
         */
        template <typename T> [[nodiscard]] T *As() const noexcept
        {
            return static_cast<T *>(Data);
        }

        /** {brief} Reports whether this view refers to mapped memory. */
        [[nodiscard]] bool Valid() const noexcept
        {
            return Data != nullptr;
        }
    };

} // namespace rhi
