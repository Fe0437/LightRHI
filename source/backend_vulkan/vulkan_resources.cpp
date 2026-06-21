// vulkan_resources.cpp — Format / sampler conversion tables
// Factored out from vulkan_device.cpp to keep that file focused on lifecycle.

module;
#define VK_NO_PROTOTYPES
#include <array>
#include <cassert>
#include <cstdint>
#include <vulkan/vulkan.h>

module lightRHI;
import rhi;

namespace rhi::vulkan
{

    // ---------------------------------------------------------------------------
    // Format → VkFormat lookup table
    // Indexed by rhi::Format enum value.
    // ---------------------------------------------------------------------------

    static constexpr std::array<VkFormat, 64> kFormatTable{
        []
        {
            std::array<VkFormat, 64> t{};
            t.fill(VK_FORMAT_UNDEFINED);

            using F                                = Format;
            t[static_cast<int>(F::R8Unorm)]        = VK_FORMAT_R8_UNORM;
            t[static_cast<int>(F::R8Snorm)]        = VK_FORMAT_R8_SNORM;
            t[static_cast<int>(F::R8Uint)]         = VK_FORMAT_R8_UINT;
            t[static_cast<int>(F::R8Sint)]         = VK_FORMAT_R8_SINT;
            t[static_cast<int>(F::RG8Unorm)]       = VK_FORMAT_R8G8_UNORM;
            t[static_cast<int>(F::RG8Snorm)]       = VK_FORMAT_R8G8_SNORM;
            t[static_cast<int>(F::RG8Uint)]        = VK_FORMAT_R8G8_UINT;
            t[static_cast<int>(F::RG8Sint)]        = VK_FORMAT_R8G8_SINT;
            t[static_cast<int>(F::RGBA8Unorm)]     = VK_FORMAT_R8G8B8A8_UNORM;
            t[static_cast<int>(F::RGBA8Snorm)]     = VK_FORMAT_R8G8B8A8_SNORM;
            t[static_cast<int>(F::RGBA8Uint)]      = VK_FORMAT_R8G8B8A8_UINT;
            t[static_cast<int>(F::RGBA8Sint)]      = VK_FORMAT_R8G8B8A8_SINT;
            t[static_cast<int>(F::RGBA8Srgb)]      = VK_FORMAT_R8G8B8A8_SRGB;
            t[static_cast<int>(F::BGRA8Unorm)]     = VK_FORMAT_B8G8R8A8_UNORM;
            t[static_cast<int>(F::BGRA8Srgb)]      = VK_FORMAT_B8G8R8A8_SRGB;
            t[static_cast<int>(F::R16Float)]       = VK_FORMAT_R16_SFLOAT;
            t[static_cast<int>(F::RG16Float)]      = VK_FORMAT_R16G16_SFLOAT;
            t[static_cast<int>(F::RGBA16Float)]    = VK_FORMAT_R16G16B16A16_SFLOAT;
            t[static_cast<int>(F::R16Uint)]        = VK_FORMAT_R16_UINT;
            t[static_cast<int>(F::R16Sint)]        = VK_FORMAT_R16_SINT;
            t[static_cast<int>(F::RG16Uint)]       = VK_FORMAT_R16G16_UINT;
            t[static_cast<int>(F::RG16Sint)]       = VK_FORMAT_R16G16_SINT;
            t[static_cast<int>(F::R32Float)]       = VK_FORMAT_R32_SFLOAT;
            t[static_cast<int>(F::RG32Float)]      = VK_FORMAT_R32G32_SFLOAT;
            t[static_cast<int>(F::RGBA32Float)]    = VK_FORMAT_R32G32B32A32_SFLOAT;
            t[static_cast<int>(F::R32Uint)]        = VK_FORMAT_R32_UINT;
            t[static_cast<int>(F::R32Sint)]        = VK_FORMAT_R32_SINT;
            t[static_cast<int>(F::RG32Uint)]       = VK_FORMAT_R32G32_UINT;
            t[static_cast<int>(F::RG32Sint)]       = VK_FORMAT_R32G32_SINT;
            t[static_cast<int>(F::RGBA32Uint)]     = VK_FORMAT_R32G32B32A32_UINT;
            t[static_cast<int>(F::RGBA32Sint)]     = VK_FORMAT_R32G32B32A32_SINT;
            t[static_cast<int>(F::RGB10A2Unorm)]   = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            t[static_cast<int>(F::RGB10A2Uint)]    = VK_FORMAT_A2B10G10R10_UINT_PACK32;
            t[static_cast<int>(F::R11G11B10Float)] = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            t[static_cast<int>(F::RGB9E5Float)]    = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
            t[static_cast<int>(F::D16Unorm)]       = VK_FORMAT_D16_UNORM;
            t[static_cast<int>(F::D32Float)]       = VK_FORMAT_D32_SFLOAT;
            t[static_cast<int>(F::D24UnormS8Uint)] = VK_FORMAT_D24_UNORM_S8_UINT;
            t[static_cast<int>(F::D32FloatS8Uint)] = VK_FORMAT_D32_SFLOAT_S8_UINT;
            t[static_cast<int>(F::BC1Unorm)]       = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            t[static_cast<int>(F::BC1Srgb)]        = VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
            t[static_cast<int>(F::BC2Unorm)]       = VK_FORMAT_BC2_UNORM_BLOCK;
            t[static_cast<int>(F::BC2Srgb)]        = VK_FORMAT_BC2_SRGB_BLOCK;
            t[static_cast<int>(F::BC3Unorm)]       = VK_FORMAT_BC3_UNORM_BLOCK;
            t[static_cast<int>(F::BC3Srgb)]        = VK_FORMAT_BC3_SRGB_BLOCK;
            t[static_cast<int>(F::BC4Unorm)]       = VK_FORMAT_BC4_UNORM_BLOCK;
            t[static_cast<int>(F::BC4Snorm)]       = VK_FORMAT_BC4_SNORM_BLOCK;
            t[static_cast<int>(F::BC5Unorm)]       = VK_FORMAT_BC5_UNORM_BLOCK;
            t[static_cast<int>(F::BC5Snorm)]       = VK_FORMAT_BC5_SNORM_BLOCK;
            t[static_cast<int>(F::BC6HUfloat)]     = VK_FORMAT_BC6H_UFLOAT_BLOCK;
            t[static_cast<int>(F::BC6HSfloat)]     = VK_FORMAT_BC6H_SFLOAT_BLOCK;
            t[static_cast<int>(F::BC7Unorm)]       = VK_FORMAT_BC7_UNORM_BLOCK;
            t[static_cast<int>(F::BC7Srgb)]        = VK_FORMAT_BC7_SRGB_BLOCK;
            return t;
        }()};

    [[nodiscard]] VkFormat toVkFormat(Format f) noexcept
    {
        const auto idx = static_cast<size_t>(f);
        if (idx >= kFormatTable.size())
        {
            return VK_FORMAT_UNDEFINED;
        }
        return kFormatTable[idx];
    }

    // ---------------------------------------------------------------------------
    // Sampler descriptor → VkSamplerCreateInfo
    // ---------------------------------------------------------------------------

    [[nodiscard]] VkSamplerCreateInfo toVkSamplerCreateInfo(const SamplerDesc &desc) noexcept
    {
        auto toFilter = [](SamplerFilter f)
        { return f == SamplerFilter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; };
        auto toMipMode = [](SamplerMipMode m)
        { return m == SamplerMipMode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST; };
        auto toAddressMode = [](SamplerAddressMode m) -> VkSamplerAddressMode
        {
            switch (m)
            {
                case SamplerAddressMode::Repeat:
                    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                case SamplerAddressMode::MirroredRepeat:
                    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                case SamplerAddressMode::ClampToEdge:
                    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                case SamplerAddressMode::ClampToBorder:
                    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            }
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        };
        auto toBorderColor = [](BorderColor c) -> VkBorderColor
        {
            switch (c)
            {
                case BorderColor::TransparentBlack:
                    return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                case BorderColor::OpaqueBlack:
                    return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
                case BorderColor::OpaqueWhite:
                    return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            }
            return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        };
        auto toCompareOp = [](CompareOp op) -> VkCompareOp
        {
            switch (op)
            {
                case CompareOp::Never:
                    return VK_COMPARE_OP_NEVER;
                case CompareOp::Less:
                    return VK_COMPARE_OP_LESS;
                case CompareOp::Equal:
                    return VK_COMPARE_OP_EQUAL;
                case CompareOp::LessEqual:
                    return VK_COMPARE_OP_LESS_OR_EQUAL;
                case CompareOp::Greater:
                    return VK_COMPARE_OP_GREATER;
                case CompareOp::NotEqual:
                    return VK_COMPARE_OP_NOT_EQUAL;
                case CompareOp::GreaterEqual:
                    return VK_COMPARE_OP_GREATER_OR_EQUAL;
                case CompareOp::Always:
                    return VK_COMPARE_OP_ALWAYS;
            }
            return VK_COMPARE_OP_ALWAYS;
        };

        return VkSamplerCreateInfo{
            .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter               = toFilter(desc.MagFilter),
            .minFilter               = toFilter(desc.MinFilter),
            .mipmapMode              = toMipMode(desc.MipMode),
            .addressModeU            = toAddressMode(desc.AddressU),
            .addressModeV            = toAddressMode(desc.AddressV),
            .addressModeW            = toAddressMode(desc.AddressW),
            .mipLodBias              = desc.MipLodBias,
            .anisotropyEnable        = desc.Anisotropy ? VK_TRUE : VK_FALSE,
            .maxAnisotropy           = desc.MaxAniso,
            .compareEnable           = desc.CompareEnable ? VK_TRUE : VK_FALSE,
            .compareOp               = toCompareOp(desc.CompareOp),
            .minLod                  = desc.MinLod,
            .maxLod                  = desc.MaxLod,
            .borderColor             = toBorderColor(desc.BorderColor),
            .unnormalizedCoordinates = VK_FALSE,
        };
    }

} // namespace rhi::vulkan
