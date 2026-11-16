/**
 * {file}
 * {brief} Implements single-level 2D textures for host-callable compute shaders.
 * The device owns their storage and records transfers.
 */
module;
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <slang-cpp-prelude.h>
#include <span>

module lightRHI.cpu:texture;

import rhi;

namespace rhi::cpu
{
    struct Sampler
    {
        SamplerDesc Desc;
    };

    [[nodiscard]] std::size_t TexelBytes(Format format);

    class Texture final : public IRWTexture
    {
      public:
        Texture(TextureDesc desc, std::span<std::byte> bytes);

        [[nodiscard]] TextureInfo          Info() const;
        [[nodiscard]] std::size_t          CopyRowBytes(const TextureCopyRegion &region) const;
        [[nodiscard]] std::span<std::byte> Row(const TextureCopyRegion &region, std::size_t row);
        void                               Clear(const ClearColor &color);

        TextureDimensions GetDimensions(int mipLevel) override;
        void              Load(const std::int32_t *location, void *out, std::size_t size) override;
        void              Sample(SamplerState sampler, const float *location, void *out, std::size_t size) override;
        void              SampleLevel(SamplerState sampler, const float *location, float level, void *out,
                                      std::size_t size) override;
        void             *refAt(const std::uint32_t *location) override;

      private:
        [[nodiscard]] float4               _read(int x, int y);
        [[nodiscard]] float4               _sample(const SamplerDesc &sampler, int x, int y);
        [[nodiscard]] std::span<std::byte> _pixel(std::size_t x, std::size_t y);

        TextureInfo          _info;
        std::span<std::byte> _bytes;
    };

    namespace
    {
        [[nodiscard]] std::optional<int> Address(int value, int extent, SamplerAddressMode mode)
        {
            switch (mode)
            {
                case SamplerAddressMode::ClampToEdge:
                    return std::clamp(value, 0, extent - 1);
                case SamplerAddressMode::ClampToBorder:
                    if (value < 0 || value >= extent)
                    {
                        return std::nullopt;
                    }
                    return value;
                case SamplerAddressMode::Repeat:
                {
                    const auto wrapped{value % extent};
                    return wrapped < 0 ? wrapped + extent : wrapped;
                }
                case SamplerAddressMode::MirroredRepeat:
                {
                    const auto period{2 * extent};
                    const auto remainder{value % period};
                    const auto wrapped{remainder < 0 ? remainder + period : remainder};
                    return wrapped < extent ? wrapped : period - 1 - wrapped;
                }
            }
            FailContract("CPU sampler address mode");
        }

        [[nodiscard]] double TextureCoordinate(float normalized, std::uint32_t extent, SamplerAddressMode mode)
        {
            switch (mode)
            {
                case SamplerAddressMode::ClampToEdge:
                case SamplerAddressMode::ClampToBorder:
                    return std::clamp(double(normalized), -1.0, 2.0) * extent;
                case SamplerAddressMode::Repeat:
                    return std::fmod(double(normalized), 1.0) * extent;
                case SamplerAddressMode::MirroredRepeat:
                    return std::fmod(double(normalized), 2.0) * extent;
            }
            FailContract("CPU sampler address mode");
        }
    } // namespace

    std::size_t TexelBytes(const Format format)
    {
        switch (format)
        {
            case Format::R8Unorm:
                return 1;
            case Format::RGBA8Unorm:
            case Format::BGRA8Unorm:
            case Format::R32Float:
                return 4;
            case Format::RGBA16Float:
            case Format::RG32Float:
                return 8;
            case Format::RGBA32Float:
                return 16;
            default:
                return 0;
        }
    }

    Texture::Texture(const TextureDesc desc, const std::span<std::byte> bytes)
        : _info{.Extent = desc.Extent, .Format = desc.Format, .Usage = desc.Usage}, _bytes{bytes}
    {
    }

    TextureInfo Texture::Info() const
    {
        return _info;
    }

    std::size_t Texture::CopyRowBytes(const TextureCopyRegion &region) const
    {
        const auto &at{region.DstOffset};
        const auto &extent{region.Extent};
        if (region.MipLevel != 0 || region.ArrayLayer != 0 || at.Z != 0 || extent.Depth != 1 || at.X < 0 || at.Y < 0 ||
            std::uint64_t(at.X) + extent.Width > _info.Extent.Width ||
            std::uint64_t(at.Y) + extent.Height > _info.Extent.Height)
        {
            FailContract("CPU texture copy region is outside the texture");
        }
        return TexelBytes(_info.Format) * extent.Width;
    }

    std::span<std::byte> Texture::Row(const TextureCopyRegion &region, const std::size_t row)
    {
        return _bytes.subspan(
            ((std::size_t(region.DstOffset.Y) + row) * _info.Extent.Width + std::size_t(region.DstOffset.X)) *
                TexelBytes(_info.Format),
            TexelBytes(_info.Format) * region.Extent.Width);
    }

    std::span<std::byte> Texture::_pixel(const std::size_t x, const std::size_t y)
    {
        const auto stride{TexelBytes(_info.Format)};
        return _bytes.subspan((y * _info.Extent.Width + x) * stride, stride);
    }

    void Texture::Clear(const ClearColor &color)
    {
        std::array<std::byte, 16> encoded{};
        auto                      values{std::array{color.R, color.G, color.B, color.A}};
        const auto                stride{TexelBytes(_info.Format)};
        switch (_info.Format)
        {
            case Format::BGRA8Unorm:
                std::swap(values[0], values[2]);
                [[fallthrough]];
            case Format::R8Unorm:
            case Format::RGBA8Unorm:
                for (std::size_t i{}; i < stride; ++i)
                {
                    encoded.at(i) =
                        std::byte(static_cast<std::uint8_t>(std::round(std::clamp(values.at(i), 0.0F, 1.0F) * 255.0F)));
                }
                break;
            case Format::RGBA16Float:
            {
                const std::array result{half(color.R), half(color.G), half(color.B), half(color.A)};
                std::memcpy(encoded.data(), result.data(), stride);
                break;
            }
            default:
                std::memcpy(encoded.data(), values.data(), stride);
                break;
        }
        for (std::size_t at{}; at < _bytes.size(); at += stride)
        {
            std::memcpy(_bytes.subspan(at, stride).data(), encoded.data(), stride);
        }
    }

    TextureDimensions Texture::GetDimensions(const int mipLevel)
    {
        if (mipLevel > 0)
        {
            FailContract("CPU textures have only one level");
        }
        return {
            .shape = SLANG_TEXTURE_2D, .width = _info.Extent.Width, .height = _info.Extent.Height, .numberOfLevels = 1};
    }

    float4 Texture::_read(const int x, const int y)
    {
        if (x < 0 || y < 0 || std::uint32_t(x) >= _info.Extent.Width || std::uint32_t(y) >= _info.Extent.Height)
        {
            return {};
        }
        const auto pixel{_pixel(std::size_t(x), std::size_t(y))};
        const auto unorm{[pixel](std::size_t i) { return float(std::to_integer<unsigned>(pixel[i])) / 255.0F; }};
        switch (_info.Format)
        {
            case Format::R8Unorm:
                return {unorm(0), 0, 0, 1};
            case Format::RGBA8Unorm:
                return {unorm(0), unorm(1), unorm(2), unorm(3)};
            case Format::BGRA8Unorm:
                return {unorm(2), unorm(1), unorm(0), unorm(3)};
            case Format::RGBA16Float:
            {
                std::array<half, 4> values{};
                std::memcpy(values.data(), pixel.data(), pixel.size());
                return {float(values[0]), float(values[1]), float(values[2]), float(values[3])};
            }
            default:
            {
                std::array values{0.F, 0.F, 0.F, 1.F};
                std::memcpy(values.data(), pixel.data(), pixel.size());
                return {values[0], values[1], values[2], values[3]};
            }
        }
    }

    void Texture::Load(const std::int32_t *location, void *out, const std::size_t size)
    {
        const auto value{_read(location[0], location[1])};
        if (size > sizeof(value))
        {
            FailContract("CPU texture load type");
        }
        std::memcpy(out, &value, size);
    }

    float4 Texture::_sample(const SamplerDesc &sampler, const int x, const int y)
    {
        const auto u{Address(x, int(_info.Extent.Width), sampler.AddressU)};
        const auto v{Address(y, int(_info.Extent.Height), sampler.AddressV)};
        if (u && v)
        {
            return _read(*u, *v);
        }
        switch (sampler.BorderColor)
        {
            case BorderColor::TransparentBlack:
                return {};
            case BorderColor::OpaqueWhite:
                return {1, 1, 1, 1};
            default:
                return {0, 0, 0, 1};
        }
    }

    void Texture::Sample(SamplerState sampler, const float *location, void *out, const std::size_t size)
    {
        SampleLevel(sampler, location, 0.0F, out, size);
    }

    void Texture::SampleLevel(SamplerState sampler, const float *location, const float level, void *out,
                              const std::size_t size)
    {
        if (!sampler.state || size > sizeof(float4) || !std::isfinite(location[0]) || !std::isfinite(location[1]))
        {
            FailContract("invalid CPU texture sample");
        }
        const auto &desc{reinterpret_cast<const Sampler *>(sampler.state)->Desc};
        const auto  x{TextureCoordinate(location[0], _info.Extent.Width, desc.AddressU)};
        const auto  y{TextureCoordinate(location[1], _info.Extent.Height, desc.AddressV)};
        float4      value{};
        if ((level > 0.0F ? desc.MinFilter : desc.MagFilter) == SamplerFilter::Nearest)
        {
            value = _sample(desc, int(std::floor(x)), int(std::floor(y)));
        }
        else
        {
            const auto px{std::floor(x - 0.5)};
            const auto py{std::floor(y - 0.5)};
            const auto i{int(px)};
            const auto j{int(py)};
            const auto tx{float(x - 0.5 - px)};
            const auto ty{float(y - 0.5 - py)};
            const auto top{_sample(desc, i, j) * float4(1.0F - tx) + _sample(desc, i + 1, j) * float4(tx)};
            const auto bottom{_sample(desc, i, j + 1) * float4(1.0F - tx) + _sample(desc, i + 1, j + 1) * float4(tx)};
            value = top * float4(1.0F - ty) + bottom * float4(ty);
        }
        std::memcpy(out, &value, size);
    }

    void *Texture::refAt(const std::uint32_t *location)
    {
        if (location[0] >= _info.Extent.Width || location[1] >= _info.Extent.Height)
        {
            FailContract("CPU texture write outside the texture");
        }
        return _pixel(location[0], location[1]).data();
    }
} // namespace rhi::cpu
