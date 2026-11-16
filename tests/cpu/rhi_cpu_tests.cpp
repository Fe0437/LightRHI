/**
 * {file}
 * {brief} Runs Slang compute shaders on host memory through the public RHI.
 * It opens no GPU device and uses no native window.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <ranges>
#include <span>
#include <string_view>

import lightRHI.cpu;

namespace
{
    void Require(bool passed, std::string_view what)
    {
        if (!passed)
        {
            std::println(stderr, "rhi_cpu: {}", what);
            std::abort();
        }
    }

    struct Constants
    {
        rhi::GpuAddress Source;
        rhi::GpuAddress Sampler;
        rhi::GpuAddress Target;
        rhi::GpuAddress Result;
        std::uint32_t   Value;
    };
} // namespace

int main()
{
    auto made{rhi::cpu::CreateDevice()};
    Require(made.has_value(), "device");
    auto &device{**made};

    const auto library{rhi::cpu::ReadShaderLibrary(RHI_CPU_SHADER_DIR, "compute")};
    Require(library.has_value(), "shader library");
    const auto layout{rhi::ReadShaderLayout(RHI_CPU_SHADER_DIR, "compute")};
    Require(layout.has_value(), "shader layout");
    const auto *const sourceField{layout->Field("g_pc", "Source")};
    Require(sourceField && sourceField->Type == "descriptor" && sourceField->Offset == offsetof(Constants, Source),
            "descriptor layout");
    const auto pipeline{
        [&](std::string_view name)
        {
            const auto shader{library->Describe(name, rhi::ShaderStage::Compute)};
            Require(shader.has_value(), "entry point descriptor");
            return device.CreateComputePipeline({.Shader = *shader, .PushConstantBytes = sizeof(Constants)});
        }};
    const auto write{pipeline("write")};
    const auto increment{pipeline("increment")};
    Require(write.has_value() && increment.has_value() && !pipeline("absent"), "entry point lookup");

    constexpr std::size_t     pixels{16};
    constexpr rhi::BufferDesc resultDesc{
        .Size  = pixels * sizeof(std::uint32_t),
        .Usage = rhi::BufferUsage::Storage | rhi::BufferUsage::TransferSrc | rhi::BufferUsage::DeviceAddress,
    };
    const auto required{device.BufferPlacementRequirements(resultDesc)};
    Require(required.has_value(), "placement requirements");
    const auto heap{device.CreateMemoryHeap({.Size = required->Alignment + required->Size})};
    Require(heap.has_value(), "host heap");
    const auto result{device.CreateBuffer(resultDesc, {.Heap = *heap, .Offset = required->Alignment})};
    const auto wrong{device.CreateBuffer(resultDesc, {.Heap = *heap, .Offset = 1})};
    Require(result.has_value() && !wrong && wrong.error() == rhi::PlacementError::MisalignedOffset, "placed buffer");
    const auto overlap{device.CreateTexture(rhi::Texture2D(4, 4, rhi::Format::R8Unorm),
                                            {.Heap = *heap, .Offset = required->Alignment})};
    Require(!overlap && overlap.error() == rhi::PlacementError::Overlapping,
            "placed buffer and texture cannot overlap");
    const auto counts{device.CreateBuffer({.Size  = sizeof(rhi::DispatchIndirectArgs),
                                           .Usage = rhi::BufferUsage::IndirectArgs | rhi::BufferUsage::TransferDst})};
    Require(counts.has_value(), "indirect buffer");
    device.UploadBuffer(*counts, std::array{rhi::DispatchIndirectArgs{.X = 2, .Y = 2}});
    const auto readback{device.CreateBuffer({.Size       = resultDesc.Size + pixels * 4 * sizeof(float),
                                             .Usage      = rhi::BufferUsage::TransferDst,
                                             .MemoryType = rhi::MemoryType::GpuToCpu})};
    const auto source{device.CreateTexture(rhi::Texture2D(4, 4, rhi::Format::R8Unorm))};
    const auto target{device.CreateTexture(
        rhi::Texture2D(4, 4, rhi::Format::RGBA32Float, rhi::TextureUsage::Storage | rhi::TextureUsage::TransferSrc))};
    const auto sampler{
        device.CreateSampler({.MinFilter = rhi::SamplerFilter::Nearest, .MagFilter = rhi::SamplerFilter::Nearest})};
    Require(readback.has_value() && source.has_value() && target.has_value() && sampler.has_value(), "resources");

    std::array<std::uint8_t, 20> upload{};
    for (std::size_t i{}; i < pixels; ++i)
    {
        upload.at((i / 4) * 5 + i % 4) = static_cast<std::uint8_t>(i * 17);
    }
    const rhi::TextureCopyRegion region{.Extent{.Width = 4, .Height = 4}};
    device.UploadTexture(*source, upload, 5, upload.size(), region);
    const auto commands{device.CreateCommandList(rhi::QueueType::Compute)};
    Require(commands.has_value(), "command list");
    auto &list{**commands};
    list.Begin();
    list.SetPipeline(*write);
    list.SetPushConstants(Constants{.Source  = device.TextureAddress(*source),
                                    .Sampler = device.SamplerAddress(*sampler),
                                    .Target  = device.TextureAddress(*target),
                                    .Result  = device.BufferAddress(*result),
                                    .Value   = 10});
    list.Dispatch(2, 2);
    list.Transition(rhi::MemoryBarrier{});
    list.FlushBarriers();
    list.SetPipeline(*increment);
    // The new pipeline uses the same buffer address and updates only the added value.
    list.SetPushConstants(std::uint32_t{3}, offsetof(Constants, Value));
    list.DispatchIndirect(*counts, 0);
    list.Transition(rhi::MemoryBarrier{});
    list.FlushBarriers();
    list.CopyBuffer(*result, *readback, {.Size = resultDesc.Size});
    list.CopyTextureToBuffer(*target, region, *readback, resultDesc.Size);
    list.End();
    const auto before{device.MapBuffer(*readback)};
    Require(before.Valid() && before.As<std::uint32_t>()[0] == 0, "recording does not execute");
    device.UnmapBuffer(*readback);

    const auto fence{device.Submit(list)};
    Require(device.IsFenceComplete(fence), "submission completes synchronously");
    list.Begin();
    list.End();
    const auto dependent{device.Submit(list, {.WaitFence = fence})};
    Require(dependent.Id > fence.Id && device.IsFenceComplete(dependent), "fence dependency");
    const auto      mapped{device.MapBuffer(*readback)};
    const std::span values{mapped.As<const std::uint32_t>(), pixels};
    const auto      texels{std::span{mapped.As<const float>(), mapped.Size / sizeof(float)}.subspan(pixels)};
    for (std::size_t i{}; i < pixels; ++i)
    {
        const auto opposite{i ^ 5U};
        Require(values[i] == opposite + 13, "indirect dispatch sees shared workgroup output");
        Require(texels[i * 4] == float(i * 17) / 255.0F && texels[i * 4 + 1] == 0.0F && texels[i * 4 + 2] == 0.0F &&
                    texels[i * 4 + 3] == 1.0F,
                "sampled texel and storage write");
    }
    device.UnmapBuffer(*readback);
    device.DestroyPipeline(*increment);
    device.DestroyPipeline(*write);
    device.DestroySampler(*sampler);
    device.DestroyTexture(*target);
    device.DestroyTexture(*source);
    device.DestroyBuffer(*readback);
    device.DestroyBuffer(*counts);
    device.DestroyBuffer(*result);
    const auto reused{device.CreateTexture(rhi::Texture2D(4, 4, rhi::Format::R8Unorm),
                                           {.Heap = *heap, .Offset = required->Alignment})};
    Require(reused.has_value(), "released placement can be reused by a texture");
    device.DestroyTexture(*reused);
    device.DestroyMemoryHeap(*heap);
    Require(device.BindlessHeap().UsedBuffers() == 0 && device.BindlessHeap().UsedTextures() == 0 &&
                device.BindlessHeap().UsedSamplers() == 0,
            "all slots released");
}
