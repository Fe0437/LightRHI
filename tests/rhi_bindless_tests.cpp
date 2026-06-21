// rhi_bindless_tests.cpp — focused LightRHI bindless resource tests.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

import lightRHI;

#include "shader_artifact_loader.h"

#define REQUIRE(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "FAIL: %s  [%s:%d]\n", #expr, __FILE__, __LINE__);                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (false)

namespace
{
    void TestResourceLifetimeAccounting()
    {
        auto  device = rhi::CreateDevice({});
        auto &heap   = device->BindlessHeap();

        REQUIRE(heap.MaxBuffers() > 0);
        REQUIRE(heap.MaxTextures() > 0);
        REQUIRE(heap.MaxSamplers() > 0);

        const uint32_t buffersBefore{heap.UsedBuffers()};
        const uint32_t texturesBefore{heap.UsedTextures()};
        const uint32_t samplersBefore{heap.UsedSamplers()};

        auto buffer  = device->CreateBuffer({
            .Size       = 64,
            .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
            .MemoryType = rhi::MemoryType::CpuToGpu,
            .DebugName  = "bindless_lifetime.buffer",
        });
        auto texture = device->CreateTexture(
            rhi::Texture2D(1, 1, rhi::Format::RGBA8Unorm, rhi::TextureUsage::Sampled, "bindless_lifetime.texture"));
        auto sampler = device->CreateSampler(rhi::NearestClamp());

        REQUIRE(buffer.Valid());
        REQUIRE(texture.Valid());
        REQUIRE(sampler.Valid());
        REQUIRE(heap.UsedBuffers() == buffersBefore + 1);
        REQUIRE(heap.UsedTextures() == texturesBefore + 1);
        REQUIRE(heap.UsedSamplers() == samplersBefore + 1);

        device->DestroySampler(sampler);
        device->DestroyTexture(texture);
        device->DestroyBuffer(buffer);
        REQUIRE(heap.UsedBuffers() == buffersBefore);
        REQUIRE(heap.UsedTextures() == texturesBefore);
        REQUIRE(heap.UsedSamplers() == samplersBefore);
    }

    void TestTextureSelectionAndSampling()
    {
        auto           device = rhi::CreateDevice({});
        const uint32_t texturesBefore{device->BindlessHeap().UsedTextures()};

        auto pipeline = device->CreateComputePipeline({
            .Shader          = rhitest::loadShaderArtifact("bindless_texture_test", "bindless_texture_test_main",
                                                           rhi::ShaderStage::Compute),
            .ThreadGroupSize = {8, 1, 1},
            .DebugName       = "bindless_texture_test.pso",
        });
        REQUIRE(pipeline.Valid());

        const auto textureDesc = [](const char *name)
        {
            return rhi::Texture2D(2, 2, rhi::Format::RGBA8Unorm,
                                  rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferDst, name);
        };
        auto texture0 = device->CreateTexture(textureDesc("bindless_texture_test.texture0"));
        auto texture1 = device->CreateTexture(textureDesc("bindless_texture_test.texture1"));
        REQUIRE(texture0.Valid());
        REQUIRE(texture1.Valid());
        REQUIRE(texture0.Index != texture1.Index);
        REQUIRE(device->BindlessHeap().UsedTextures() == texturesBefore + 2);

        constexpr uint8_t            pixels0[16]{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255};
        constexpr uint8_t            pixels1[16]{0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255};
        constexpr uint64_t           rowPitch{8};
        constexpr uint64_t           slicePitch{16};
        const rhi::TextureCopyRegion region{.Extent = {2, 2, 1}};
        device->UploadTexture(texture0, pixels0, rowPitch, slicePitch, region);
        device->UploadTexture(texture1, pixels1, rowPitch, slicePitch, region);

        auto transition = device->CreateCommandList(rhi::QueueType::Compute, "bindless_texture_test.transition");
        transition->Begin();
        transition->Transition(texture0, rhi::ResourceState::TransferDst, rhi::ResourceState::ShaderRead);
        transition->Transition(texture1, rhi::ResourceState::TransferDst, rhi::ResourceState::ShaderRead);
        transition->FlushBarriers();
        transition->End();
        device->WaitForFence(device->Submit(*transition));

        auto output = device->CreateBuffer({
            .Size       = 32 * sizeof(float),
            .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
            .MemoryType = rhi::MemoryType::GpuToCpu,
            .DebugName  = "bindless_texture_test.output",
        });
        REQUIRE(output.Valid());

        struct alignas(8) PushConstants
        {
            uint64_t outputAddress;
            uint64_t texture0;
            uint64_t texture1;
        };
        const PushConstants pc{device->BufferAddress(output).Address, device->TextureAddress(texture0).Address,
                               device->TextureAddress(texture1).Address};

        auto command = device->CreateCommandList(rhi::QueueType::Compute, "bindless_texture_test.dispatch");
        command->Begin();
        command->SetPipeline(pipeline);
        command->SetPushConstants(&pc, sizeof(pc), 0);
        command->Dispatch(1, 1, 1);
        command->End();
        device->WaitForFence(device->Submit(*command));

        constexpr float expected[32]{1.F, 0.F, 0.F, 1.F, 0.F, 1.F, 0.F, 1.F, 0.F, 0.F, 1.F, 1.F, 1.F, 1.F, 0.F, 1.F,
                                     0.F, 1.F, 1.F, 1.F, 1.F, 0.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 0.F, 0.F, 0.F, 1.F};
        const auto      mapped = device->MapBuffer(output);
        REQUIRE(mapped.Data != nullptr);
        const auto *actual = static_cast<const float *>(mapped.Data);
        bool        matches{true};
        for (uint32_t i{0}; i < 32; ++i)
        {
            if (std::abs(actual[i] - expected[i]) >= 1e-3F)
            {
                matches = false;
                std::fprintf(stderr, "bindless mismatch component=%u actual=%g expected=%g textureSlots=(%u,%u)\n", i,
                             static_cast<double>(actual[i]), static_cast<double>(expected[i]), texture0.Index,
                             texture1.Index);
            }
        }
        REQUIRE(matches);
        device->UnmapBuffer(output);

        device->DestroyBuffer(output);
        device->DestroyTexture(texture1);
        device->DestroyTexture(texture0);
        device->DestroyPipeline(pipeline);
        REQUIRE(device->BindlessHeap().UsedTextures() == texturesBefore);
    }
} // namespace

int main()
{
    TestResourceLifetimeAccounting();
    TestTextureSelectionAndSampling();
    std::printf("LightRHI bindless tests passed.\n");
    return 0;
}
