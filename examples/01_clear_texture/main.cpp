// 01_clear_texture — create a 2D texture, clear it on the GPU, read back on CPU.
// Exercises: buffer/texture creation, transitions, clear commands, readback.

// Standard headers before module imports (required by Homebrew LLVM / libc++).
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

import lightRHI;

int main()
{
    try
    {
        auto device = rhi::CreateDevice({
            .EnableValidation = true,
            .AppName          = "01_clear_texture",
        });

        constexpr uint32_t    kWidth{256};
        constexpr uint32_t    kHeight{256};
        constexpr rhi::Format kFmt{rhi::Format::RGBA8Unorm};
        constexpr uint64_t    kBufSize{kWidth * kHeight * 4};

        // ---- Create target texture ----
        auto tex = device->CreateTexture(rhi::TextureDesc{
            .Format       = kFmt,
            .Extent       = {kWidth, kHeight, 1},
            .Usage        = rhi::TextureUsage::TransferDst | rhi::TextureUsage::TransferSrc,
            .InitialState = rhi::ResourceState::Undefined,
            .DebugName    = "ClearTarget",
        });

        // ---- Create readback buffer ----
        auto readback{device->CreateBuffer(rhi::BufferDesc{
            .Size       = kBufSize,
            .Usage      = rhi::BufferUsage::TransferDst,
            .MemoryType = rhi::MemoryType::GpuToCpu,
            .DebugName  = "Readback",
        })};

        // ---- Record commands ----
        auto cmd{device->CreateCommandList(rhi::QueueType::Graphics, "ClearCmds")};
        cmd->Begin();

        cmd->Transition(tex, rhi::ResourceState::Undefined, rhi::ResourceState::TransferDst);

        // G = 128/255 so the unorm round-trip is exact (0.5f rounds ambiguously).
        cmd->ClearTexture(tex, rhi::ClearColor{0.f, 128.f / 255.f, 1.f, 1.f});

        cmd->Transition(tex, rhi::ResourceState::TransferDst, rhi::ResourceState::TransferSrc);

        cmd->CopyTextureToBuffer(tex,
                                 rhi::TextureCopyRegion{
                                     .Extent = {kWidth, kHeight, 1},
                                 },
                                 readback, 0);

        cmd->End();

        auto fence{device->Submit(*cmd)};
        device->WaitForFence(fence);

        // ---- Verify on CPU ----
        auto        mapped{device->MapBuffer(readback)};
        const auto *pixels = mapped.As<const uint8_t>();

        bool ok{true};
        for (uint32_t i = 0; i < kWidth * kHeight; ++i)
        {
            if (pixels[i * 4 + 0] != 0 ||   // R = 0
                pixels[i * 4 + 1] != 128 || // G = 128/255
                pixels[i * 4 + 2] != 255 || // B = 1
                pixels[i * 4 + 3] != 255)   // A = 1
            {
                ok = false;
                break;
            }
        }
        device->UnmapBuffer(readback);

        std::printf("Clear texture %s\n", ok ? "PASSED" : "FAILED");

        device->DestroyTexture(tex);
        device->DestroyBuffer(readback);
        device->WaitIdle();

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
