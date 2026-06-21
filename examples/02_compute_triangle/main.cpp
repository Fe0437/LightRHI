// 02_compute_triangle — rasterize a triangle into an offscreen RGBA8 texture
//   using a minimal bindless graphics pipeline.
//
// Shaders are authored once in Slang (shaders/triangle.slang) and compiled by
// the build to the active backend's format — SPIR-V for Vulkan, MSL for Metal.
// The compiled artifact is loaded via the public rhi::ToShaderDesc() path (see
// example_shader_loader.h), so this example is backend-neutral.
//
// Pattern:
//   1. Create vertex buffer with BDA, push GPU address as root constant.
//   2. Pipeline reads vertices directly via GpuAddress (no vertex binding).
//   3. Render into a color attachment, read back and save as raw RGBA bytes.

// Standard headers before module imports (required by Homebrew LLVM / libc++).
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

import lightRHI;

#include "../example_shader_loader.h"

// ---------------------------------------------------------------------------

struct Vertex
{
    float x, y, r, g, b;
};

struct RootConstants
{
    rhi::GpuAddress vertices;
};

int main()
{
    try
    {
        auto device = rhi::CreateDevice({
            .EnableValidation = true,
            .AppName          = "02_compute_triangle",
        });

        constexpr uint32_t W{512}, H{512};

        // ---- Vertex buffer (GPU-accessible, BDA) ----
        const std::array<Vertex, 3> triangleVerts{{
            {0.f, 0.5f, 1.f, 0.f, 0.f},
            {0.5f, -0.5f, 0.f, 1.f, 0.f},
            {-0.5f, -0.5f, 0.f, 0.f, 1.f},
        }};

        auto vertBuf = device->CreateBuffer(rhi::BufferDesc{
            .Size       = sizeof(triangleVerts),
            .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferDst,
            .MemoryType = rhi::MemoryType::GpuOnly,
            .DebugName  = "TriangleVerts",
        });
        device->UploadBuffer(vertBuf, triangleVerts.data(), sizeof(triangleVerts));

        // ---- Render target ----
        auto colorTex = device->CreateTexture(rhi::RenderTarget2D(W, H, rhi::Format::RGBA8Unorm, "ColorTarget"));

        auto readback = device->CreateBuffer(rhi::BufferDesc{
            .Size       = W * H * 4,
            .Usage      = rhi::BufferUsage::TransferDst,
            .MemoryType = rhi::MemoryType::GpuToCpu,
            .DebugName  = "Readback",
        });

        // ---- Pipeline ----
        // Shaders authored in Slang, compiled by the build to this backend's
        // format and loaded from RHI_EXAMPLE_SHADER_DIR.
        const auto vertShader{
            rhiexample::loadShaderArtifact("triangle_vert_main", "vert_main", rhi::ShaderStage::Vertex)};
        const auto fragShader{
            rhiexample::loadShaderArtifact("triangle_frag_main", "frag_main", rhi::ShaderStage::Fragment)};

        auto pipeline = device->CreateGraphicsPipeline(rhi::GraphicsPipelineDesc{
            .VertexShader      = vertShader,
            .FragmentShader    = fragShader,
            .Rasterizer        = {.CullMode = rhi::CullMode::None},
            .DepthStencil      = rhi::DepthDisabled(),
            .ColorBlend        = {rhi::BlendDisabled()},
            .ColorFormats      = {rhi::Format::RGBA8Unorm},
            .PushConstantBytes = static_cast<uint32_t>(sizeof(RootConstants)),
            .DebugName         = "TrianglePipeline",
        });

        // ---- Record ----
        auto cmd = device->CreateCommandList(rhi::QueueType::Graphics, "TriangleCmds");
        cmd->Begin();

        cmd->BeginRendering(rhi::RenderingDesc{
            .Color      = {{.Texture = colorTex, .ClearValue = {0.1f, 0.1f, 0.1f, 1.f}}},
            .RenderArea = {W, H},
        });

        cmd->SetPipeline(pipeline);
        cmd->SetViewport({.Width = static_cast<float>(W), .Height = static_cast<float>(H)});
        cmd->SetScissor({.Width = W, .Height = H});

        cmd->SetPushConstants(RootConstants{
            .vertices = device->BufferAddress(vertBuf),
        });

        cmd->Draw(3);
        cmd->EndRendering();

        cmd->CopyTextureToBuffer(colorTex, rhi::TextureCopyRegion{.Extent = {W, H, 1}}, readback, 0);

        cmd->End();

        auto fence = device->Submit(*cmd);
        device->WaitForFence(fence);

        // ---- Report ----
        [[maybe_unused]] auto mapped = device->MapBuffer(readback);
        std::printf("Triangle rendered — %zu bytes ready for output\n", static_cast<size_t>(W * H * 4));
        device->UnmapBuffer(readback);

        // ---- Cleanup ----
        device->DestroyPipeline(pipeline);
        device->DestroyTexture(colorTex);
        device->DestroyBuffer(vertBuf);
        device->DestroyBuffer(readback);
        device->WaitIdle();

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
