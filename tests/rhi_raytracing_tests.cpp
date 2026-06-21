// rhi_raytracing_tests.cpp — acceleration structure + inline ray-query tests.
//
// Standard headers must precede module imports to avoid include-guard
// isolation issues (__promote_t redefinition) with LLVM libc++ and C++23 modules.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

import lightRHI;

#include "shader_artifact_loader.h"

// ---------------------------------------------------------------------------
// Minimal test harness (mirrors rhi_tests.cpp)
// ---------------------------------------------------------------------------

#define REQUIRE(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "FAIL: %s  [%s:%d]\n", #expr, __FILE__, __LINE__);                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (false)

#define TEST(name) void test_##name()

namespace
{
    // Single CCW triangle in the XY plane at Z=0: (0,0,0), (1,0,0), (0,1,0).
    // A ray from (0.25, 0.25, -1) along +Z hits it at t == 1.0 exactly.
    constexpr float kTriangleVertices[9] = {
        0.f, 0.f, 0.f, //
        1.f, 0.f, 0.f, //
        0.f, 1.f, 0.f, //
    };
    constexpr float kExpectedHitT{1.0f};
    constexpr float kEpsilon{1e-4f};

    // 3 separate, non-overlapping CCW triangles in the XY plane at Z=0, each
    // occupying X in [2*i, 2*i+1], Y in [0,1] — a single non-indexed BLAS
    // with more than one triangle, unlike kTriangleVertices above. Matches
    // the ray origins hardcoded in ray_query_multi_triangle.slang.
    constexpr float kMultiTriangleVertices[27] = {
        0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, // triangle 0
        2.f, 0.f, 0.f, 3.f, 0.f, 0.f, 2.f, 1.f, 0.f, // triangle 1
        4.f, 0.f, 0.f, 5.f, 0.f, 0.f, 4.f, 1.f, 0.f, // triangle 2
    };
    constexpr uint32_t kMultiTriangleThreadCount{4}; // 3 hits + 1 deliberate miss
} // namespace

// ---------------------------------------------------------------------------

TEST(build_acceleration_structure)
{
    auto device = rhi::CreateDevice({});

    if (!device->SupportsRayTracing())
    {
        std::printf("  build_acceleration_structure: SKIPPED (device does not support ray tracing)\n");
        return;
    }

    auto vtxBuf = device->CreateBuffer({
        .Size       = sizeof(kTriangleVertices),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "triangle_vertices",
    });
    REQUIRE(vtxBuf.Valid());
    device->UploadBuffer(vtxBuf, kTriangleVertices, sizeof(kTriangleVertices));
    auto vtxAddr = device->BufferAddress(vtxBuf);
    REQUIRE(vtxAddr.Valid());

    auto blasDesc =
        rhi::BlasFromTriangleBuffer(vtxAddr, /*vertexStride=*/12, /*vertexCount=*/3, /*indexBufferAddress=*/{},
                                    /*indexCount=*/0, rhi::IndexType::Uint32, "triangle_blas");

    auto blasSizes = device->QueryAccelerationStructureBuildSizes(blasDesc);
    REQUIRE(blasSizes.AccelerationStructureSize > 0);
    REQUIRE(blasSizes.BuildScratchSize > 0);

    auto blasScratch = device->CreateBuffer({
        .Size       = blasSizes.BuildScratchSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "blas_scratch",
    });
    REQUIRE(blasScratch.Valid());

    auto blas = device->CreateAccelerationStructure(blasDesc);
    REQUIRE(blas.Valid());

    auto cmd = device->CreateCommandList(rhi::QueueType::Compute, "build_blas_cmd");
    cmd->Begin();
    cmd->BuildAccelerationStructure(blas, blasDesc, blasScratch);
    cmd->Transition(blas, rhi::ResourceState::AccelerationStructureWrite,
                    rhi::ResourceState::AccelerationStructureRead);
    cmd->FlushBarriers();
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    // Single instance, identity transform.
    std::vector<rhi::AccelerationStructureInstance> instances{
        rhi::AccelerationStructureInstance{.Blas = blas},
    };
    auto tlasDesc = rhi::TlasFromInstances(std::move(instances), "triangle_tlas");

    auto tlasSizes = device->QueryAccelerationStructureBuildSizes(tlasDesc);
    REQUIRE(tlasSizes.AccelerationStructureSize > 0);
    REQUIRE(tlasSizes.BuildScratchSize > 0);

    auto tlasScratch = device->CreateBuffer({
        .Size       = tlasSizes.BuildScratchSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "tlas_scratch",
    });
    REQUIRE(tlasScratch.Valid());

    auto tlas = device->CreateAccelerationStructure(tlasDesc);
    REQUIRE(tlas.Valid());

    auto cmd2 = device->CreateCommandList(rhi::QueueType::Compute, "build_tlas_cmd");
    cmd2->Begin();
    cmd2->BuildAccelerationStructure(tlas, tlasDesc, tlasScratch);
    cmd2->Transition(tlas, rhi::ResourceState::AccelerationStructureWrite,
                     rhi::ResourceState::AccelerationStructureRead);
    cmd2->FlushBarriers();
    cmd2->End();
    device->WaitForFence(device->Submit(*cmd2));

    REQUIRE(device->AccelerationStructureAddress(blas).Valid());
    REQUIRE(device->AccelerationStructureAddress(tlas).Valid());

    device->DestroyAccelerationStructure(tlas);
    device->DestroyAccelerationStructure(blas);
    device->DestroyBuffer(tlasScratch);
    device->DestroyBuffer(blasScratch);
    device->DestroyBuffer(vtxBuf);
    std::printf("  build_acceleration_structure: BLAS + TLAS built, handles and addresses valid\n");
}

TEST(ray_query_compute_dispatch)
{
    auto device = rhi::CreateDevice({});

    if (!device->SupportsRayTracing())
    {
        std::printf("  ray_query_compute_dispatch: SKIPPED (device does not support ray tracing)\n");
        return;
    }

    auto vtxBuf = device->CreateBuffer({
        .Size       = sizeof(kTriangleVertices),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "triangle_vertices",
    });
    device->UploadBuffer(vtxBuf, kTriangleVertices, sizeof(kTriangleVertices));
    auto vtxAddr = device->BufferAddress(vtxBuf);

    auto blasDesc    = rhi::BlasFromTriangleBuffer(vtxAddr, 12, 3, {}, 0, rhi::IndexType::Uint32, "triangle_blas");
    auto blasSizes   = device->QueryAccelerationStructureBuildSizes(blasDesc);
    auto blasScratch = device->CreateBuffer({
        .Size       = blasSizes.BuildScratchSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuOnly,
    });
    auto blas        = device->CreateAccelerationStructure(blasDesc);

    std::vector<rhi::AccelerationStructureInstance> instances{rhi::AccelerationStructureInstance{
        .Blas = blas, .Flags = rhi::AccelerationStructureInstanceFlags::ForceOpaque}};
    auto tlasDesc    = rhi::TlasFromInstances(std::move(instances), "triangle_tlas");
    auto tlasSizes   = device->QueryAccelerationStructureBuildSizes(tlasDesc);
    auto tlasScratch = device->CreateBuffer({
        .Size       = tlasSizes.BuildScratchSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuOnly,
    });
    auto tlas        = device->CreateAccelerationStructure(tlasDesc);

    auto buildCmd = device->CreateCommandList(rhi::QueueType::Compute, "build_cmd");
    buildCmd->Begin();
    buildCmd->BuildAccelerationStructure(blas, blasDesc, blasScratch);
    buildCmd->Transition(blas, rhi::ResourceState::AccelerationStructureWrite,
                         rhi::ResourceState::AccelerationStructureRead);
    buildCmd->FlushBarriers();
    buildCmd->BuildAccelerationStructure(tlas, tlasDesc, tlasScratch);
    buildCmd->Transition(tlas, rhi::ResourceState::AccelerationStructureWrite,
                         rhi::ResourceState::AccelerationStructureRead);
    buildCmd->FlushBarriers();
    buildCmd->End();
    device->WaitForFence(device->Submit(*buildCmd));

    auto outBuf = device->CreateBuffer({
        .Size       = sizeof(float),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "ray_query_output",
    });
    REQUIRE(outBuf.Valid());
    auto outAddr = device->BufferAddress(outBuf);
    REQUIRE(outAddr.Valid());

    auto pipeline = device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("ray_query_triangle", "ray_query_main", rhi::ShaderStage::Compute),
        .DebugName = "ray_query_pso",
    });
    REQUIRE(pipeline.Valid());

    // The AS travels through push constants as its bindless 8-byte handle —
    // no per-dispatch bind call, exactly like the buffer BDA next to it.
    struct alignas(8) PC
    {
        uint64_t outputAddr;
        uint64_t accel; // DescriptorHandle<RaytracingAccelerationStructure> in the shader
    };
    PC pc{outAddr.Address, device->AccelerationStructureAddress(tlas).Address};

    auto cmd = device->CreateCommandList(rhi::QueueType::Compute, "ray_query_cmd");
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    cmd->SetPushConstants(&pc, sizeof(pc), 0);
    cmd->Dispatch(1, 1, 1);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped = device->MapBuffer(outBuf);
    REQUIRE(mapped.Data != nullptr);
    float hitT{*mapped.As<float>()};
    device->UnmapBuffer(outBuf);

    REQUIRE(hitT > 0.f); // hit, not a miss (-1.0)
    REQUIRE(std::abs(hitT - kExpectedHitT) < kEpsilon);

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(outBuf);
    device->DestroyAccelerationStructure(tlas);
    device->DestroyAccelerationStructure(blas);
    device->DestroyBuffer(tlasScratch);
    device->DestroyBuffer(blasScratch);
    device->DestroyBuffer(vtxBuf);
    std::printf("  ray_query_compute_dispatch: hit t=%.4f (expected %.4f)\n", static_cast<double>(hitT),
                static_cast<double>(kExpectedHitT));
}

TEST(ray_query_multi_triangle_mesh)
{
    auto device = rhi::CreateDevice({});

    if (!device->SupportsRayTracing())
    {
        std::printf("  ray_query_multi_triangle_mesh: SKIPPED (device does not support ray tracing)\n");
        return;
    }

    auto vtxBuf = device->CreateBuffer({
        .Size       = sizeof(kMultiTriangleVertices),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "multi_triangle_vertices",
    });
    device->UploadBuffer(vtxBuf, kMultiTriangleVertices, sizeof(kMultiTriangleVertices));
    auto vtxAddr = device->BufferAddress(vtxBuf);

    // 9 vertices / 3 per triangle = 3 triangles in one BLAS — exercises
    // per-primitive indexing (MetalAccelerationStructureTriangleGeometryDescriptor::
    // triangleCount / Vulkan's primitiveCount) beyond the single-triangle case.
    auto blasDesc  = rhi::BlasFromTriangleBuffer(vtxAddr, 12, 9, {}, 0, rhi::IndexType::Uint32, "multi_triangle_blas");
    auto blasSizes = device->QueryAccelerationStructureBuildSizes(blasDesc);
    auto blasScratch = device->CreateBuffer({
        .Size       = blasSizes.BuildScratchSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuOnly,
    });
    auto blas        = device->CreateAccelerationStructure(blasDesc);

    std::vector<rhi::AccelerationStructureInstance> instances{rhi::AccelerationStructureInstance{
        .Blas = blas, .Flags = rhi::AccelerationStructureInstanceFlags::ForceOpaque}};
    auto tlasDesc    = rhi::TlasFromInstances(std::move(instances), "multi_triangle_tlas");
    auto tlasSizes   = device->QueryAccelerationStructureBuildSizes(tlasDesc);
    auto tlasScratch = device->CreateBuffer({
        .Size       = tlasSizes.BuildScratchSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuOnly,
    });
    auto tlas        = device->CreateAccelerationStructure(tlasDesc);

    auto buildCmd = device->CreateCommandList(rhi::QueueType::Compute, "build_multi_cmd");
    buildCmd->Begin();
    buildCmd->BuildAccelerationStructure(blas, blasDesc, blasScratch);
    buildCmd->Transition(blas, rhi::ResourceState::AccelerationStructureWrite,
                         rhi::ResourceState::AccelerationStructureRead);
    buildCmd->FlushBarriers();
    buildCmd->BuildAccelerationStructure(tlas, tlasDesc, tlasScratch);
    buildCmd->Transition(tlas, rhi::ResourceState::AccelerationStructureWrite,
                         rhi::ResourceState::AccelerationStructureRead);
    buildCmd->FlushBarriers();
    buildCmd->End();
    device->WaitForFence(device->Submit(*buildCmd));

    auto outBuf  = device->CreateBuffer({
        .Size       = kMultiTriangleThreadCount * sizeof(float),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "ray_query_multi_output",
    });
    auto outAddr = device->BufferAddress(outBuf);

    auto pipeline = device->CreateComputePipeline({
        .Shader = rhitest::loadShaderArtifact("ray_query_multi_triangle", "ray_query_main", rhi::ShaderStage::Compute),
        .DebugName = "ray_query_multi_pso",
    });
    REQUIRE(pipeline.Valid());

    struct alignas(8) PC
    {
        uint64_t outputAddr;
        uint64_t accel;
    };
    PC pc{outAddr.Address, device->AccelerationStructureAddress(tlas).Address};

    auto cmd = device->CreateCommandList(rhi::QueueType::Compute, "ray_query_multi_cmd");
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    cmd->SetPushConstants(&pc, sizeof(pc), 0);
    cmd->Dispatch(1, 1, 1);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped = device->MapBuffer(outBuf);
    REQUIRE(mapped.Data != nullptr);
    const auto *results = mapped.As<float>();

    // Threads 0-2: one ray per triangle, each hits at t == 1.0.
    for (uint32_t i = 0; i < 3; ++i)
    {
        REQUIRE(results[i] > 0.f);
        REQUIRE(std::abs(results[i] - kExpectedHitT) < kEpsilon);
    }
    // Thread 3: ray aimed outside all triangles — must miss.
    REQUIRE(results[3] < 0.f);

    device->UnmapBuffer(outBuf);

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(outBuf);
    device->DestroyAccelerationStructure(tlas);
    device->DestroyAccelerationStructure(blas);
    device->DestroyBuffer(tlasScratch);
    device->DestroyBuffer(blasScratch);
    device->DestroyBuffer(vtxBuf);
    std::printf("  ray_query_multi_triangle_mesh: 3/3 triangle hits at t=%.4f, 1/1 miss confirmed\n",
                static_cast<double>(kExpectedHitT));
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

int main()
{
    std::printf("── Ray tracing tests ───────────────────────────\n");
    test_build_acceleration_structure();
    test_ray_query_compute_dispatch();
    test_ray_query_multi_triangle_mesh();
    std::printf("\nAll ray tracing tests passed.\n");
    return 0;
}
