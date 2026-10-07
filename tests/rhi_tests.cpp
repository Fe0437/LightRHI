// rhi_tests.cpp — unit / integration tests for LightRHI

// Standard headers must precede module imports to avoid include-guard
// isolation issues (__promote_t redefinition) with LLVM libc++ and C++23 modules.
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

import lightRHI;

#include "device_result.h"
#include "gpu_flags.h"

// ---------------------------------------------------------------------------
// Shader artifacts (compiled by tools/compile_shaders.py from tests/shaders/
// shaders.json, loaded at runtime by shader_artifact_loader.h). No shader
// source is embedded in this translation unit.
// ---------------------------------------------------------------------------
#include "shader_artifact_loader.h"

// ---------------------------------------------------------------------------
// Minimal test harness
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

// ---------------------------------------------------------------------------
// Smoke tests — no GPU required
// ---------------------------------------------------------------------------

TEST(handles_are_invalid_by_default)
{
    rhi::BufferHandle             buf{};
    rhi::TextureHandle            tex{};
    rhi::SamplerHandle            smp{};
    rhi::PipelineHandle           pip{};
    rhi::TimestampQueryPoolHandle timestamps{};
    rhi::FenceHandle              fen{};
    rhi::GpuAddress               adr{};

    REQUIRE(!buf.Valid());
    REQUIRE(!tex.Valid());
    REQUIRE(!smp.Valid());
    REQUIRE(!pip.Valid());
    REQUIRE(!timestamps.Valid());
    REQUIRE(!fen.Valid());
    REQUIRE(!adr.Valid());
}

TEST(mapped_buffer_views_its_bytes)
{
    std::array<std::byte, 8> storage{};
    const rhi::MappedBuffer  mapped{.Data = storage.data(), .Size = storage.size()};

    REQUIRE(mapped.Bytes().data() == storage.data());
    REQUIRE(mapped.Bytes().size() == storage.size());
    REQUIRE(rhi::MappedBuffer{}.Bytes().empty());
}

// A value template that accepted a span would upload the span's own pointer and size instead of the
// bytes it names, so spans must only ever reach the byte overload.
template <typename T>
concept PushConstantValue =
    requires(rhi::ICommandList &list, const T &value) { list.template SetPushConstants<T>(value); };
static_assert(PushConstantValue<float>);
static_assert(!PushConstantValue<std::span<const std::byte>>);
static_assert(!PushConstantValue<std::span<std::byte>>);

TEST(format_helpers)
{
    REQUIRE(rhi::IsDepthFormat(rhi::Format::D32Float));
    REQUIRE(rhi::IsDepthFormat(rhi::Format::D24UnormS8Uint));
    REQUIRE(!rhi::IsDepthFormat(rhi::Format::RGBA8Unorm));
    REQUIRE(rhi::IsStencilFormat(rhi::Format::D24UnormS8Uint));
    REQUIRE(!rhi::IsStencilFormat(rhi::Format::D32Float));
}

TEST(resource_state_bitmask)
{
    using RS = rhi::ResourceState;
    auto combined{RS::ShaderRead | RS::UnorderedAccess};
    REQUIRE(rhi::HasState(combined, RS::ShaderRead));
    REQUIRE(rhi::HasState(combined, RS::UnorderedAccess));
    REQUIRE(!rhi::HasState(combined, RS::RenderTarget));
}

TEST(buffer_usage_bitmask)
{
    using BU = rhi::BufferUsage;
    auto use{BU::Storage | BU::DeviceAddress | BU::TransferDst};
    REQUIRE(rhi::HasUsage(use, BU::Storage));
    REQUIRE(rhi::HasUsage(use, BU::DeviceAddress));
    REQUIRE(!rhi::HasUsage(use, BU::Vertex));
}

TEST(gpu_boolean_flags)
{
    constexpr rhi::GpuFlags kVisible{1U << 0U};
    constexpr rhi::GpuFlags kSelected{1U << 1U};
    rhi::GpuFlags           flags{0};

    rhi::SetFlag(flags, kVisible, true);
    REQUIRE(rhi::HasFlag(flags, kVisible));
    REQUIRE(!rhi::HasFlag(flags, kSelected));

    flags |= rhi::FlagIf(kSelected, true);
    REQUIRE(rhi::HasFlag(flags, kVisible | kSelected));

    rhi::SetFlag(flags, kVisible, false);
    REQUIRE(!rhi::HasFlag(flags, kVisible));
    REQUIRE(rhi::HasFlag(flags, kSelected));
}

TEST(sampler_presets_are_valid)
{
    auto lr{rhi::LinearRepeat()};
    REQUIRE(lr.MinFilter == rhi::SamplerFilter::Linear);
    REQUIRE(lr.AddressU == rhi::SamplerAddressMode::Repeat);

    auto nc{rhi::NearestClamp()};
    REQUIRE(nc.MinFilter == rhi::SamplerFilter::Nearest);
    REQUIRE(nc.AddressU == rhi::SamplerAddressMode::ClampToEdge);

    auto sh{rhi::ShadowSampler()};
    REQUIRE(sh.CompareEnable);
    REQUIRE(sh.CompareOp == rhi::CompareOp::LessEqual);
}

TEST(blend_presets)
{
    REQUIRE(!rhi::BlendDisabled().Enable);
    REQUIRE(rhi::BlendAlphaPremultiplied().Enable);
    REQUIRE(rhi::BlendAlphaTraditional().Enable);
}

TEST(gpu_address_offset)
{
    rhi::GpuAddress base{1000};
    auto            shifted{base.Offset(256)};
    REQUIRE(shifted.Address == 1256);
    REQUIRE(shifted.Valid());
}

TEST(shader_library_refuses_bad_sizes)
{
    // A library's size is checked before anything is allocated for it: an empty file and one
    // beyond the bound are refused, and one exactly at the bound is read.
    std::error_code error;
    const auto      directory{std::filesystem::temp_directory_path(error) / "lightrhi_shader_library_test"};
    std::filesystem::create_directories(directory, error);
    const auto write{[&](const char *name, const std::size_t bytes)
                     {
                         const auto              path{directory / name};
                         std::ofstream           file{path, std::ios::binary};
                         const std::vector<char> contents(bytes, 'x');
                         file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
                         return path.string();
                     }};

    const auto missing{rhi::ShaderLibrary::Read((directory / "missing.spv").string(), rhi::ShaderFormat::Spirv)};
    REQUIRE(!missing && missing.error() == rhi::ShaderArtifactError::NotFound);
    const auto empty{rhi::ShaderLibrary::Read(write("empty.spv", 0), rhi::ShaderFormat::Spirv)};
    REQUIRE(!empty && empty.error() == rhi::ShaderArtifactError::InvalidSize);
    const auto oversized{rhi::ShaderLibrary::Read(write("large.spv", 64), rhi::ShaderFormat::Spirv, 32)};
    REQUIRE(!oversized && oversized.error() == rhi::ShaderArtifactError::InvalidSize);
    const auto atBound{rhi::ShaderLibrary::Read(write("bound.spv", 32), rhi::ShaderFormat::Spirv, 32)};
    REQUIRE(atBound.has_value());

    std::filesystem::remove_all(directory, error);
}

TEST(texture_convenience_constructors)
{
    auto t{rhi::Texture2D(1920, 1080, rhi::Format::RGBA16Float)};
    REQUIRE(t.Extent.Width == 1920);
    REQUIRE(t.Extent.Height == 1080);
    REQUIRE(t.Format == rhi::Format::RGBA16Float);

    auto rt{rhi::RenderTarget2D(512, 512)};
    REQUIRE(rhi::HasUsage(rt.Usage, rhi::TextureUsage::RenderTarget));

    auto dt{rhi::DepthTarget2D(512, 512)};
    REQUIRE(rhi::IsDepthFormat(dt.Format));
    REQUIRE(rhi::HasUsage(dt.Usage, rhi::TextureUsage::DepthStencil));
}

// ---------------------------------------------------------------------------
// Device integration tests (require a real GPU)
// ---------------------------------------------------------------------------

TEST(shared_device_reuses_live_device)
{
    auto first{rhitest::Required(rhi::AcquireSharedDevice({}))};
    auto second{rhitest::Required(rhi::AcquireSharedDevice({}))};
    REQUIRE(static_cast<bool>(first));
    REQUIRE(static_cast<bool>(second));

    rhi::IDevice *firstDevice{nullptr};
    {
        auto lockedDevice{first->Synchronize()};
        firstDevice = &*lockedDevice;
        REQUIRE(!lockedDevice->AdapterName().empty());
    }
    {
        auto lockedDevice{second->Synchronize()};
        REQUIRE(&*lockedDevice == firstDevice);
    }
}

TEST(device_create_and_buffer)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);
    REQUIRE(!device->AdapterName().empty());
    std::printf("  adapter: %s\n", std::string{device->AdapterName()}.c_str());
    REQUIRE(device->VideoMemoryBytes() > 0);

    auto buf{rhitest::Required(device->CreateBuffer({
        .Size       = 1024,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "test_buffer",
    }))};
    REQUIRE(buf.Valid());

    auto addr{device->BufferAddress(buf)};
    REQUIRE(addr.Valid());

    auto mapped{device->MapBuffer(buf)};
    REQUIRE(mapped.Data != nullptr);
    std::memset(mapped.Data, 0xAB, 64);
    device->UnmapBuffer(buf);

    device->DestroyBuffer(buf);
}

TEST(device_texture_and_sampler)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    auto tex{rhitest::Required(device->CreateTexture(rhi::RenderTarget2D(64, 64)))};
    REQUIRE(tex.Valid());

    auto smp{rhitest::Required(device->CreateSampler(rhi::LinearRepeat()))};
    REQUIRE(smp.Valid());

    device->DestroyTexture(tex);
    device->DestroySampler(smp);
}

TEST(bindless_heap_tracks_resource_lifetimes)
{
    auto  device{rhitest::Required(rhi::CreateDevice({}))};
    auto &heap{device->BindlessHeap()};
    REQUIRE(heap.MaxBuffers() > 0);
    REQUIRE(heap.MaxTextures() > 0);
    REQUIRE(heap.MaxSamplers() > 0);

    const uint32_t buffersBefore{heap.UsedBuffers()};
    const uint32_t texturesBefore{heap.UsedTextures()};
    const uint32_t samplersBefore{heap.UsedSamplers()};

    auto buffer{rhitest::Required(device->CreateBuffer({
        .Size       = 64,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "bindless_lifetime.buffer",
    }))};
    auto texture{rhitest::Required(device->CreateTexture(
        rhi::Texture2D(1, 1, rhi::Format::RGBA8Unorm, rhi::TextureUsage::Sampled, "bindless_lifetime.texture")))};
    auto sampler{rhitest::Required(device->CreateSampler(rhi::NearestClamp()))};

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

TEST(command_list_begin_end)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "test_cmd"))};
    REQUIRE(cmd != nullptr);
    cmd->Begin();
    cmd->End();

    auto fence{device->Submit(*cmd)};
    REQUIRE(fence.Valid());
    device->WaitForFence(fence);
    REQUIRE(device->IsFenceComplete(fence));
}

TEST(timestamp_queries_measure_gpu_work)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    if (!device->SupportsComputeTimestamps())
    {
        std::printf("  timestamp queries unsupported on compute queue -- skipped\n");
        return;
    }

    constexpr uint32_t kCount{1U << 20U};
    auto               buffer{rhitest::Required(device->CreateBuffer({
        .Size       = static_cast<uint64_t>(kCount) * sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "timestamp_query_work",
    }))};
    const auto         pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "timestamp_query_pipeline",
    }))};
    const auto         queries{rhitest::Required(device->CreateTimestampQueryPool(2))};
    REQUIRE(buffer.Valid());
    REQUIRE(pipeline.Valid());
    REQUIRE(queries.Valid());
    REQUIRE(device->TimestampPeriodNanoseconds() > 0.0);

    struct alignas(8) PushConstants
    {
        uint64_t OutputAddress{};
        uint32_t Value{};
        uint32_t Count{};
    };
    const PushConstants constants{
        .OutputAddress = device->BufferAddress(buffer).Address,
        .Value         = 0x5AU,
        .Count         = kCount,
    };

    device->ResetTimestampQueries(queries, 0, 2);
    auto command{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "timestamp_query_command"))};
    command->Begin();
    command->SetPipeline(pipeline);
    command->SetPushConstants(constants);
    command->WriteComputeTimestamp(queries, 0);
    command->Dispatch((kCount + 63U) / 64U, 1, 1);
    command->WriteComputeTimestamp(queries, 1);
    command->End();
    device->WaitForFence(device->Submit(*command));

    std::array<uint64_t, 2> ticks{};
    device->ReadTimestampQueries(queries, 0, ticks);
    REQUIRE(ticks[1] > ticks[0]);
    const double elapsedMs{static_cast<double>(ticks[1] - ticks[0]) * device->TimestampPeriodNanoseconds() / 1.0e6};
    REQUIRE(elapsedMs > 0.0);
    std::printf("  timestamp query: %.3f ms GPU time\n", elapsedMs);

    device->DestroyTimestampQueryPool(queries);
    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(buffer);
}

// ---------------------------------------------------------------------------
// Upload a pattern into a GPU-private buffer, copy to a readback buffer,
// verify bytes on the CPU.
// ---------------------------------------------------------------------------
TEST(buffer_upload_readback)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kSize{256};

    auto src{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::TransferSrc | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "upload_src",
    }))};
    REQUIRE(src.Valid());

    uint8_t pattern[kSize];
    for (uint32_t i{0}; i < kSize; ++i)
    {
        pattern[i] = static_cast<uint8_t>(i & 0xFF);
    }
    device->UploadBuffer(src, pattern);

    auto readback{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "readback",
    }))};
    REQUIRE(readback.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "copy_cmd"))};
    cmd->Begin();
    cmd->CopyBuffer(src, readback, {.SrcOffset = 0, .DstOffset = 0, .Size = kSize});
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(readback)};
    REQUIRE(mapped.Data != nullptr);
    const auto *bytes{static_cast<const uint8_t *>(mapped.Data)};
    for (uint32_t i{0}; i < kSize; ++i)
    {
        REQUIRE(bytes[i] == static_cast<uint8_t>(i & 0xFF));
    }
    device->UnmapBuffer(readback);

    device->DestroyBuffer(src);
    device->DestroyBuffer(readback);
    std::printf("  upload+readback: %u bytes verified\n", kSize);
}

// ---------------------------------------------------------------------------
// A command list recorded again after its submission completed runs the new recording, as
// Begin() promises, so a caller can keep one list per in-flight slot instead of creating one per
// submission.
// ---------------------------------------------------------------------------
TEST(command_list_records_again_after_completion)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kSize{64};
    auto               first{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferSrc | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "reuse_first",
    }))};
    auto               second{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferSrc | rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "reuse_second",
    }))};
    uint8_t            ones[kSize];
    uint8_t            twos[kSize];
    std::memset(ones, 1, kSize);
    std::memset(twos, 2, kSize);
    device->UploadBuffer(first, ones);
    device->UploadBuffer(second, twos);
    auto readback{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "reuse_readback",
    }))};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "reuse_cmd"))};
    for (const rhi::BufferHandle source : {first, second, first})
    {
        cmd->Begin();
        cmd->CopyBuffer(source, readback, {.SrcOffset = 0, .DstOffset = 0, .Size = kSize});
        cmd->End();
        device->WaitForFence(device->Submit(*cmd));

        auto mapped{device->MapBuffer(readback)};
        REQUIRE(mapped.Data != nullptr);
        const uint8_t expected{source == first ? uint8_t{1} : uint8_t{2}};
        const auto   *bytes{static_cast<const uint8_t *>(mapped.Data)};
        for (uint32_t i{0}; i < kSize; ++i)
        {
            REQUIRE(bytes[i] == expected);
        }
        device->UnmapBuffer(readback);
    }

    device->DestroyBuffer(first);
    device->DestroyBuffer(second);
    device->DestroyBuffer(readback);
    std::printf("  command list recorded three times, each recording ran\n");
}

// ---------------------------------------------------------------------------
// Stress: create 1024 buffers, verify uniqueness, destroy, re-create.
// ---------------------------------------------------------------------------
TEST(slot_pool_stress)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t             N{1024};
    std::vector<rhi::BufferHandle> handles(N);
    std::unordered_set<uint64_t>   addresses{};

    for (uint32_t i{0}; i < N; ++i)
    {
        handles[i] = rhitest::Required(device->CreateBuffer({
            .Size       = 64,
            .Usage      = rhi::BufferUsage::Storage,
            .MemoryType = rhi::MemoryType::CpuToGpu,
        }));
        REQUIRE(handles[i].Valid());
        uint64_t addr{device->BufferAddress(handles[i]).Address};
        REQUIRE(addr != 0);
        REQUIRE(addresses.insert(addr).second);
    }

    for (auto h : handles)
    {
        device->DestroyBuffer(h);
    }
    handles.clear();

    for (uint32_t i{0}; i < N; ++i)
    {
        handles.push_back(rhitest::Required(device->CreateBuffer({
            .Size       = 64,
            .Usage      = rhi::BufferUsage::Storage,
            .MemoryType = rhi::MemoryType::CpuToGpu,
        })));
        REQUIRE(handles.back().Valid());
    }

    for (auto h : handles)
    {
        device->DestroyBuffer(h);
    }
    std::printf("  slot pool stress: %u × 2 alloc/free cycles OK\n", N);
}

// ---------------------------------------------------------------------------
// Compute — BDA fill: push a GPU address as root constant, kernel writes
// to it via raw pointer dereference.
// ---------------------------------------------------------------------------
TEST(compute_fill_via_bda)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kCount{64};
    constexpr uint32_t kValue{0xDEADBEEFu};

    auto outBuf{rhitest::Required(device->CreateBuffer({
        .Size       = kCount * sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "compute_output",
    }))};
    REQUIRE(outBuf.Valid());

    auto outAddr{device->BufferAddress(outBuf)};
    REQUIRE(outAddr.Valid());

    auto pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "fill_via_bda_pso",
    }))};
    REQUIRE(pipeline.Valid());

    struct alignas(8) PC
    {
        uint64_t addr{};
        uint32_t value{};
        uint32_t count{};
    };
    PC pc{outAddr.Address, kValue, kCount};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "compute_cmd"))};
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    cmd->SetPushConstants(pc);
    cmd->Dispatch(1, 1, 1);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(outBuf)};
    REQUIRE(mapped.Data != nullptr);
    const auto *words{static_cast<const uint32_t *>(mapped.Data)};
    for (uint32_t i{0}; i < kCount; ++i)
    {
        REQUIRE(words[i] == kValue);
    }
    device->UnmapBuffer(outBuf);

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(outBuf);
    std::printf("  compute BDA fill: %u × 0x%08X verified\n", kCount, kValue);
}

// ---------------------------------------------------------------------------
// Root data belongs to the command that was recorded with it. The GPU reads
// push constants when a command runs, not when it is recorded, so a command
// list that records many commands must give each one its own copy — otherwise
// every dispatch in the list sees whatever was set last. The count here also
// takes the scratch past one block, so the chaining that keeps earlier copies
// alive is exercised rather than assumed.
// ---------------------------------------------------------------------------
TEST(push_constants_belong_to_their_own_dispatch)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kDispatches{200};

    auto outBuf{rhitest::Required(device->CreateBuffer({
        .Size       = kDispatches * sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "per_dispatch_output",
    }))};
    REQUIRE(outBuf.Valid());
    const auto base{device->BufferAddress(outBuf)};
    REQUIRE(base.Valid());

    auto pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "per_dispatch_pso",
    }))};
    REQUIRE(pipeline.Valid());

    struct alignas(8) PC
    {
        uint64_t addr{};
        uint32_t value{};
        uint32_t count{};
    };

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "per_dispatch_cmd"))};
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    for (uint32_t i{0}; i < kDispatches; ++i)
    {
        const PC pc{base.Address + (i * sizeof(uint32_t)), i + 1, 1};
        cmd->SetPushConstants(pc);
        cmd->Dispatch(1, 1, 1);
    }
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(outBuf)};
    REQUIRE(mapped.Data != nullptr);
    const auto *words{static_cast<const uint32_t *>(mapped.Data)};
    for (uint32_t i{0}; i < kDispatches; ++i)
    {
        REQUIRE(words[i] == i + 1);
    }
    device->UnmapBuffer(outBuf);

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(outBuf);
    std::printf("  %u dispatches in one list each kept their own root data\n", kDispatches);
}

// ---------------------------------------------------------------------------
// Command-list resources must be reclaimed and reused on every backend.
// The original regression was observed on Metal 4, where creating fresh
// allocators for every submission eventually exhausted the driver's
// command-buffer storage table during long renders.
// ---------------------------------------------------------------------------
TEST(command_resource_reuse_stress)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kSubmissions{40000};
    constexpr uint32_t kValue{0xA110CA7Eu};

    auto outBuf{rhitest::Required(device->CreateBuffer({
        .Size       = sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "command_resource_reuse_stress.output",
    }))};
    REQUIRE(outBuf.Valid());

    auto pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "command_resource_reuse_stress.pipeline",
    }))};
    REQUIRE(pipeline.Valid());

    struct alignas(8) PushConstants
    {
        uint64_t address{};
        uint32_t value{};
        uint32_t count{};
    };
    const PushConstants pushConstants{
        .address = device->BufferAddress(outBuf).Address,
        .value   = kValue,
        .count   = 1,
    };

    for (uint32_t submission{0}; submission < kSubmissions; ++submission)
    {
        auto cmd{
            rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "command_resource_reuse_stress"))};
        cmd->Begin();
        cmd->SetPipeline(pipeline);
        cmd->SetPushConstants(pushConstants);
        cmd->Dispatch(1, 1, 1);
        cmd->End();
        device->WaitForFence(device->Submit(*cmd));
    }

    auto mapped{device->MapBuffer(outBuf)};
    REQUIRE(mapped.Valid());
    REQUIRE(*static_cast<const uint32_t *>(mapped.Data) == kValue);
    device->UnmapBuffer(outBuf);

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(outBuf);
    std::printf("  command resource reuse: %u submissions completed\n", kSubmissions);
}

// ---------------------------------------------------------------------------
// Compute — bindless fill: kernel reads a buffer's address from the
// bindless heap via a slot index in push constants.
// ---------------------------------------------------------------------------
TEST(compute_bindless_fill)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    // The slot-to-BDA lookup table is a Vulkan representation detail. Metal's
    // heap is its queue residency set and buffers use native BDA directly.
    if (device->BindlessHeap().HeapAddress().Address == 0)
    {
        std::printf("  compute bindless fill: skipped (backend has no bindless heap)\n");
        return;
    }

    constexpr uint32_t kCount{64};

    auto outBuf{rhitest::Required(device->CreateBuffer({
        .Size       = kCount * sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "bindless_output",
    }))};
    REQUIRE(outBuf.Valid());

    uint32_t slot{outBuf.Index};

    auto &heap{device->BindlessHeap()};
    REQUIRE(heap.UsedBuffers() > 0);

    auto pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bindless", "bindless_fill", rhi::ShaderStage::Compute),
        .DebugName = "bindless_fill_pso",
    }))};
    REQUIRE(pipeline.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "bindless_cmd"))};
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    cmd->SetPushConstants(slot);
    cmd->Dispatch(1, 1, 1);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(outBuf)};
    REQUIRE(mapped.Data != nullptr);
    const auto *words{static_cast<const uint32_t *>(mapped.Data)};
    for (uint32_t i{0}; i < kCount; ++i)
    {
        REQUIRE(words[i] == 0xCAFEBABEu);
    }
    device->UnmapBuffer(outBuf);

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(outBuf);
    std::printf("  compute bindless fill: slot %u → 0xCAFEBABE × %u verified\n", slot, kCount);
}

// ---------------------------------------------------------------------------
// Compute — bindless sampled texture: only a DescriptorHandle<Texture2D>
// travels through push constants; there is no per-dispatch texture binding.
// ---------------------------------------------------------------------------
TEST(compute_bindless_texture_sampling)
{
    auto           device{rhitest::Required(rhi::CreateDevice({}))};
    const uint32_t usedTexturesBefore{device->BindlessHeap().UsedTextures()};

    auto pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader          = rhitest::loadShaderArtifact("bindless_texture_test", "bindless_texture_test_main",
                                                       rhi::ShaderStage::Compute),
        .ThreadGroupSize = {8, 1, 1}, // must match bindless_texture_test.slang's numthreads(8, 1, 1)
        .DebugName       = "bindless_texture_test_pso",
    }))};
    REQUIRE(pipeline.Valid());

    // 2x2 RGBA8Unorm checkerboard: row 0 = red, green; row 1 = blue, yellow.
    auto texture0{rhitest::Required(device->CreateTexture(
        rhi::Texture2D(2, 2, rhi::Format::RGBA8Unorm, rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferDst,
                       "bindless_texture_test.checkerboard0")))};
    auto texture1{rhitest::Required(device->CreateTexture(
        rhi::Texture2D(2, 2, rhi::Format::RGBA8Unorm, rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferDst,
                       "bindless_texture_test.checkerboard1")))};
    REQUIRE(texture0.Valid());
    REQUIRE(texture1.Valid());
    REQUIRE(texture0.Index != texture1.Index);
    REQUIRE(device->BindlessHeap().UsedTextures() == usedTexturesBefore + 2);

    constexpr uint8_t pixels0[16] = {
        255, 0,   0,   255, // (0,0) red
        0,   255, 0,   255, // (1,0) green
        0,   0,   255, 255, // (0,1) blue
        255, 255, 0,   255, // (1,1) yellow
    };
    constexpr uint8_t pixels1[16] = {
        0,   255, 255, 255, // (0,0) cyan
        255, 0,   255, 255, // (1,0) magenta
        255, 255, 255, 255, // (0,1) white
        0,   0,   0,   255, // (1,1) black
    };
    device->UploadTexture(texture0, pixels0, /*rowPitch=*/4 * 2, /*slicePitch=*/4 * 2 * 2,
                          rhi::TextureCopyRegion{.Extent = {2, 2, 1}});
    device->UploadTexture(texture1, pixels1, /*rowPitch=*/4 * 2, /*slicePitch=*/4 * 2 * 2,
                          rhi::TextureCopyRegion{.Extent = {2, 2, 1}});

    auto transitionCmd{
        rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "bindless_texture_test.transitionCmd"))};
    transitionCmd->Begin();
    transitionCmd->Transition(texture0, rhi::ResourceState::TransferDst, rhi::ResourceState::ShaderRead);
    transitionCmd->Transition(texture1, rhi::ResourceState::TransferDst, rhi::ResourceState::ShaderRead);
    transitionCmd->FlushBarriers();
    transitionCmd->End();
    device->WaitForFence(device->Submit(*transitionCmd));

    auto output{rhitest::Required(device->CreateBuffer({
        .Size       = 32 * sizeof(float),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "bindless_texture_test.output",
    }))};
    REQUIRE(output.Valid());

    struct alignas(8) PC
    {
        uint64_t outputAddr{};
        uint64_t texture0{};
        uint64_t texture1{};
    };
    PC pc{device->BufferAddress(output).Address, device->TextureAddress(texture0).Address,
          device->TextureAddress(texture1).Address};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "bindless_texture_test.cmd"))};
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    cmd->SetPushConstants(pc);
    cmd->Dispatch(1, 1, 1);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(output)};
    REQUIRE(mapped.Data != nullptr);
    const auto *result{static_cast<const float *>(mapped.Data)};
    // Expected RGBA from exact integer texel loads.
    constexpr float kExpected[32] = {
        1.F, 0.F, 0.F, 1.F, // texel (0,0): red
        0.F, 1.F, 0.F, 1.F, // texel (1,0): green
        0.F, 0.F, 1.F, 1.F, // texel (0,1): blue
        1.F, 1.F, 0.F, 1.F, // texel (1,1): yellow
        0.F, 1.F, 1.F, 1.F, // texel (0,0): cyan
        1.F, 0.F, 1.F, 1.F, // texel (1,0): magenta
        1.F, 1.F, 1.F, 1.F, // texel (0,1): white
        0.F, 0.F, 0.F, 1.F, // texel (1,1): black
    };
    for (int i{0}; i < 32; ++i)
    {
        REQUIRE(std::abs(result[i] - kExpected[i]) < 1e-3F);
    }
    device->UnmapBuffer(output);

    device->DestroyBuffer(output);
    device->DestroyTexture(texture1);
    device->DestroyTexture(texture0);
    REQUIRE(device->BindlessHeap().UsedTextures() == usedTexturesBefore);
    device->DestroyPipeline(pipeline);
    std::printf("  compute bindless texture sampling: two resident 2x2 textures verified\n");
}

// ---------------------------------------------------------------------------
// Render + readback: full-screen triangle in solid red, verify all pixels.
// ---------------------------------------------------------------------------
TEST(render_triangle_readback)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{8}, H{8};

    auto rt{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "rt_8x8",
    }))};
    REQUIRE(rt.Valid());

    constexpr uint64_t kReadbackSize{W * H * 4};
    auto               staging{rhitest::Required(device->CreateBuffer({
        .Size       = kReadbackSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "rt_staging",
    }))};
    REQUIRE(staging.Valid());

    auto gfxPipeline{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("fullscreen", "vert_main", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("fullscreen", "red_frag", rhi::ShaderStage::Fragment),
        .Topology       = rhi::PrimitiveTopology::TriangleList,
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthDisabled(),
        .ColorBlend     = {rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm},
        .DebugName      = "triangle_pso",
    }))};
    REQUIRE(gfxPipeline.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "render_cmd"))};
    cmd->Begin();

    cmd->BeginRendering({
        .Color      = {{
            .Texture    = rt,
            .LoadOp     = rhi::LoadOp::Clear,
            .StoreOp    = rhi::StoreOp::Store,
            .ClearValue = {0.f, 0.f, 0.f, 1.f},
        }},
        .RenderArea = {W, H},
    });
    cmd->SetViewport({0.f, 0.f, float(W), float(H), 0.f, 1.f});
    cmd->SetScissor({0, 0, W, H});
    cmd->SetPipeline(gfxPipeline);
    cmd->Draw(3, 1, 0, 0);
    cmd->EndRendering();

    cmd->Transition(rt, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->CopyTextureToBuffer(rt, {.MipLevel = 0, .ArrayLayer = 0, .Extent = {W, H, 1}}, staging, 0);

    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(staging)};
    REQUIRE(mapped.Data != nullptr);
    const auto *pixels{static_cast<const uint8_t *>(mapped.Data)};
    uint32_t    mismatches{0};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{pixels + i * 4};
        if (p[0] != 255 || p[1] != 0 || p[2] != 0 || p[3] != 255)
        {
            ++mismatches;
        }
    }
    device->UnmapBuffer(staging);

    REQUIRE(mismatches == 0);

    device->DestroyPipeline(gfxPipeline);
    device->DestroyTexture(rt);
    device->DestroyBuffer(staging);
    std::printf("  render readback: %ux%u pixels all red\n", W, H);
}

// ---------------------------------------------------------------------------
// DrawIndexed — quad via 6-index buffer.
// ---------------------------------------------------------------------------
TEST(draw_indexed)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{4}, H{4};

    constexpr uint32_t kIndices[6]{0, 1, 2, 0, 2, 3};
    auto               idxBuf{rhitest::Required(device->CreateBuffer({
        .Size       = sizeof(kIndices),
        .Usage      = rhi::BufferUsage::Index,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "quad_indices",
    }))};
    {
        auto m{device->MapBuffer(idxBuf)};
        std::memcpy(m.Data, kIndices, sizeof(kIndices));
        device->UnmapBuffer(idxBuf);
    }

    auto rt{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "indexed_rt",
    }))};

    constexpr uint64_t kRbSize{W * H * 4};
    auto               rb{rhitest::Required(device->CreateBuffer({
        .Size       = kRbSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "indexed_rb",
    }))};

    auto pso{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("quad", "quad_vert", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("quad", "blue_frag", rhi::ShaderStage::Fragment),
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthDisabled(),
        .ColorBlend     = {rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm},
        .DebugName      = "indexed_pso",
    }))};
    REQUIRE(pso.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "indexed_cmd"))};
    cmd->Begin();
    cmd->BeginRendering({
        .Color      = {{
            .Texture    = rt,
            .LoadOp     = rhi::LoadOp::Clear,
            .StoreOp    = rhi::StoreOp::Store,
            .ClearValue = {0.f, 0.f, 0.f, 1.f},
        }},
        .RenderArea = {W, H},
    });
    cmd->SetViewport({0.f, 0.f, float(W), float(H), 0.f, 1.f});
    cmd->SetScissor({0, 0, W, H});
    cmd->SetPipeline(pso);
    cmd->BindIndexBuffer(idxBuf, 0, rhi::IndexType::Uint32);
    cmd->DrawIndexed(6);
    cmd->EndRendering();
    cmd->Transition(rt, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->CopyTextureToBuffer(rt, {.Extent = {W, H, 1}}, rb, 0);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(rb)};
    REQUIRE(mapped.Data != nullptr);
    const auto *pixels{static_cast<const uint8_t *>(mapped.Data)};
    uint32_t    mismatches{0};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{pixels + i * 4};
        if (p[0] != 0 || p[1] != 0 || p[2] != 255 || p[3] != 255)
        {
            ++mismatches;
        }
    }
    device->UnmapBuffer(rb);
    REQUIRE(mismatches == 0);

    device->DestroyPipeline(pso);
    device->DestroyBuffer(idxBuf);
    device->DestroyTexture(rt);
    device->DestroyBuffer(rb);
    std::printf("  DrawIndexed quad: %ux%u all blue\n", W, H);
}

// ---------------------------------------------------------------------------
// Depth test occlusion — front Draw (z=0.1) red, back Draw (z=0.9) green
// must be discarded by the depth test.
// ---------------------------------------------------------------------------
TEST(depth_test_occlusion)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{8}, H{8};

    auto               colorRt{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "depth_color_rt",
    }))};
    auto               depthRt{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::D32Float,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::DepthStencil,
        .DebugName = "depth_rt",
    }))};
    constexpr uint64_t kRbSize{W * H * 4};
    auto               rb{rhitest::Required(device->CreateBuffer({
        .Size       = kRbSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "depth_rb",
    }))};

    auto redPso{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("depth", "depth_vert", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("fullscreen", "red_frag", rhi::ShaderStage::Fragment),
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthReadWrite(rhi::CompareOp::Less),
        .ColorBlend     = {rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm},
        .DepthFormat    = rhi::Format::D32Float,
        .DebugName      = "red_depth_pso",
    }))};
    auto greenPso{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("depth", "depth_vert", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("fullscreen", "green_frag", rhi::ShaderStage::Fragment),
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthReadWrite(rhi::CompareOp::Less),
        .ColorBlend     = {rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm},
        .DepthFormat    = rhi::Format::D32Float,
        .DebugName      = "green_depth_pso",
    }))};
    REQUIRE(redPso.Valid());
    REQUIRE(greenPso.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "depth_cmd"))};
    cmd->Begin();

    cmd->BeginRendering({
        .Color = {{
            .Texture    = colorRt,
            .LoadOp     = rhi::LoadOp::Clear,
            .StoreOp    = rhi::StoreOp::Store,
            .ClearValue = {0.f, 0.f, 0.f, 1.f},
        }},
        .Depth =
            {
                .Texture    = depthRt,
                .LoadOp     = rhi::LoadOp::Clear,
                .StoreOp    = rhi::StoreOp::DontCare,
                .ClearValue = {1.f, 0},
            },
        .RenderArea = {W, H},
    });
    cmd->SetViewport({0.f, 0.f, float(W), float(H), 0.f, 1.f});
    cmd->SetScissor({0, 0, W, H});

    struct PC
    {
        float depth{};
    };
    cmd->SetPipeline(redPso);
    cmd->SetPushConstants(PC{0.1f});
    cmd->Draw(3);

    cmd->SetPipeline(greenPso);
    cmd->SetPushConstants(PC{0.9f});
    cmd->Draw(3);

    cmd->EndRendering();
    cmd->Transition(colorRt, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->CopyTextureToBuffer(colorRt, {.Extent = {W, H, 1}}, rb, 0);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(rb)};
    REQUIRE(mapped.Data != nullptr);
    const auto *pixels{static_cast<const uint8_t *>(mapped.Data)};
    uint32_t    green_leaks{0};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{pixels + i * 4};
        if (p[0] != 255 || p[1] != 0 || p[2] != 0 || p[3] != 255)
        {
            ++green_leaks;
        }
    }
    device->UnmapBuffer(rb);
    REQUIRE(green_leaks == 0);

    device->DestroyPipeline(redPso);
    device->DestroyPipeline(greenPso);
    device->DestroyTexture(colorRt);
    device->DestroyTexture(depthRt);
    device->DestroyBuffer(rb);
    std::printf("  depth test occlusion: %ux%u all red (green occluded)\n", W, H);
}

// ---------------------------------------------------------------------------
// Multiple render targets.
// ---------------------------------------------------------------------------
TEST(multiple_render_targets)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{4}, H{4};
    constexpr uint64_t kRbSize{W * H * 4};

    auto rt0{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "mrt_rt0",
    }))};
    auto rt1{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "mrt_rt1",
    }))};
    auto rb0{rhitest::Required(device->CreateBuffer({
        .Size       = kRbSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "mrt_rb0",
    }))};
    auto rb1{rhitest::Required(device->CreateBuffer({
        .Size       = kRbSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "mrt_rb1",
    }))};

    auto pso{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("fullscreen", "vert_main", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("mrt", "mrt_frag", rhi::ShaderStage::Fragment),
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthDisabled(),
        .ColorBlend     = {rhi::BlendDisabled(), rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm, rhi::Format::RGBA8Unorm},
        .DebugName      = "mrt_pso",
    }))};
    REQUIRE(pso.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "mrt_cmd"))};
    cmd->Begin();
    cmd->BeginRendering({
        .Color =
            {
                {.Texture    = rt0,
                 .LoadOp     = rhi::LoadOp::Clear,
                 .StoreOp    = rhi::StoreOp::Store,
                 .ClearValue = {0.f, 0.f, 0.f, 1.f}},
                {.Texture    = rt1,
                 .LoadOp     = rhi::LoadOp::Clear,
                 .StoreOp    = rhi::StoreOp::Store,
                 .ClearValue = {0.f, 0.f, 0.f, 1.f}},
            },
        .RenderArea = {W, H},
    });
    cmd->SetViewport({0.f, 0.f, float(W), float(H), 0.f, 1.f});
    cmd->SetScissor({0, 0, W, H});
    cmd->SetPipeline(pso);
    cmd->Draw(3);
    cmd->EndRendering();
    cmd->Transition(rt0, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->Transition(rt1, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->FlushBarriers();
    cmd->CopyTextureToBuffer(rt0, {.Extent = {W, H, 1}}, rb0, 0);
    cmd->CopyTextureToBuffer(rt1, {.Extent = {W, H, 1}}, rb1, 0);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto        m0{device->MapBuffer(rb0)};
    uint32_t    rt0_err{0};
    const auto *p0{static_cast<const uint8_t *>(m0.Data)};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{p0 + i * 4};
        if (p[0] != 255 || p[1] != 0 || p[2] != 0 || p[3] != 255)
        {
            ++rt0_err;
        }
    }
    device->UnmapBuffer(rb0);
    REQUIRE(rt0_err == 0);

    auto        m1{device->MapBuffer(rb1)};
    uint32_t    rt1_err{0};
    const auto *p1{static_cast<const uint8_t *>(m1.Data)};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{p1 + i * 4};
        if (p[0] != 0 || p[1] != 0 || p[2] != 255 || p[3] != 255)
        {
            ++rt1_err;
        }
    }
    device->UnmapBuffer(rb1);
    REQUIRE(rt1_err == 0);

    device->DestroyPipeline(pso);
    device->DestroyTexture(rt0);
    device->DestroyTexture(rt1);
    device->DestroyBuffer(rb0);
    device->DestroyBuffer(rb1);
    std::printf("  MRT: rt0 all red, rt1 all blue\n");
}

// ---------------------------------------------------------------------------
// Texture buffer round-trip — upload a gradient, read back, verify.
// ---------------------------------------------------------------------------
TEST(texture_buffer_roundtrip)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{8}, H{8};
    constexpr uint64_t kSize{W * H * 4};

    uint8_t pattern[kSize];
    for (uint32_t y{0}; y < H; ++y)
    {
        for (uint32_t x{0}; x < W; ++x)
        {
            uint8_t *p{pattern + (y * W + x) * 4};
            p[0] = static_cast<uint8_t>(x * 32);
            p[1] = static_cast<uint8_t>(y * 32);
            p[2] = 128;
            p[3] = 255;
        }
    }

    auto staging{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferSrc,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "tex_staging",
    }))};
    {
        auto m{device->MapBuffer(staging)};
        std::memcpy(m.Data, pattern, kSize);
        device->UnmapBuffer(staging);
    }

    auto tex{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::TransferDst | rhi::TextureUsage::TransferSrc,
        .DebugName = "roundtrip_tex",
    }))};

    auto rb{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "tex_rb",
    }))};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "roundtrip_cmd"))};
    cmd->Begin();
    cmd->CopyBufferToTexture(staging, 0, tex, {.Extent = {W, H, 1}});
    cmd->Transition(tex, rhi::ResourceState::TransferDst, rhi::ResourceState::TransferSrc);
    cmd->CopyTextureToBuffer(tex, {.Extent = {W, H, 1}}, rb, 0);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(rb)};
    REQUIRE(mapped.Data != nullptr);
    const auto *bytes{static_cast<const uint8_t *>(mapped.Data)};
    uint32_t    mismatches{0};
    for (uint64_t i{0}; i < kSize; ++i)
    {
        if (bytes[i] != pattern[i])
        {
            ++mismatches;
        }
    }
    device->UnmapBuffer(rb);
    REQUIRE(mismatches == 0);

    device->DestroyTexture(tex);
    device->DestroyBuffer(staging);
    device->DestroyBuffer(rb);
    std::printf("  texture buffer round-trip: %ux%u OK (%llu bytes)\n", W, H, kSize);
}

// ---------------------------------------------------------------------------
// Compute barrier — kernel A fills, barrier, kernel B XORs → 0xFFFFFFFF.
// ---------------------------------------------------------------------------
TEST(compute_barrier)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kCount{128};

    auto buf{rhitest::Required(device->CreateBuffer({
        .Size       = kCount * sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "barrier_buf",
    }))};
    REQUIRE(buf.Valid());
    auto addr{device->BufferAddress(buf)};

    auto psoA{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_seq", "fill_a", rhi::ShaderStage::Compute),
        .DebugName = "fill_a_pso",
    }))};
    auto psoB{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_seq", "xor_b", rhi::ShaderStage::Compute),
        .DebugName = "xor_b_pso",
    }))};
    REQUIRE(psoA.Valid());
    REQUIRE(psoB.Valid());

    struct alignas(8) PC
    {
        uint64_t addr{};
        uint32_t count{};
    };
    PC pc{addr.Address, kCount};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "barrier_cmd"))};
    cmd->Begin();

    cmd->SetPipeline(psoA);
    cmd->SetPushConstants(pc);
    cmd->Dispatch(2); // 2 groups × 64 threads = kCount (128) elements

    cmd->Transition(buf, rhi::ResourceState::UnorderedAccess, rhi::ResourceState::UnorderedAccess);

    cmd->SetPipeline(psoB);
    cmd->SetPushConstants(pc);
    cmd->Dispatch(2); // 2 groups × 64 threads = kCount (128) elements

    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(buf)};
    REQUIRE(mapped.Data != nullptr);
    const auto *words{static_cast<const uint32_t *>(mapped.Data)};
    uint32_t    errors{0};
    for (uint32_t i{0}; i < kCount; ++i)
    {
        if (words[i] != 0xFFFFFFFFu)
        {
            ++errors;
        }
    }
    device->UnmapBuffer(buf);
    REQUIRE(errors == 0);

    device->DestroyPipeline(psoA);
    device->DestroyPipeline(psoB);
    device->DestroyBuffer(buf);
    std::printf("  compute barrier: %u elements 0xAAAAAAAA^0x55555555=0xFFFFFFFF OK\n", kCount);
}

// ---------------------------------------------------------------------------
// DrawIndirect — fills DrawIndirectArgs on CPU, uses DrawIndirect(count=1).
// ---------------------------------------------------------------------------
TEST(draw_indirect)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{8}, H{8};

    auto               rt{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "indirect_rt",
    }))};
    constexpr uint64_t kRbSize{W * H * 4};
    auto               rb{rhitest::Required(device->CreateBuffer({
        .Size       = kRbSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "indirect_rb",
    }))};

    auto argsBuf{rhitest::Required(device->CreateBuffer({
        .Size       = sizeof(rhi::DrawIndirectArgs),
        .Usage      = rhi::BufferUsage::IndirectArgs,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "draw_args",
    }))};
    {
        auto  m{device->MapBuffer(argsBuf)};
        auto *args{m.As<rhi::DrawIndirectArgs>()};
        *args = {.VertexCount = 3, .InstanceCount = 1, .FirstVertex = 0, .FirstInstance = 0};
        device->UnmapBuffer(argsBuf);
    }

    auto pso{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("fullscreen", "vert_main", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("fullscreen", "green_frag", rhi::ShaderStage::Fragment),
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthDisabled(),
        .ColorBlend     = {rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm},
        .DebugName      = "indirect_pso",
    }))};
    REQUIRE(pso.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "indirect_cmd"))};
    cmd->Begin();
    cmd->BeginRendering({
        .Color      = {{
            .Texture    = rt,
            .LoadOp     = rhi::LoadOp::Clear,
            .StoreOp    = rhi::StoreOp::Store,
            .ClearValue = {0.f, 0.f, 0.f, 1.f},
        }},
        .RenderArea = {W, H},
    });
    cmd->SetViewport({0.f, 0.f, float(W), float(H), 0.f, 1.f});
    cmd->SetScissor({0, 0, W, H});
    cmd->SetPipeline(pso);
    cmd->DrawIndirect(argsBuf, 0, 1);
    cmd->Transition(rt, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->EndRendering();
    cmd->CopyTextureToBuffer(rt, {.Extent = {W, H, 1}}, rb, 0);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(rb)};
    REQUIRE(mapped.Data != nullptr);
    const auto *pixels{static_cast<const uint8_t *>(mapped.Data)};
    uint32_t    mismatches{0};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{pixels + i * 4};
        if (p[0] != 0 || p[1] != 255 || p[2] != 0 || p[3] != 255)
        {
            ++mismatches;
        }
    }
    device->UnmapBuffer(rb);
    REQUIRE(mismatches == 0);

    device->DestroyPipeline(pso);
    device->DestroyTexture(rt);
    device->DestroyBuffer(rb);
    device->DestroyBuffer(argsBuf);
    std::printf("  DrawIndirect: %ux%u all green\n", W, H);
}

// ---------------------------------------------------------------------------
// Explicit barrier before a transfer. Regression: Transition() only queues the
// barrier, so the transfer that depends on it must force a flush first — a
// deferred barrier emitted at end() lands *after* the copy and the copy reads
// the buffer before the shader's writes are visible.
// ---------------------------------------------------------------------------
TEST(explicit_barrier_before_copy)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kCount{64};
    constexpr uint32_t kValue{0x12345678u};
    constexpr uint64_t kSize{kCount * sizeof(uint32_t)};

    auto buf{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferSrc,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "barrier_copy_src",
    }))};
    auto staging{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "barrier_copy_dst",
    }))};
    REQUIRE(buf.Valid());
    REQUIRE(staging.Valid());

    auto pso{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "barrier_copy_pso",
    }))};
    REQUIRE(pso.Valid());

    struct alignas(8) PC
    {
        uint64_t addr{};
        uint32_t value{};
        uint32_t count{};
    };
    PC pc{device->BufferAddress(buf).Address, kValue, kCount};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "barrier_copy_cmd"))};
    cmd->Begin();
    cmd->SetPipeline(pso);
    cmd->SetPushConstants(pc);
    cmd->Dispatch(1);
    cmd->Transition(buf, rhi::ResourceState::UnorderedAccess, rhi::ResourceState::TransferSrc);
    cmd->CopyBuffer(buf, staging, {.SrcOffset = 0, .DstOffset = 0, .Size = kSize});
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(staging)};
    REQUIRE(mapped.Data != nullptr);
    const auto *words{static_cast<const uint32_t *>(mapped.Data)};
    uint32_t    errors{0};
    for (uint32_t i{0}; i < kCount; ++i)
    {
        if (words[i] != kValue)
        {
            ++errors;
        }
    }
    device->UnmapBuffer(staging);
    REQUIRE(errors == 0);

    device->DestroyPipeline(pso);
    device->DestroyBuffer(buf);
    device->DestroyBuffer(staging);
    std::printf("  explicit barrier before CopyBuffer: %u words verified\n", kCount);
}

// ---------------------------------------------------------------------------
// A submission given a WaitFence starts only after that fence's work. A queue
// may overlap command lists submitted one after another, so a write submitted
// right after a copy could overwrite the buffer before the copy has read it.
// The copy is large enough to give it time; it must hold only the old value.
// ---------------------------------------------------------------------------
TEST(submission_waits_for_its_fence)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t kCount{1u << 21};
    constexpr uint64_t kSize{uint64_t{kCount} * sizeof(uint32_t)};
    constexpr uint32_t kRounds{8};

    auto buf{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferSrc,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "wait_fence_src",
    }))};
    auto staging{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "wait_fence_dst",
    }))};
    auto pso{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "wait_fence_pso",
    }))};

    struct alignas(8) PC
    {
        uint64_t addr{};
        uint32_t value{};
        uint32_t count{};
    };
    const uint64_t address{device->BufferAddress(buf).Address};

    auto       fill{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "wait_fence_fill"))};
    auto       copy{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "wait_fence_copy"))};
    auto       overwrite{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "wait_fence_overwrite"))};
    const auto record{[&](rhi::ICommandList &commands, const uint32_t value)
                      {
                          commands.Begin();
                          commands.SetPipeline(pso);
                          commands.SetPushConstants(PC{address, value, kCount});
                          commands.Dispatch(kCount / 64);
                          commands.End();
                      }};
    uint32_t   errors{0};
    for (uint32_t round{1}; round <= kRounds; ++round)
    {
        const uint32_t before{0xA0000000u + round};
        record(*fill, before);
        device->WaitForFence(device->Submit(*fill));

        // Flushed inside the copy, the closing transition orders only this list; the next
        // submission is held back by its WaitFence alone.
        copy->Begin();
        copy->Transition(buf, rhi::ResourceState::UnorderedAccess, rhi::ResourceState::TransferSrc);
        copy->FlushBarriers();
        copy->CopyBuffer(buf, staging, {.SrcOffset = 0, .DstOffset = 0, .Size = kSize});
        copy->Transition(buf, rhi::ResourceState::TransferSrc, rhi::ResourceState::UnorderedAccess);
        copy->FlushBarriers();
        copy->End();
        const auto copied{device->Submit(*copy)};
        record(*overwrite, 0xB0000000u + round);
        device->WaitForFence(device->Submit(*overwrite, {.WaitFence = copied}));
        device->WaitForFence(copied);

        auto mapped{device->MapBuffer(staging)};
        REQUIRE(mapped.Data != nullptr);
        const auto *words{static_cast<const uint32_t *>(mapped.Data)};
        for (uint32_t i{0}; i < kCount; ++i)
        {
            if (words[i] != before)
            {
                ++errors;
            }
        }
        device->UnmapBuffer(staging);
    }
    REQUIRE(errors == 0);

    device->DestroyPipeline(pso);
    device->DestroyBuffer(buf);
    device->DestroyBuffer(staging);
    std::printf("  submission waits for its fence: %u rounds of %u words verified\n", kRounds, kCount);
}

// Dome-light pixel/CDF uploads are several MiB. Keep this above the Metal 4
// upload path's regression surface rather than validating only tiny buffers.
TEST(large_buffer_upload_readback)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint64_t   kSize{6 * 1024 * 1024 + 12};
    std::vector<uint8_t> source(kSize);
    for (uint64_t i{0}; i < kSize; ++i)
    {
        source[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
    }

    auto gpu{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferDst |
                      rhi::BufferUsage::TransferSrc,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "large_upload_gpu",
    }))};
    auto readback{rhitest::Required(device->CreateBuffer({
        .Size       = kSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "large_upload_readback",
    }))};
    REQUIRE(gpu.Valid());
    REQUIRE(readback.Valid());

    device->UploadBuffer(gpu, source);
    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "large_upload_copy"))};
    cmd->Begin();
    cmd->CopyBuffer(gpu, readback, {.SrcOffset = 0, .DstOffset = 0, .Size = kSize});
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(readback)};
    REQUIRE(mapped.Valid());
    REQUIRE(std::memcmp(mapped.Data, source.data(), kSize) == 0);
    device->UnmapBuffer(readback);
    device->DestroyBuffer(readback);
    device->DestroyBuffer(gpu);
}

// ---------------------------------------------------------------------------
// Explicit texture transition between a render pass and a transfer pass.
// Regression: a transition recorded with no encoder open needs a cross-pass
// consumer barrier in the next encoder, not an intra-pass barrier.
// ---------------------------------------------------------------------------
TEST(explicit_transition_then_readback)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};

    constexpr uint32_t W{8}, H{8};
    constexpr uint64_t kRbSize{W * H * 4};

    auto rt{rhitest::Required(device->CreateTexture({
        .Format    = rhi::Format::RGBA8Unorm,
        .Extent    = {W, H, 1},
        .Usage     = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferSrc,
        .DebugName = "explicit_rt",
    }))};
    auto rb{rhitest::Required(device->CreateBuffer({
        .Size       = kRbSize,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "explicit_rb",
    }))};
    REQUIRE(rt.Valid());
    REQUIRE(rb.Valid());

    auto pso{rhitest::Required(device->CreateGraphicsPipeline({
        .VertexShader   = rhitest::loadShaderArtifact("fullscreen", "vert_main", rhi::ShaderStage::Vertex),
        .FragmentShader = rhitest::loadShaderArtifact("fullscreen", "red_frag", rhi::ShaderStage::Fragment),
        .Topology       = rhi::PrimitiveTopology::TriangleList,
        .Rasterizer     = {.CullMode = rhi::CullMode::None},
        .DepthStencil   = rhi::DepthDisabled(),
        .ColorBlend     = {rhi::BlendDisabled()},
        .ColorFormats   = {rhi::Format::RGBA8Unorm},
        .DebugName      = "explicit_pso",
    }))};
    REQUIRE(pso.Valid());

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "explicit_cmd"))};
    cmd->Begin();
    cmd->BeginRendering({
        .Color      = {{
            .Texture    = rt,
            .LoadOp     = rhi::LoadOp::Clear,
            .StoreOp    = rhi::StoreOp::Store,
            .ClearValue = {0.f, 0.f, 0.f, 1.f},
        }},
        .RenderArea = {W, H},
    });
    cmd->SetViewport({0.f, 0.f, float(W), float(H), 0.f, 1.f});
    cmd->SetScissor({0, 0, W, H});
    cmd->SetPipeline(pso);
    cmd->Draw(3, 1, 0, 0);
    cmd->EndRendering();

    cmd->Transition(rt, rhi::ResourceState::RenderTarget, rhi::ResourceState::TransferSrc);
    cmd->CopyTextureToBuffer(rt, {.MipLevel = 0, .ArrayLayer = 0, .Extent = {W, H, 1}}, rb, 0);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    auto mapped{device->MapBuffer(rb)};
    REQUIRE(mapped.Data != nullptr);
    const auto *pixels{static_cast<const uint8_t *>(mapped.Data)};
    uint32_t    mismatches{0};
    for (uint32_t i{0}; i < W * H; ++i)
    {
        const uint8_t *p{pixels + i * 4};
        if (p[0] != 255 || p[1] != 0 || p[2] != 0 || p[3] != 255)
        {
            ++mismatches;
        }
    }
    device->UnmapBuffer(rb);
    REQUIRE(mismatches == 0);

    device->DestroyPipeline(pso);
    device->DestroyTexture(rt);
    device->DestroyBuffer(rb);
    std::printf("  explicit transition + readback: %ux%u all red\n", W, H);
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

int main()
{
    // --- Smoke tests (no GPU needed) ---
    test_handles_are_invalid_by_default();
    test_mapped_buffer_views_its_bytes();
    test_format_helpers();
    test_resource_state_bitmask();
    test_buffer_usage_bitmask();
    test_gpu_boolean_flags();
    test_sampler_presets_are_valid();
    test_blend_presets();
    test_gpu_address_offset();
    test_shader_library_refuses_bad_sizes();
    test_texture_convenience_constructors();

    std::printf("\n── Device tests ────────────────────────────────\n");
    test_shared_device_reuses_live_device();
    test_device_create_and_buffer();
    test_device_texture_and_sampler();
    test_bindless_heap_tracks_resource_lifetimes();
    test_command_list_begin_end();
    test_timestamp_queries_measure_gpu_work();

    std::printf("\n── Integration tests ───────────────────────────\n");
    test_buffer_upload_readback();
    test_command_list_records_again_after_completion();
    test_large_buffer_upload_readback();
    test_slot_pool_stress();
    test_compute_fill_via_bda();
    test_push_constants_belong_to_their_own_dispatch();
    test_command_resource_reuse_stress();
    test_compute_bindless_fill();
    test_compute_bindless_texture_sampling();
    test_render_triangle_readback();

    std::printf("\n── Advanced tests ──────────────────────────────\n");
    test_draw_indexed();
    test_depth_test_occlusion();
    test_multiple_render_targets();
    test_texture_buffer_roundtrip();
    test_compute_barrier();
    test_draw_indirect();
    test_explicit_barrier_before_copy();
    test_submission_waits_for_its_fence();
    test_explicit_transition_then_readback();
    std::printf("\n");

    std::printf("All tests passed.\n");
    return 0;
}
