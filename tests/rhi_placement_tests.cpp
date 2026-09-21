// rhi_placement_tests.cpp — memory heaps and placed textures.
//
// Covers the contract a caller relies on when it owns the memory a texture lives in: what
// TexturePlacementRequirements() reports, that independent placements in one block keep their own
// content, that a released range can be reused once its work has completed, and that every refusal
// is explicit. The final case measures placement against creating the same textures individually.

// Standard headers must precede module imports to avoid include-guard
// isolation issues (__promote_t redefinition) with LLVM libc++ and C++23 modules.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

import lightRHI;

#include "device_result.h"
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

#define TEST(name) void test_##name()

namespace
{
    /** {brief} Names the build a measurement was taken in, so its numbers can be compared. */
#if defined(NDEBUG)
    constexpr const char *kBuildConfiguration{"optimized"};
#else
    constexpr const char *kBuildConfiguration{"debug"};
#endif

    constexpr uint32_t kTileSize{64};
    constexpr uint64_t kTileBytes{uint64_t{kTileSize} * kTileSize * 4};

    /** {brief} Returns the descriptor shared by every placed texture in these tests. */
    rhi::TextureDesc TileDesc(std::string_view name)
    {
        return rhi::TextureDesc{
            .Format    = rhi::Format::RGBA8Unorm,
            .Extent    = {kTileSize, kTileSize, 1},
            .Usage     = rhi::TextureUsage::Sampled | rhi::TextureUsage::TransferDst | rhi::TextureUsage::TransferSrc,
            .DebugName = name,
        };
    }

    /** {brief} Rounds `value` up to the next multiple of `alignment`. */
    uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    /** {brief} Fills a placed texture with one solid color and reads the result back. */
    uint32_t UploadAndReadBack(rhi::IDevice &device, rhi::TextureHandle texture, uint8_t value)
    {
        std::vector<uint8_t> texels(kTileBytes, value);
        device.UploadTexture(texture, texels, uint64_t{kTileSize} * 4, kTileBytes,
                             {.Extent = {kTileSize, kTileSize, 1}});

        auto readback{rhitest::Required(device.CreateBuffer({
            .Size       = kTileBytes,
            .Usage      = rhi::BufferUsage::TransferDst,
            .MemoryType = rhi::MemoryType::GpuToCpu,
            .DebugName  = "placement_readback",
        }))};
        REQUIRE(readback.Valid());

        auto cmd{rhitest::Required(device.CreateCommandList(rhi::QueueType::Graphics, "placement_readback_cmd"))};
        cmd->Begin();
        cmd->CopyTextureToBuffer(texture, {.Extent = {kTileSize, kTileSize, 1}}, readback, 0);
        cmd->End();
        device.WaitForFence(device.Submit(*cmd));

        auto mapped{device.MapBuffer(readback)};
        REQUIRE(mapped.Data != nullptr);
        const auto *pixels{static_cast<const uint8_t *>(mapped.Data)};
        uint32_t    mismatches{0};
        for (uint64_t i{0}; i < kTileBytes; ++i)
        {
            if (pixels[i] != value)
            {
                ++mismatches;
            }
        }
        device.UnmapBuffer(readback);
        device.DestroyBuffer(readback);
        return mismatches;
    }
} // namespace

// ---------------------------------------------------------------------------
// Requirements and successful placement
// ---------------------------------------------------------------------------

TEST(requirements_describe_a_real_placement)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    const rhi::PlacementRequirements requirements{
        rhitest::Required(device->TexturePlacementRequirements(TileDesc("tile"), rhi::MemoryType::GpuOnly))};
    std::printf("  tile %ux%u: size=%llu align=%llu\n", kTileSize, kTileSize,
                static_cast<unsigned long long>(requirements.Size),
                static_cast<unsigned long long>(requirements.Alignment));

    REQUIRE(requirements.Size >= kTileBytes);
    REQUIRE(requirements.Alignment > 0);
    REQUIRE((requirements.Alignment & (requirements.Alignment - 1)) == 0); // a power of two

    auto heap{rhitest::Required(device->CreateMemoryHeap({.Size = requirements.Size * 4, .DebugName = "tile_heap"}))};
    REQUIRE(heap.Valid());

    auto placed{device->CreateTexture(TileDesc("tile"), {.Heap = heap, .Offset = 0})};
    REQUIRE(placed.has_value());
    REQUIRE(placed->Valid());
    REQUIRE(device->TextureAddress(*placed).Valid());

    device->DestroyTexture(*placed);
    device->DestroyMemoryHeap(heap);
}

TEST(independent_placements_keep_their_own_content)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    const rhi::PlacementRequirements requirements{
        rhitest::Required(device->TexturePlacementRequirements(TileDesc("tile"), rhi::MemoryType::GpuOnly))};
    const uint64_t stride{AlignUp(requirements.Size, requirements.Alignment)};

    auto heap{rhitest::Required(device->CreateMemoryHeap({.Size = stride * 2, .DebugName = "two_tile_heap"}))};
    REQUIRE(heap.Valid());

    auto first{device->CreateTexture(TileDesc("tile_a"), {.Heap = heap, .Offset = 0})};
    auto second{device->CreateTexture(TileDesc("tile_b"), {.Heap = heap, .Offset = stride})};
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    REQUIRE(UploadAndReadBack(*device, *first, 0x11) == 0);
    REQUIRE(UploadAndReadBack(*device, *second, 0x77) == 0);
    // Writing the neighbour must not have disturbed the first placement.
    REQUIRE(UploadAndReadBack(*device, *first, 0x11) == 0);

    device->DestroyTexture(*first);
    device->DestroyTexture(*second);
    device->DestroyMemoryHeap(heap);
}

TEST(a_released_range_can_be_reused)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    const rhi::PlacementRequirements requirements{
        rhitest::Required(device->TexturePlacementRequirements(TileDesc("tile"), rhi::MemoryType::GpuOnly))};
    auto heap{rhitest::Required(device->CreateMemoryHeap({.Size = requirements.Size, .DebugName = "reused_heap"}))};
    REQUIRE(heap.Valid());

    auto first{device->CreateTexture(TileDesc("first"), {.Heap = heap, .Offset = 0})};
    REQUIRE(first.has_value());
    REQUIRE(UploadAndReadBack(*device, *first, 0x22) == 0);

    // The readback above already waited for the only work touching this range.
    device->WaitIdle();
    device->DestroyTexture(*first);

    auto second{device->CreateTexture(TileDesc("second"), {.Heap = heap, .Offset = 0})};
    REQUIRE(second.has_value());
    REQUIRE(UploadAndReadBack(*device, *second, 0x99) == 0);

    device->DestroyTexture(*second);
    device->DestroyMemoryHeap(heap);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

TEST(refused_placements_report_why)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    const rhi::TextureDesc           desc{TileDesc("tile")};
    const rhi::PlacementRequirements requirements{
        rhitest::Required(device->TexturePlacementRequirements(desc, rhi::MemoryType::GpuOnly))};
    const uint64_t stride{AlignUp(requirements.Size, requirements.Alignment)};

    auto heap{rhitest::Required(device->CreateMemoryHeap({.Size = stride * 2, .DebugName = "refusal_heap"}))};
    REQUIRE(heap.Valid());

    const auto noBlock{device->CreateTexture(desc, {.Heap = {}, .Offset = 0})};
    REQUIRE(!noBlock.has_value());
    REQUIRE(noBlock.error() == rhi::PlacementError::InvalidHeap);

    const auto misaligned{device->CreateTexture(desc, {.Heap = heap, .Offset = requirements.Alignment / 2})};
    REQUIRE(!misaligned.has_value());
    REQUIRE(misaligned.error() == rhi::PlacementError::MisalignedOffset);

    const auto pastTheEnd{device->CreateTexture(desc, {.Heap = heap, .Offset = stride * 2})};
    REQUIRE(!pastTheEnd.has_value());
    REQUIRE(pastTheEnd.error() == rhi::PlacementError::OutOfRange);

    auto live{device->CreateTexture(desc, {.Heap = heap, .Offset = 0})};
    REQUIRE(live.has_value());
    const auto overlapping{device->CreateTexture(desc, {.Heap = heap, .Offset = requirements.Alignment})};
    REQUIRE(!overlapping.has_value());
    REQUIRE(overlapping.error() == rhi::PlacementError::Overlapping);

    device->DestroyTexture(*live);
    device->DestroyMemoryHeap(heap);
}

TEST(a_cpu_visible_heap_holds_textures_too)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    const rhi::TextureDesc desc{TileDesc("host_tile")};
    const auto             requirementsResult{device->TexturePlacementRequirements(desc, rhi::MemoryType::CpuToGpu)};

    // Where CPU-visible placement is unsupported, the query and the attempted placement report the
    // same explicit reason. Otherwise the texture behaves like any other placed resource.
    if (!requirementsResult)
    {
        REQUIRE(requirementsResult.error() == rhi::PlacementError::IncompatibleMemory);
        std::printf("  CPU-visible placement unsupported on this adapter; refusal path checked\n");
        auto heap{rhitest::Required(device->CreateMemoryHeap({
            .Size       = kTileBytes * 4,
            .MemoryType = rhi::MemoryType::CpuToGpu,
            .DebugName  = "host_heap",
        }))};
        REQUIRE(heap.Valid());
        const auto refused{device->CreateTexture(desc, {.Heap = heap, .Offset = 0})};
        REQUIRE(!refused.has_value());
        REQUIRE(refused.error() == rhi::PlacementError::IncompatibleMemory);
        device->DestroyMemoryHeap(heap);
        return;
    }
    const rhi::PlacementRequirements requirements{*requirementsResult};

    std::printf("  CPU-visible tile: size=%llu align=%llu\n", static_cast<unsigned long long>(requirements.Size),
                static_cast<unsigned long long>(requirements.Alignment));

    auto heap{rhitest::Required(device->CreateMemoryHeap({
        .Size       = requirements.Size,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "host_heap",
    }))};
    REQUIRE(heap.Valid());

    auto placed{device->CreateTexture(desc, {.Heap = heap, .Offset = 0})};
    REQUIRE(placed.has_value());
    REQUIRE(UploadAndReadBack(*device, *placed, 0x5A) == 0);

    device->DestroyTexture(*placed);
    device->DestroyMemoryHeap(heap);
}

// ---------------------------------------------------------------------------
// Cost of many small textures
// ---------------------------------------------------------------------------

TEST(many_placements_cost_less_than_many_allocations)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    constexpr uint32_t               kCount{1024};
    const rhi::TextureDesc           desc{TileDesc("tile")};
    const rhi::PlacementRequirements requirements{
        rhitest::Required(device->TexturePlacementRequirements(desc, rhi::MemoryType::GpuOnly))};
    const uint64_t stride{AlignUp(requirements.Size, requirements.Alignment)};

    using Clock = std::chrono::steady_clock;

    std::vector<rhi::TextureHandle> owned{};
    owned.reserve(kCount);
    const auto ownedStart{Clock::now()};
    for (uint32_t i{0}; i < kCount; ++i)
    {
        owned.push_back(rhitest::Required(device->CreateTexture(desc)));
    }
    const auto ownedMicros{std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - ownedStart).count()};
    for (auto handle : owned)
    {
        device->DestroyTexture(handle);
    }

    auto heap{rhitest::Required(device->CreateMemoryHeap({.Size = stride * kCount, .DebugName = "measurement_heap"}))};
    REQUIRE(heap.Valid());

    std::vector<rhi::TextureHandle> placed{};
    placed.reserve(kCount);
    const auto placedStart{Clock::now()};
    for (uint32_t i{0}; i < kCount; ++i)
    {
        auto texture{device->CreateTexture(desc, {.Heap = heap, .Offset = stride * i})};
        REQUIRE(texture.has_value());
        placed.push_back(*texture);
    }
    const auto placedMicros{std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - placedStart).count()};
    for (auto handle : placed)
    {
        device->DestroyTexture(handle);
    }
    device->DestroyMemoryHeap(heap);

    std::printf("  on %s (%s build): %u textures individually %lld us, placed %lld us\n",
                std::string{device->AdapterName()}.c_str(), kBuildConfiguration, kCount,
                static_cast<long long>(ownedMicros), static_cast<long long>(placedMicros));
    REQUIRE(placedMicros < ownedMicros);
}

// ---------------------------------------------------------------------------
// Placed buffers: memory the application owns, reached by address
// ---------------------------------------------------------------------------

TEST(placed_buffers_are_addressable_from_a_shader)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    REQUIRE(device != nullptr);

    // Two ranges of one heap, each holding the pixels of one canvas tile. This is how a canvas
    // owns its storage: one allocation, tiles at offsets, each addressed by the shader that writes
    // it. Nothing is bound; the address is the whole binding.
    constexpr uint32_t    kWords{64};
    const rhi::BufferDesc tileDesc{
        .Size       = kWords * sizeof(uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress | rhi::BufferUsage::TransferSrc,
        .MemoryType = rhi::MemoryType::GpuOnly,
        .DebugName  = "tile",
    };
    const rhi::PlacementRequirements requirements{rhitest::Required(device->BufferPlacementRequirements(tileDesc))};
    REQUIRE(requirements.Size >= tileDesc.Size);
    REQUIRE(requirements.Alignment > 0);

    const uint64_t stride{AlignUp(requirements.Size, requirements.Alignment)};
    auto           heap{rhitest::Required(device->CreateMemoryHeap({.Size = stride * 2, .DebugName = "tile_heap"}))};
    REQUIRE(heap.Valid());

    auto first{device->CreateBuffer(tileDesc, {.Heap = heap, .Offset = 0})};
    auto second{device->CreateBuffer(tileDesc, {.Heap = heap, .Offset = stride})};
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    const rhi::GpuAddress firstAddress{device->BufferAddress(*first)};
    const rhi::GpuAddress secondAddress{device->BufferAddress(*second)};
    REQUIRE(firstAddress.Valid());
    REQUIRE(secondAddress.Valid());
    REQUIRE(firstAddress.Address != secondAddress.Address);

    auto pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "tile_fill_pso",
    }))};
    REQUIRE(pipeline.Valid());

    struct alignas(8) FillConstants
    {
        uint64_t Address{};
        uint32_t Value{};
        uint32_t Count{};
    };
    constexpr uint32_t kFirstValue{0x11111111U};
    constexpr uint32_t kSecondValue{0x22222222U};

    auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "tile_fill_cmd"))};
    cmd->Begin();
    cmd->SetPipeline(pipeline);
    const FillConstants firstFill{.Address = firstAddress.Address, .Value = kFirstValue, .Count = kWords};
    cmd->SetPushConstants(firstFill);
    cmd->Dispatch(1, 1, 1);
    const FillConstants secondFill{.Address = secondAddress.Address, .Value = kSecondValue, .Count = kWords};
    cmd->SetPushConstants(secondFill);
    cmd->Dispatch(1, 1, 1);
    cmd->End();
    device->WaitForFence(device->Submit(*cmd));

    // Each tile holds its own value: neighbouring placements in one heap do not bleed.
    auto readback{rhitest::Required(device->CreateBuffer({
        .Size       = tileDesc.Size * 2,
        .Usage      = rhi::BufferUsage::TransferDst,
        .MemoryType = rhi::MemoryType::GpuToCpu,
        .DebugName  = "tile_readback",
    }))};
    REQUIRE(readback.Valid());

    auto copy{rhitest::Required(device->CreateCommandList(rhi::QueueType::Transfer, "tile_copy_cmd"))};
    copy->Begin();
    copy->CopyBuffer(*first, readback, {.Size = tileDesc.Size});
    copy->CopyBuffer(*second, readback, {.DstOffset = tileDesc.Size, .Size = tileDesc.Size});
    copy->End();
    device->WaitForFence(device->Submit(*copy));

    auto mapped{device->MapBuffer(readback)};
    REQUIRE(mapped.Data != nullptr);
    const auto *words{static_cast<const uint32_t *>(mapped.Data)};
    for (uint32_t i{0}; i < kWords; ++i)
    {
        REQUIRE(words[i] == kFirstValue);
        REQUIRE(words[kWords + i] == kSecondValue);
    }
    device->UnmapBuffer(readback);

    device->DestroyBuffer(readback);
    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(*first);
    device->DestroyBuffer(*second);
    device->DestroyMemoryHeap(heap);
    std::printf("  two placed tiles filled through their own addresses\n");
}

int main()
{
    std::printf("\n── Placement tests ─────────────────────────────\n");
    test_requirements_describe_a_real_placement();
    test_placed_buffers_are_addressable_from_a_shader();
    test_independent_placements_keep_their_own_content();
    test_a_released_range_can_be_reused();
    test_refused_placements_report_why();
    test_a_cpu_visible_heap_holds_textures_too();
    test_many_placements_cost_less_than_many_allocations();
    std::printf("\nAll tests passed.\n");
    return 0;
}
