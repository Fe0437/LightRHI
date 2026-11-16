/**
 * {file}
 * {brief} Checks that recording compute work on a thread with no autorelease pool leaves nothing pending.
 *
 * The Metal backend takes each encoder Metal autoreleases straight into its own ownership, without a pool, so
 * recording costs nothing extra. If a build ever stopped claiming it, every dispatch on such a thread would leave
 * an encoder waiting for a pool that never drains, and the process would grow. This test records many dispatches
 * on a plain thread and fails if the process's memory grows with them. Metal only.
 */
// Standard headers must precede module imports to avoid include-guard isolation issues with libc++ modules.
#include <cstdint>
#include <cstdio>
#include <mach/mach.h>
#include <thread>

import lightRHI;

#include "../device_result.h"
#include "../shader_artifact_loader.h"

namespace
{
    /** {brief} The memory the process holds now, as the system counts it against its footprint. */
    [[nodiscard]] std::uint64_t Footprint()
    {
        task_vm_info_data_t    info{};
        mach_msg_type_number_t count{TASK_VM_INFO_COUNT};
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the Mach API takes untyped task info
        const auto result{task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count)};
        return result == KERN_SUCCESS ? info.phys_footprint : 0U;
    }

    /** {brief} The root constants of the fill kernel. */
    struct alignas(8) FillConstants
    {
        std::uint64_t Output{};
        std::uint32_t Value{};
        std::uint32_t Count{};
    };
} // namespace

int main()
{
    constexpr std::uint32_t kWarmUp{2000U};
    constexpr std::uint32_t kMeasured{20000U};
    // A leaked encoder holds about 170 bytes, so 20000 of them grow the process by over 3 MB; claimed ones by
    // under 64 KB.
    constexpr std::uint64_t kAllowedGrowth{1024U * 1024U};

    auto                device{rhitest::Required(rhi::CreateDevice({}))};
    auto                buffer{rhitest::Required(device->CreateBuffer({
        .Size       = 64U * sizeof(std::uint32_t),
        .Usage      = rhi::BufferUsage::Storage | rhi::BufferUsage::DeviceAddress,
        .MemoryType = rhi::MemoryType::CpuToGpu,
        .DebugName  = "encoder_ownership",
    }))};
    auto                pipeline{rhitest::Required(device->CreateComputePipeline({
        .Shader    = rhitest::loadShaderArtifact("compute_fill_bda", "fill_via_bda", rhi::ShaderStage::Compute),
        .DebugName = "encoder_ownership",
    }))};
    const FillConstants constants{.Output = device->BufferAddress(buffer).Address, .Value = 7U, .Count = 64U};

    std::uint64_t grown{};
    // A std::thread has no autorelease pool, as a painting thread has none.
    std::thread recording{
        [&]
        {
            auto commands{rhitest::Required(device->CreateCommandList(rhi::QueueType::Compute, "encoder_ownership"))};
            const auto record{
                [&](const std::uint32_t times)
                {
                    for (std::uint32_t at{}; at < times; ++at)
                    {
                        commands->Begin();
                        commands->SetPipeline(pipeline);
                        commands->SetPushConstants(constants);
                        commands->Dispatch(1);
                        commands->End();
                        device->WaitForFence(device->Submit(*commands));
                    }
                },
            };
            record(kWarmUp);
            const auto before{Footprint()};
            record(kMeasured);
            const auto after{Footprint()};
            grown = after > before ? after - before : 0U;
        }};
    recording.join();

    device->DestroyPipeline(pipeline);
    device->DestroyBuffer(buffer);
    std::printf("encoder ownership: %u dispatches on a thread with no pool grew the process by %llu bytes\n", kMeasured,
                static_cast<unsigned long long>(grown));
    if (grown > kAllowedGrowth)
    {
        std::fprintf(stderr, "FAIL: recording leaves autoreleased objects pending on a thread with no pool\n");
        return 1;
    }
    return 0;
}
