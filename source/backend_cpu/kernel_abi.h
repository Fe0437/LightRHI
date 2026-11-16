/**
 * {file}
 * {brief} Private interface between the CPU device and a compiled Slang library.
 * It contains only dispatch coordinates, native parameter storage, and workgroup services.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace light_rhi_cpu
{
    struct Grid
    {
        std::uint32_t X, Y, Z;
    };

    inline constexpr std::size_t ScratchBytes{32U * 1024U};

    struct Workgroup
    {
        void *Barrier;
        void (*Arrive)(void *);
        void *Scratch;
    };

    using Dispatch   = void (*)(Grid groups, void *globals);
    using InvokeLane = void (*)(Grid group, Grid lane, void *globals, const Workgroup *workgroup);

    struct Kernel
    {
        Dispatch      Run;
        InvokeLane    RunLane;
        Grid          Threads;
        std::uint32_t PushConstantBytes;
    };

    using FindKernel = const Kernel *(*)(const char *entry);

    // Only a generated library sets this. The functions below are its Slang CPU intrinsics.
    inline thread_local const Workgroup *CurrentWorkgroup{};
} // namespace light_rhi_cpu

inline void LightRhiCpuGroupSync()
{
    const auto &group{*light_rhi_cpu::CurrentWorkgroup};
    group.Arrive(group.Barrier);
}

inline std::uint64_t LightRhiCpuGroupAddress(std::uint32_t offset, std::uint32_t bytes)
{
    if (offset > light_rhi_cpu::ScratchBytes || bytes > light_rhi_cpu::ScratchBytes - offset)
    {
        std::abort();
    }
    return reinterpret_cast<std::uintptr_t>(static_cast<std::byte *>(light_rhi_cpu::CurrentWorkgroup->Scratch) +
                                            offset);
}
