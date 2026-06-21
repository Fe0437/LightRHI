#pragma once

// Cross-language helpers for 32-bit flag words stored in GPU-visible data.
// Keep bool for local control flow; use these helpers for buffer and
// push-constant ABI fields, where bool size and representation are target-defined.

#ifdef __cplusplus

#include <cstdint>

namespace rhi
{
    using GpuFlags = std::uint32_t;

    [[nodiscard]] constexpr bool HasFlag(GpuFlags flags, GpuFlags flag) noexcept
    {
        return flag != 0U && (flags & flag) == flag;
    }

    constexpr void SetFlag(GpuFlags &flags, GpuFlags flag, bool enabled) noexcept
    {
        flags = enabled ? (flags | flag) : (flags & ~flag);
    }

    [[nodiscard]] constexpr GpuFlags FlagIf(GpuFlags flag, bool enabled) noexcept
    {
        return enabled ? flag : 0U;
    }
} // namespace rhi

#else

bool HasFlag(uint flags, uint flag)
{
    return flag != 0U && (flags & flag) == flag;
}

void SetFlag(inout uint flags, uint flag, bool enabled)
{
    flags = enabled ? (flags | flag) : (flags & ~flag);
}

uint FlagIf(uint flag, bool enabled)
{
    return enabled ? flag : 0U;
}

#endif
