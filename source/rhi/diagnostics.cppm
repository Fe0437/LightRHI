/**
 * {file}
 * {brief} How this library reports a failure it cannot carry out.
 *
 * LightRHI never unwinds. A consumer may be compiled without exceptions - Flexible Drawing is, and
 * a C++ module cannot be imported across that difference at all - so failure travels the same way
 * results do: through the return value.
 *
 * Three kinds of failure exist. A device refusal is recoverable and travels in the return value. A
 * caller contract violation stops in every build because continuing would use invalid data. A
 * private implementation precondition that its caller already checked uses `assert`, so a debug
 * build stops at the internal boundary while an optimized build pays no cost for the repeated
 * check.
 */
module;

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

export module rhi:diagnostics;

export namespace rhi
{
    /**
     * {brief} Why a device could not carry out a request.
     *
     * Every call that can fail reports one of these beside its result, so a caller reads the reason
     * from the value it already has rather than from a log or an exception it cannot catch.
     */
    enum class DeviceError : std::uint8_t
    {
        NoDevice,        ///< No usable GPU or driver on this system.
        Refused,         ///< The device declined the request; memory or a driver limit is the usual cause.
        Exhausted,       ///< The device is already holding as many resources of that kind as it can.
        Unsupported,     ///< The device does not have the feature the request needs.
        InvalidArgument, ///< The request cannot be satisfied as described.
    };

    /**
     * {brief} Reports a request the device refused, which the caller is expected to handle.
     *
     * The caller learns the outcome from the invalid handle or empty result it receives; this is
     * the explanation beside it.
     */
    inline void ReportRefusal(std::string_view what) noexcept
    {
        std::fprintf(stderr, "[LightRHI] %.*s\n", static_cast<int>(what.size()), what.data());
    }

    /**
     * {brief} Stops the process on a broken contract, naming what was broken.
     *
     * For mistakes a return value cannot repair. Everything a caller can check beforehand - a
     * capability, a format, a size - is reported through ReportRefusal() instead.
     */
    [[noreturn]] inline void FailContract(std::string_view what) noexcept
    {
        std::fprintf(stderr, "[LightRHI] broken contract: %.*s\n", static_cast<int>(what.size()), what.data());
        std::abort();
    }
} // namespace rhi
