#pragma once
// device_result.h — unwraps a device result in a test.
//
// The RHI reports a refusal by returning it, so a test that cannot continue without the resource
// says so here rather than repeating the same three lines at every creation.

#include <cstdio>
#include <cstdlib>
#include <expected>
#include <utility>

namespace rhitest
{
    /** {brief} Returns the value, or ends the run naming the reason the device gave. */
    template <typename Value, typename Error> Value Required(std::expected<Value, Error> result)
    {
        if (!result)
        {
            std::fprintf(stderr, "FAIL: the device refused the request (reason %d)\n",
                         static_cast<int>(result.error()));
            std::exit(1);
        }
        return std::move(*result);
    }
} // namespace rhitest
