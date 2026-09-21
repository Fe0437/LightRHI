#pragma once
// example_required.h — what an example does when the RHI refuses a request.
//
// The RHI reports a refusal by returning it, and an example shows that being handled rather than
// ignored. There is nothing to recover to in a program whose whole purpose is the request, so this
// says which one was turned down and stops. Nothing here is about devices in particular: every
// request that can be refused goes through it.

#include <cstdio>
#include <cstdlib>
#include <expected>
#include <utility>

namespace rhiexample
{
    /** {brief} Returns the value, or reports which request was refused and stops. */
    template <typename Value, typename Error> Value required(std::expected<Value, Error> result)
    {
        if (!result)
        {
            std::fprintf(stderr, "[example] the device refused the request (reason %d)\n",
                         static_cast<int>(result.error()));
            std::exit(EXIT_FAILURE);
        }
        return std::move(*result);
    }
} // namespace rhiexample
