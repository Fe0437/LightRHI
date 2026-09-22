/**
 * {file}
 * {brief} Loads the examples' compiled shader libraries.
 *
 * The examples author shaders once in Slang, as .slang files in examples/<name>/shaders/, and
 * light_rhi_compile_shaders() compiles them for the active backend. This helper reads them from
 * RHI_EXAMPLE_SHADER_DIR with the backend's rhi::ReadShaderLibrary.
 */
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef RHI_EXAMPLE_SHADER_DIR
#error "RHI_EXAMPLE_SHADER_DIR must be defined by the build (see examples/CMakeLists.txt)"
#endif

namespace rhiexample
{

    /** {brief} Stops with a message: a missing or unreadable artifact means the build is wrong. */
    [[noreturn]] inline void failArtifact(const std::string &message)
    {
        std::fprintf(stderr, "[example] %s\n", message.c_str());
        std::exit(1);
    }

    // Describes `entryPoint` in the library compiled from `library`.slang. Each library
    // is read once and kept for the process, so the ShaderDesc's view stays valid.
    inline rhi::ShaderDesc loadShaderArtifact(std::string_view library, std::string_view entryPoint,
                                              rhi::ShaderStage stage)
    {
        static std::unordered_map<std::string, rhi::ShaderLibrary> cache{};

        std::string name{library};
        auto        found{cache.find(name)};
        if (found == cache.end())
        {
            auto read{rhi::ReadShaderLibrary(RHI_EXAMPLE_SHADER_DIR, library)};
            if (!read)
            {
                failArtifact("shader library not found or unreadable: " + name);
            }
            found = cache.emplace(std::move(name), std::move(*read)).first;
        }

        auto desc{found->second.Describe(entryPoint, stage)};
        if (!desc)
        {
            failArtifact("cannot describe " + std::string{entryPoint} + " in shader library " + found->first);
        }
        return *desc;
    }

} // namespace rhiexample
