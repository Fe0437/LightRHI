/**
 * {file}
 * {brief} Test-only loader for compiled shader libraries.
 *
 * Reads the libraries light_rhi_compile_shaders() wrote to RHI_TEST_SHADER_DIR with the backend's
 * rhi::ReadShaderLibrary, and caches them for the process. Test scaffolding, not part of the RHI
 * public API: the RHI core knows nothing about how these bytes reached disk.
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

#ifndef RHI_TEST_SHADER_DIR
#error "RHI_TEST_SHADER_DIR must be defined by the build (see tests/CMakeLists.txt)"
#endif

namespace rhitest
{

    /** {brief} Stops with a message: a missing or unreadable artifact means the build is wrong. */
    [[noreturn]] inline void failArtifact(const std::string &message)
    {
        std::fprintf(stderr, "[rhi_tests] %s\n", message.c_str());
        std::exit(1);
    }

    // Describes `entryPoint` in the library compiled from `source`.slang, for the backend
    // this binary was built for, ready for GraphicsPipelineDesc / ComputePipelineDesc.
    // Each library is read once and kept for the process, so every ShaderDesc's
    // non-owning view stays valid.
    inline rhi::ShaderDesc loadShaderArtifact(std::string_view source, std::string_view entryPoint,
                                              rhi::ShaderStage stage)
    {
        static std::unordered_map<std::string, rhi::ShaderLibrary> cache{};

        std::string name{source};
        auto        found{cache.find(name)};
        if (found == cache.end())
        {
            auto read{rhi::ReadShaderLibrary(RHI_TEST_SHADER_DIR, source)};
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

} // namespace rhitest
