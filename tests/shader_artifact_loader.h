#pragma once
// shader_artifact_loader.h — test-only loader for compiled shader artifacts.
//
// Reads the .spv / .metal files produced by tools/compile_shaders.py (driven
// by tests/shaders/shaders_registry.generated.json) from RHI_TEST_SHADER_DIR
// and hands them to the RHI as an rhi::ShaderArtifactView. This file is test
// scaffolding, not part of the RHI public API — the RHI core knows nothing
// about how these bytes reached disk.

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef RHI_TEST_SHADER_DIR
#error "RHI_TEST_SHADER_DIR must be defined by the build (see tests/CMakeLists.txt)"
#endif

namespace rhitest
{

    inline std::vector<std::byte> readShaderArtifactBytes(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("[rhi_tests] failed to open shader artifact: " + path);
        }

        const std::streamsize size{file.tellg()};
        file.seekg(0);

        std::vector<std::byte> bytes(static_cast<size_t>(size));
        if (size > 0 && !file.read(reinterpret_cast<char *>(bytes.data()), size))
        {
            throw std::runtime_error("[rhi_tests] failed to read shader artifact: " + path);
        }
        return bytes;
    }

    // Loads the artifact `source` compiled for `entryPoint` (e.g. "fullscreen",
    // "vert_main") for the backend the test binary was built for, and returns it
    // as an rhi::ShaderDesc ready to plug into GraphicsPipelineDesc /
    // ComputePipelineDesc. The artifact name is composed the same way
    // tools/generate_shader_registry.py names it: <source>_<entryPoint>.
    //
    // Bytes are cached per artifact and kept alive for the lifetime of the
    // process (test binaries are short-lived), so the ShaderDesc's non-owning
    // spans stay valid and repeated loads of the same shader (common across
    // tests) don't re-read the file from disk.
    inline rhi::ShaderDesc loadShaderArtifact(std::string_view source, std::string_view entryPoint,
                                              rhi::ShaderStage stage)
    {
        static std::unordered_map<std::string, std::vector<std::byte>> cache;

        const std::string name{std::string{source} + "_" + std::string{entryPoint}};

#if defined(RHI_BACKEND_METAL)
        const std::string       path{std::string(RHI_TEST_SHADER_DIR) + "/metal/" + name + ".generated.metal"};
        const rhi::ShaderFormat format{rhi::ShaderFormat::MslSource};
        const std::string_view  backendEntryPoint{entryPoint};
#elif defined(RHI_BACKEND_VULKAN)
        const std::string       path{std::string(RHI_TEST_SHADER_DIR) + "/vulkan/" + name + ".generated.spv"};
        const rhi::ShaderFormat format{rhi::ShaderFormat::Spirv};
        const std::string_view  backendEntryPoint{"main"};
#else
#error "loadShaderArtifact: no RHI_BACKEND_METAL / RHI_BACKEND_VULKAN defined"
#endif

        auto [it, inserted] = cache.try_emplace(name);
        if (inserted)
        {
            it->second = readShaderArtifactBytes(path);
        }
        const auto &bytes = it->second;

        rhi::ShaderArtifactView artifact{
            .Format     = format,
            .Stage      = stage,
            .EntryPoint = backendEntryPoint,
            .Data       = bytes,
        };
        auto desc = rhi::ToShaderDesc(artifact);
        if (!desc)
        {
            throw std::runtime_error("[rhi_tests] ToShaderDesc failed for shader artifact: " + path);
        }
        return *desc;
    }

} // namespace rhitest
