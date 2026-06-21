#pragma once
// example_shader_loader.h — loads Slang-compiled shader artifacts for examples.
//
// The examples author shaders once in Slang (examples/<name>/shaders/*.slang)
// and the build compiles them to the active backend's format via
// tools/compile_shaders.py (Slang -> SPIR-V for Vulkan, Slang -> MSL for
// Metal). This helper reads the resulting .spv / .metal file from
// RHI_EXAMPLE_SHADER_DIR and hands it to the RHI through the public
// rhi::ToShaderDesc() path — the same artifact flow the tests use.

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef RHI_EXAMPLE_SHADER_DIR
#error "RHI_EXAMPLE_SHADER_DIR must be defined by the build (see examples/CMakeLists.txt)"
#endif

namespace rhiexample
{

    inline std::vector<std::byte> readShaderArtifactBytes(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("[example] failed to open shader artifact: " + path);
        }

        const std::streamsize size{file.tellg()};
        file.seekg(0);

        std::vector<std::byte> bytes(static_cast<size_t>(size));
        if (size > 0 && !file.read(reinterpret_cast<char *>(bytes.data()), size))
        {
            throw std::runtime_error("[example] failed to read shader artifact: " + path);
        }
        return bytes;
    }

    // Loads the artifact for `name` (as declared in the example's
    // shaders_registry.generated.json) for the backend this binary was built
    // for. Bytes are cached for the process lifetime so the ShaderDesc's
    // non-owning spans stay valid.
    inline rhi::ShaderDesc loadShaderArtifact(const std::string &name, [[maybe_unused]] std::string_view entryPoint,
                                              rhi::ShaderStage stage)
    {
        static std::unordered_map<std::string, std::vector<std::byte>> cache;

#if defined(RHI_BACKEND_METAL)
        const std::string       path{std::string(RHI_EXAMPLE_SHADER_DIR) + "/metal/" + name + ".generated.metal"};
        const rhi::ShaderFormat format{rhi::ShaderFormat::MslSource};
        const std::string_view  backendEntryPoint{entryPoint};
#elif defined(RHI_BACKEND_VULKAN)
        const std::string       path{std::string(RHI_EXAMPLE_SHADER_DIR) + "/vulkan/" + name + ".generated.spv"};
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
            throw std::runtime_error("[example] ToShaderDesc failed for shader artifact: " + path);
        }
        return *desc;
    }

} // namespace rhiexample
