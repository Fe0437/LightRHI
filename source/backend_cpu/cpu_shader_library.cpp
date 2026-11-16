/**
 * {file}
 * {brief} Where CPU shader libraries are, and what they hold: a `.lrcpu` native compute library that
 * light_rhi_compile_shaders(BACKEND cpu) built for this host.
 */
module;
#include <expected>
#include <string>
#include <string_view>

module lightRHI.cpu;
import rhi;

namespace rhi::cpu
{
    std::expected<ShaderLibrary, ShaderArtifactError> ReadShaderLibrary(const std::string_view directory,
                                                                        const std::string_view library)
    {
        return ShaderLibrary::Read(std::string{directory}.append("/").append(library).append(".lrcpu"),
                                   ShaderFormat::HostCallable);
    }
} // namespace rhi::cpu
