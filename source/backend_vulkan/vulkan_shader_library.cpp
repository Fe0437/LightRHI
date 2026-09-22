/**
 * {file}
 * {brief} Where Vulkan shader libraries are, and what they hold: one SPIR-V module per source.
 */
module;
#include <expected>
#include <string>
#include <string_view>

module lightRHI;
import rhi;

namespace rhi
{
    std::expected<ShaderLibrary, ShaderArtifactError> ReadShaderLibrary(const std::string_view directory,
                                                                         const std::string_view library)
    {
        std::string path{directory};
        path.append("/").append(library).append(".spv");
        return ShaderLibrary::Read(path, ShaderFormat::Spirv);
    }
} // namespace rhi
