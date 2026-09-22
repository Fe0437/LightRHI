/**
 * {file}
 * {brief} Where Metal shader libraries are, and what they hold.
 *
 * The build ships either a `.metallib` compiled ahead of time for the target SDK, or MSL source
 * the device compiles when a pipeline is created; LIGHT_RHI_METAL_PRECOMPILED_SHADERS says which.
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
#if LIGHT_RHI_METAL_PRECOMPILED_SHADERS
        constexpr std::string_view kExtension{".metallib"};
        constexpr ShaderFormat     kFormat{ShaderFormat::MetalLib};
#else
        constexpr std::string_view kExtension{".metal"};
        constexpr ShaderFormat     kFormat{ShaderFormat::MslSource};
#endif
        std::string path{directory};
        path.append("/").append(library).append(kExtension);
        return ShaderLibrary::Read(path, kFormat);
    }
} // namespace rhi
