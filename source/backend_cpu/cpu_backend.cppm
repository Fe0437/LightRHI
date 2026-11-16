/**
 * {file}
 * {brief} Opens the headless CPU backend and locates its compiled compute libraries.
 * Window surfaces and raster pipelines belong to graphics backends.
 */
module;
#include <expected>
#include <memory>
#include <string_view>

export module lightRHI.cpu;
export import rhi;

export namespace rhi::cpu
{
    /** {brief} Creates a headless device that executes compute commands on the calling CPU. */
    [[nodiscard]] std::expected<std::unique_ptr<IDevice>, DeviceError> CreateDevice(const DeviceDesc &desc = {});

    /** {brief} Locates a trusted native compute library built by light_rhi_compile_shaders with BACKEND cpu. */
    [[nodiscard]] std::expected<ShaderLibrary, ShaderArtifactError> ReadShaderLibrary(std::string_view directory,
                                                                                      std::string_view library);
} // namespace rhi::cpu
