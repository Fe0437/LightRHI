/**
 * {file}
 * {brief} Imports the complete LightRHI API and selects its Vulkan device factory.
 *
 * ```cpp
 * import lightRHI;
 *
 * auto device{rhi::CreateDevice({
 *     .EnableValidation = true,
 *     .AppName = "MyApp",
 * })};
 * ```
 * {note} The Vulkan backend requires Vulkan 1.3, synchronization2, dynamic
 * rendering, timeline semaphores, buffer device addresses, descriptor
 * indexing, and VK_EXT_descriptor_buffer. Supply SPIR-V through ShaderDesc
 * when creating pipelines.
 */

module;
#include <string_view>
#include "vulkan_platform.h"

#include <concepts>
#include <expected>
#include <memory>
#include <type_traits>

export module lightRHI;
export import rhi; // re-exports all rhi types and interfaces to consumers

export namespace rhi
{

    /**
     * {brief} Reads the shader library compiled from `library`.slang out of `directory`.
     * {param directory} Where the application's shader libraries were shipped.
     * {param library} The shader source's name, without extension.
     * {returns} The library, or NotFound / Unreadable.
     * {note} A `.spv` module holding every entry point.
     */
    [[nodiscard]] std::expected<ShaderLibrary, ShaderArtifactError> ReadShaderLibrary(std::string_view directory,
                                                                                     std::string_view library);

    /**
     * {brief} Creates an independently owned Vulkan device.
     * {param desc} Validation and application metadata used during creation.
     * {returns} Exclusive ownership of a device; destroy its resources before releasing it.
     */
    [[nodiscard]] std::expected<std::unique_ptr<IDevice>, DeviceError> CreateDevice(const DeviceDesc &desc = {});

    /**
     * {brief} Makes the caller's surface somewhere `device` can draw.
     * {param device} The device that will draw into it; it outlives the target.
     * {param textureSource} Asked for a surface while the target is made; the target takes it and
     * destroys it, so destroy the target before the window the surface came from.
     * {param timing} When a presented frame may reach the screen.
     * {param requestedFormat} Undefined takes the surface's own preferred format.
     * {returns} Exclusive ownership of the target, or InvalidArgument when no surface came back.
     * {note} Which window system that surface belongs to is not asked and does not matter: a
     * VkSurfaceKHR is the platform primitive, so nothing here has a Win32, Wayland or X11 spelling.
     */
    [[nodiscard]] std::expected<std::unique_ptr<IExternalTextureProvider>, DeviceError>
    CreateExternalTextureProvider(IDevice &device, ExternalTextureOwner owner,
                                  Format        requestedFormat = Format::Undefined,
                                  PresentTiming timing          = PresentTiming::OnRefresh);

    /**
     * {brief} What a caller must obtain from the device before making the platform object.
     *
     * A Vulkan surface cannot exist before the instance does, which is why it is asked for here
     * rather than passed to CreateDevice: open the device, take this, make the surface from it,
     * then hand the surface back above. The value is the VkInstance.
     */
    [[nodiscard]] ExternalTextureOwnerContext ExternalTextureOwnerContextOf(IDevice &device);

    /**
     * {brief} Acquires the process-wide synchronized Vulkan device.
     * {param desc} Must match the validation settings of an already-live shared device.
     * {returns} Shared ownership suitable for coordinated access from multiple consumers.
     */
    [[nodiscard]] std::expected<SharedDevice, DeviceError> AcquireSharedDevice(const DeviceDesc &desc = {});

} // namespace rhi
