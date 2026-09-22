/**
 * {file}
 * {brief} Imports the complete LightRHI API and selects its Metal device factory.
 *
 * ```cpp
 * import lightRHI;
 *
 * auto device{rhi::CreateDevice({
 *     .EnableValidation = true,
 *     .AppName = "MyApp",
 * })};
 * ```
 * {note} The Metal backend targets Metal 4, argument-buffer tier 2 bindless
 * resources, indirect command buffers, and resource heaps.
 *
 * Shaders are normally authored in Slang and supplied as MSL source or a
 * precompiled Metal library through ShaderDesc. A standalone MSL library can
 * be produced with:
 * ```sh
 * xcrun -sdk macosx metal -c shader.metal -o shader.air
 * xcrun -sdk macosx metallib shader.air -o shader.metallib
 * ```
 */

module;
#include <string_view>
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
     * {note} A `.metallib` compiled ahead of time, or `.metal` source when the build ships MSL.
     */
    [[nodiscard]] std::expected<ShaderLibrary, ShaderArtifactError> ReadShaderLibrary(std::string_view directory,
                                                                                     std::string_view library);

    /**
     * {brief} Creates an independently owned Metal device.
     * {param desc} Validation and application metadata used during creation.
     * {returns} Exclusive ownership of a device; destroy its resources before releasing it.
     */
    [[nodiscard]] std::expected<std::unique_ptr<IDevice>, DeviceError> CreateDevice(const DeviceDesc &desc = {});

    /**
     * {brief} Lends out the textures of `layer`, so `device` can draw them and show them.
     * {param device} The device that will draw into them; it outlives the provider.
     * {param layer} Stays the caller's, and must outlive the provider.
     * {param requestedFormat} Undefined takes the device's preferred format.
     * {returns} Exclusive ownership of the provider, or InvalidArgument when the layer is unusable.
     * {note} One device serves as many providers as an application has layers.
     * {note} Named by the concrete type, not by a callback: this build is the Metal one, so the
     * layer already exists by the time a device does and there is nothing to call back for.
     */
    [[nodiscard]] std::expected<std::unique_ptr<IExternalTextureProvider>, DeviceError>
    CreateExternalTextureProvider(IDevice &device, ExternalTextureOwner owner,
                                  Format requestedFormat = Format::Undefined);

    /**
     * {brief} What a caller must obtain from the device before making the platform object.
     * {returns} Empty on this backend: the layer belongs to the window already.
     */
    [[nodiscard]] ExternalTextureOwnerContext ExternalTextureOwnerContextOf(IDevice &device);

    /**
     * {brief} Acquires the process-wide synchronized Metal device.
     * {param desc} Must match the validation settings of an already-live shared device.
     * {returns} Shared ownership suitable for coordinated access from multiple consumers.
     */
    [[nodiscard]] std::expected<SharedDevice, DeviceError> AcquireSharedDevice(const DeviceDesc &desc = {});

} // namespace rhi
