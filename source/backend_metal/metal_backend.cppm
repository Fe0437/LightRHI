/// \file
/// \brief Imports the complete LightRHI API and selects its Metal device factory.
///
/// \code{.cpp}
/// import lightRHI;
///
/// auto device = rhi::CreateDevice({
///     .EnableValidation = true,
///     .AppName = "MyApp",
/// });
/// \endcode
/// \note The Metal backend targets Metal 4, argument-buffer tier 2 bindless
/// resources, indirect command buffers, and resource heaps.
///
/// Shaders are normally authored in Slang and supplied as MSL source or a
/// precompiled Metal library through ShaderDesc. A standalone MSL library can
/// be produced with:
/// \code{.sh}
/// xcrun -sdk macosx metal -c shader.metal -o shader.air
/// xcrun -sdk macosx metallib shader.air -o shader.metallib
/// \endcode

module;
#include <memory>

export module lightRHI;
export import rhi; // re-exports all rhi types and interfaces to consumers

export namespace rhi
{

    /// \brief Creates an independently owned Metal device.
    /// \param desc Validation and application metadata used during creation.
    /// \return Exclusive ownership of a device; destroy its resources before releasing it.
    [[nodiscard]] std::unique_ptr<IDevice> CreateDevice(const DeviceDesc &desc);

    /// \brief Acquires the process-wide synchronized Metal device.
    /// \param desc Must match the validation settings of an already-live shared device.
    /// \return Shared ownership suitable for coordinated access from multiple consumers.
    [[nodiscard]] SharedDevice AcquireSharedDevice(const DeviceDesc &desc);

} // namespace rhi
