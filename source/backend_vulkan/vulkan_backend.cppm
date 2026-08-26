/// \file
/// \brief Imports the complete LightRHI API and selects its Vulkan device factory.
///
/// \code{.cpp}
/// import lightRHI;
///
/// auto device = rhi::CreateDevice({
///     .EnableValidation = true,
///     .AppName = "MyApp",
/// });
/// \endcode
/// \note The Vulkan backend requires Vulkan 1.3, synchronization2, dynamic
/// rendering, timeline semaphores, buffer device addresses, descriptor
/// indexing, and VK_EXT_descriptor_buffer. Supply SPIR-V through ShaderDesc
/// when creating pipelines.

module;
#include <memory>

export module lightRHI;
export import rhi; // re-exports all rhi types and interfaces to consumers

export namespace rhi
{

    /// \brief Creates an independently owned Vulkan device.
    /// \param desc Validation and application metadata used during creation.
    /// \return Exclusive ownership of a device; destroy its resources before releasing it.
    [[nodiscard]] std::unique_ptr<IDevice> CreateDevice(const DeviceDesc &desc = {});

    /// \brief Acquires the process-wide synchronized Vulkan device.
    /// \param desc Must match the validation settings of an already-live shared device.
    /// \return Shared ownership suitable for coordinated access from multiple consumers.
    [[nodiscard]] SharedDevice AcquireSharedDevice(const DeviceDesc &desc = {});

} // namespace rhi
