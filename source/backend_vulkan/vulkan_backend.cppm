// vulkan_backend.cppm
// Public module for the LightRHI Vulkan backend (Linux / Windows).
//
// Consumer usage:
//   import lightRHI;
//   auto device = rhi::CreateDevice({ .EnableValidation = true });
//
// Required Vulkan features (all promoted to core by 1.3):
//   VK_KHR_synchronization2, VK_KHR_dynamic_rendering,
//   VK_KHR_timeline_semaphore, VK_KHR_buffer_device_address,
//   VK_EXT_descriptor_indexing
// Still an extension as of Vulkan 1.3:
//   VK_EXT_descriptor_buffer

module;
#include <memory>

export module lightRHI;
export import rhi; // re-exports all rhi types and interfaces to consumers

export namespace rhi
{

    [[nodiscard]] std::unique_ptr<IDevice> CreateDevice(const DeviceDesc &desc = {});
    [[nodiscard]] SharedDevice             AcquireSharedDevice(const DeviceDesc &desc = {});

} // namespace rhi
