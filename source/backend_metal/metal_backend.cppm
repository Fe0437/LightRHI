// metal_backend.cppm
// Public module for the LightRHI Metal backend (macOS / Apple Silicon).
//
// Consumer usage:
//   import lightRHI;
//
//   auto device = rhi::CreateDevice({
//       .EnableValidation = true,
//       .AppName          = "MyApp",
//   });
//
// The Metal backend targets:
//   - Metal 3+ (Apple Silicon, Intel Mac with macOS 13+)
//   - Argument buffers tier 2  (bindless resource access)
//   - Indirect command buffers (GPU-driven rendering)
//   - Resource heaps           (aliased memory)
//
// Shaders are authored in Slang and compiled to MSL via:
//   slangc shader.slang -target metal -entry <name> -o shader.metal
// Pre-compile to .metallib for faster startup:
//   xcrun -sdk macosx metal -c shader.metal -o shader.air
//   xcrun -sdk macosx metallib shader.air -o shader.metallib

module;
#include <memory>

export module lightRHI;
export import rhi; // re-exports all rhi types and interfaces to consumers

export namespace rhi
{

    // Factory — defined in metal_device.cpp (module implementation unit).
    // No default argument: avoids Clang generating a call-site wrapper in every
    // module implementation unit, which would cause duplicate symbol errors.
    [[nodiscard]] std::unique_ptr<IDevice> CreateDevice(const DeviceDesc &desc);
    [[nodiscard]] SharedDevice             AcquireSharedDevice(const DeviceDesc &desc);

} // namespace rhi
