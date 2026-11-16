/**
 * {file}
 * {brief} Acquires Metal encoders with owned return values for the C++ command list.
 *
 * Excludes recording and submission. The Swift implementation accepts Metal's autoreleased return
 * directly, so ownership does not depend on a Cocoa event loop on the calling thread. That relies on
 * the bridge being built with -O; docs/ARCHITECTURE.md explains why, and what was tried instead.
 */
#ifndef LIGHTRHI_METAL_ENCODER_OWNERSHIP_H
#define LIGHTRHI_METAL_ENCODER_OWNERSHIP_H

#include <Metal/Metal.hpp>

namespace rhi::metal
{
    /** {brief} Swift's C entry point; borrows the command buffer and returns one owned native encoder or null. */
    extern "C" [[nodiscard]] void *LightRHICreateOwnedComputeEncoder(void *commands) noexcept;

    /** {brief} Swift's C entry point; borrows both arguments and returns one owned native render encoder or null. */
    extern "C" [[nodiscard]] void *LightRHICreateOwnedRenderEncoder(void *commands, void *descriptor) noexcept;

    /** {brief} One owned compute encoder; transfer to a smart pointer and release after encoding. */
    [[nodiscard]] inline MTL4::ComputeCommandEncoder *CreateOwnedComputeEncoder(MTL4::CommandBuffer &commands) noexcept
    {
        return static_cast<MTL4::ComputeCommandEncoder *>(LightRHICreateOwnedComputeEncoder(&commands));
    }

    /** {brief} One owned render encoder; the descriptor is borrowed for this call only. */
    [[nodiscard]] inline MTL4::RenderCommandEncoder *
    CreateOwnedRenderEncoder(MTL4::CommandBuffer &commands, MTL4::RenderPassDescriptor &descriptor) noexcept
    {
        return static_cast<MTL4::RenderCommandEncoder *>(LightRHICreateOwnedRenderEncoder(&commands, &descriptor));
    }
} // namespace rhi::metal

#endif
