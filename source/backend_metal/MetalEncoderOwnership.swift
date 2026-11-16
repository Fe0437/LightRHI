/**
 * {file}
 * {brief} Transfers owned Metal encoders to C++ while borrowing their command buffer and descriptor.
 *
 * Excludes recording, submission and autorelease pools around painting batches.
 */
import Foundation
import Metal

/// An owned compute encoder, or nil when creation fails; C++ releases the returned retain.
@_cdecl("LightRHICreateOwnedComputeEncoder")
public func createOwnedMetalComputeEncoder(_ commands: UnsafeMutableRawPointer)
  -> UnsafeMutableRawPointer?
{
  // C++ supplies an MTL4CommandBuffer's native object pointer. Its Objective-C protocol has the
  // same pointer representation, so no runtime conformance check or input retain is needed.
  // No autorelease pool here, on purpose: this file is built with -O, so Swift claims the returned
  // encoder at once and leaves nothing pending, at no cost. A pool would cost every dispatch.
  // See "Metal encoders on threads with no autorelease pool" in docs/ARCHITECTURE.md before changing
  // this; tests/encoder_ownership fails if creation ever leaves an object pending.
  let native = unsafeBitCast(commands, to: (any MTL4CommandBuffer).self)
  guard let encoder = native.makeComputeCommandEncoder() else {
    return nil
  }
  return Unmanaged.passRetained(encoder as AnyObject).toOpaque()
}

/// An owned render encoder for the borrowed descriptor, or nil when creation fails.
@_cdecl("LightRHICreateOwnedRenderEncoder")
public func createOwnedMetalRenderEncoder(
  _ commands: UnsafeMutableRawPointer, _ descriptor: UnsafeMutableRawPointer
) -> UnsafeMutableRawPointer? {
  // The driver also autoreleases an internal render context. Drain creation's temporaries while
  // the returned retain keeps the encoder alive; compute recording needs no pool.
  autoreleasepool {
    let native = unsafeBitCast(commands, to: (any MTL4CommandBuffer).self)
    let pass = Unmanaged<MTL4RenderPassDescriptor>.fromOpaque(descriptor).takeUnretainedValue()
    guard let encoder = native.makeRenderCommandEncoder(descriptor: pass) else {
      return nil
    }
    return Unmanaged.passRetained(encoder as AnyObject).toOpaque()
  }
}
