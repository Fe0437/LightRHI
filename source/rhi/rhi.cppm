/**
 * {file}
 * {brief} Re-exports the complete backend-neutral LightRHI API as `rhi`.
 *
 * ```cpp
 * import rhi;
 * ```
 * Applications normally import the backend's `lightRHI` module, which re-exports
 * this module and provides CreateDevice().
 */

export module rhi;

// Re-export all partitions
export import :diagnostics;
export import :types;
export import :handles;
export import :sync;
export import :descriptors;
export import :externalTextures;
export import :pipeline;
export import :shaderLibrary;
export import :resources;
export import :bindless;
export import :raytracing;
export import :commandList;
export import :device;
export import :sharedDevice;
export import :shaderDebug;
