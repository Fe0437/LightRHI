// RHI.cppm — top-level module unit.
// Consumers write:  import rhi;
// and get everything: types, handles, descriptors, pipelines, device, command list.

export module rhi;

// Re-export all partitions
export import :types;
export import :handles;
export import :sync;
export import :descriptors;
export import :pipeline;
export import :resources;
export import :bindless;
export import :raytracing;
export import :commandList;
export import :device;
export import :shaderDebug;
