// vulkan_vma.cpp — single implementation unit for Vulkan Memory Allocator.
//
// Keep this outside the lightRHI module so VMA's implementation-only system
// includes, including compiler intrinsic headers on Windows, stay in the global
// module fragment.

// volk first: it defines VK_NO_PROTOTYPES, so VMA compiles its implementation
// against dynamically-loaded function pointers (fetched from the
// vkGetInstanceProcAddr / vkGetDeviceProcAddr we hand VMA at allocator
// creation) rather than a linked loader.
#include <volk.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
