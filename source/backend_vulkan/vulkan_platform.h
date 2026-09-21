// vulkan_platform.h — the Vulkan and platform headers the backend is built on.
//
// Include this from each translation unit's GLOBAL MODULE FRAGMENT, before `module lightRHI;`.
// Including Vulkan's headers after the module declaration would attach their declarations to the
// module, which conflicts with the copies the fragment already brought in. vulkan_internal.h
// includes this too, so a unit that forgets it fails loudly rather than subtly.

#pragma once

// clang-format off -- Volk must define VK_NO_PROTOTYPES before VMA includes Vulkan.
//
// CMake detects the native headers and defines every surface integration this target supports.
// Keep Vulkan's feature macros here so every backend translation unit sees the same declarations.
#if defined(LIGHT_RHI_VULKAN_HAS_WIN32_SURFACE)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#if defined(LIGHT_RHI_VULKAN_HAS_WAYLAND_SURFACE)
#define VK_USE_PLATFORM_WAYLAND_KHR
#endif
#if defined(LIGHT_RHI_VULKAN_HAS_XCB_SURFACE)
#define VK_USE_PLATFORM_XCB_KHR
#endif
#if defined(LIGHT_RHI_VULKAN_HAS_XLIB_SURFACE)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <volk.h> // defines VK_NO_PROTOTYPES + pulls in <vulkan/vulkan.h>
#if !defined(VK_NO_PROTOTYPES)
#error "Volk must define VK_NO_PROTOTYPES before VMA is included"
#endif
#include <vk_mem_alloc.h> // must follow volk so VMA sees VK_NO_PROTOTYPES
// clang-format on
