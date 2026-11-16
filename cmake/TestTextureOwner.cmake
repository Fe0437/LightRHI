# A surface with no window behind it, for tests of presentation: a layer that belongs to no window on Apple,
# a headless surface on Vulkan. LightRHI never creates a surface, so whoever tests presentation supplies one,
# and which one that is is the one fact that differs per backend. It is written here, once, so every test
# that presents - this library's own and those of the code built on it - exercises the same contract and
# names no backend.
#
# `light_rhi_add_test_texture_owner(TARGET <target>)` gives the target `test_texture_owner.generated.h`,
# namespace `rhitest`: `TestTextureOwner`, `kTestTextureOwnerHonoursExtent`, `MakeTestTextureOwner`,
# `ResizeTestTextureOwner`, `ReleaseTestTextureOwner` and `MakeProviderFor`. Include it from exactly one
# translation unit of the target: on Metal it is where metal-cpp emits its definitions.
# The function links the backend's own headers from the caller's directory. A target find_package() imports is
# seen only in the directory that found it, so those found here are made global once, as this is included.
foreach(_light_rhi_owner_dependency IN ITEMS metal-cpp::metal-cpp Vulkan::Headers)
  if(TARGET ${_light_rhi_owner_dependency})
    get_target_property(_light_rhi_is_imported ${_light_rhi_owner_dependency} IMPORTED)
    get_target_property(_light_rhi_is_global ${_light_rhi_owner_dependency} IMPORTED_GLOBAL)
    if(_light_rhi_is_imported AND NOT _light_rhi_is_global)
      set_property(TARGET ${_light_rhi_owner_dependency} PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()

function(light_rhi_add_test_texture_owner)
  cmake_parse_arguments(ARG "" "TARGET" "" ${ARGN})
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "light_rhi_add_test_texture_owner requires TARGET")
  endif()
  set(_directory "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_generated")
  set(_header "${_directory}/test_texture_owner.generated.h")
  if(LIGHT_RHI_BACKEND STREQUAL "Metal")
      set(_presentation_test_texture_owner
"// Included by exactly one translation unit, which is where metal-cpp emits its definitions.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace rhitest
{
    /** {brief} What owns the textures a provider lends out: here, a layer belonging to no window. */
    using TestTextureOwner = CA::MetalLayer *;

    /** {brief} A detached layer vends drawables at the size it is told, with no window server. */
    inline constexpr bool kTestTextureOwnerHonoursExtent{true};

    [[nodiscard]] inline TestTextureOwner MakeTestTextureOwner(rhi::IDevice & /*device*/, std::uint32_t width,
                                                               std::uint32_t height)
    {
        auto *layer{CA::MetalLayer::layer()->retain()};
        layer->setDrawableSize(CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height)));
        return layer;
    }

    /** {brief} Resizes what the textures come from, the way a window manager would. */
    [[nodiscard]] inline bool ResizeTestTextureOwner(TestTextureOwner owner, std::uint32_t width, std::uint32_t height)
    {
        owner->setDrawableSize(CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height)));
        return true;
    }

    inline void ReleaseTestTextureOwner(TestTextureOwner owner)
    {
        if (owner != nullptr)
        {
            owner->release();
        }
    }

    /** {brief} The one line that differs per backend: how an owner reaches the factory. */
    [[nodiscard]] inline auto MakeProviderFor(rhi::IDevice &device, TestTextureOwner owner, rhi::Format format)
    {
        return rhi::CreateExternalTextureProvider(device, rhi::ExternalTextureOwner{.Value = owner}, format);
    }
} // namespace rhitest
")
  else()
      set(_presentation_test_texture_owner
"#include <vulkan/vulkan.h>

namespace rhitest
{
    /** {brief} What owns the textures a provider lends out: here, a surface with no window. */
    using TestTextureOwner = VkSurfaceKHR;

    /** {brief} A headless swapchain chooses its own extent, so frames are not the size asked for. */
    inline constexpr bool kTestTextureOwnerHonoursExtent{false};

    /** {brief} Made from the device's instance, which is why the device comes first. */
    [[nodiscard]] inline TestTextureOwner MakeTestTextureOwner(rhi::IDevice &device, std::uint32_t /*width*/,
                                                               std::uint32_t /*height*/)
    {
        const auto instance{static_cast<VkInstance>(rhi::ExternalTextureOwnerContextOf(device).Value)};
        const auto create{reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
            vkGetInstanceProcAddr(instance, \"vkCreateHeadlessSurfaceEXT\"))};
        if (create == nullptr)
        {
            return VK_NULL_HANDLE; // no headless surface on this driver; the caller skips
        }
        const VkHeadlessSurfaceCreateInfoEXT info{.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
        VkSurfaceKHR                         surface{VK_NULL_HANDLE};
        return create(instance, &info, nullptr, &surface) == VK_SUCCESS ? surface : VK_NULL_HANDLE;
    }

    /** {brief} Nothing drives a headless owner's size from out here. */
    [[nodiscard]] inline bool ResizeTestTextureOwner(TestTextureOwner /*owner*/, std::uint32_t /*width*/,
                                                    std::uint32_t /*height*/)
    {
        return false;
    }

    /** {brief} The provider takes ownership of the surface, so nothing is destroyed here. */
    inline void ReleaseTestTextureOwner(TestTextureOwner /*owner*/) {}

    /** {brief} The one line that differs per backend: how an owner reaches the factory. */
    [[nodiscard]] inline auto MakeProviderFor(rhi::IDevice &device, TestTextureOwner owner, rhi::Format format)
    {
        return rhi::CreateExternalTextureProvider(
            device, rhi::ExternalTextureOwner{.Value = reinterpret_cast<void *>(owner)}, format);
    }
} // namespace rhitest
")
  endif()

  file(GENERATE OUTPUT "${_header}" CONTENT
"// Generated by light_rhi_add_test_texture_owner() - do not edit.
#pragma once

#include <cstdint>

${_presentation_test_texture_owner}")

  target_include_directories(${ARG_TARGET} PRIVATE "${_directory}")
  # The generated surface reaches for the selected backend's own headers.
  if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    target_link_libraries(${ARG_TARGET} PRIVATE
      metal-cpp::metal-cpp
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
      ${QUARTZCORE_FRAMEWORK})
  else()
    target_link_libraries(${ARG_TARGET} PRIVATE Vulkan::Headers)
  endif()
endfunction()
