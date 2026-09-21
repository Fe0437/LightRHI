add_executable(rhi_tests rhi_tests.cpp)
target_compile_features(rhi_tests PRIVATE cxx_std_23)
target_compile_options(rhi_tests PRIVATE -fno-rtti)
target_link_libraries(rhi_tests PRIVATE LightRHI::LightRHI LightRHI::Backend)
target_warnings(rhi_tests)

set(RHI_TEST_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders")
if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    target_compile_definitions(rhi_tests PRIVATE RHI_BACKEND_METAL=1)
else()
    target_compile_definitions(rhi_tests PRIVATE RHI_BACKEND_VULKAN=1)
endif()

light_rhi_compile_shaders(
    TARGET       rhi_tests
    SHADERS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders_registry.generated.json"
    OUTPUT_DIR   "${RHI_TEST_SHADER_DIR}")

target_compile_definitions(rhi_tests PRIVATE
    RHI_TEST_SHADER_DIR="${RHI_TEST_SHADER_DIR}"
)

add_test(NAME rhi_unit_tests COMMAND rhi_tests)
# Tag as smoke: these run without a GPU and finish in milliseconds.
#
# On Windows the RHI DLLs (LightRHI.dll / LightRHIBackend.dll) live in the build
# root, not next to rhi_tests.exe, so ctest's working directory can't load them
# — the process dies with STATUS_DLL_NOT_FOUND (or worse, blocks on a WER dialog
# and hangs the run). Prepend the DLL directories to PATH for the test. Harmless
# on Linux/macOS (they resolve shared libs via RPATH, but path_list_prepend is
# portable).
set_tests_properties(rhi_unit_tests PROPERTIES
    LABELS "smoke"
    # Bounded so a GPU wait that never returns fails the run instead of hanging it.
    TIMEOUT 120
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHI>;PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHIBackend>")

# Memory blocks and placed textures. Separate from rhi_tests so the placement contract and its
# cost measurement can be selected on their own.
add_executable(rhi_placement_tests rhi_placement_tests.cpp)
target_compile_features(rhi_placement_tests PRIVATE cxx_std_23)
target_compile_options(rhi_placement_tests PRIVATE -fno-rtti)
target_link_libraries(rhi_placement_tests PRIVATE LightRHI::LightRHI LightRHI::Backend)
target_warnings(rhi_placement_tests)
add_dependencies(rhi_placement_tests rhi_tests_shaders)
target_compile_definitions(rhi_placement_tests PRIVATE RHI_TEST_SHADER_DIR="${RHI_TEST_SHADER_DIR}")
if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    target_compile_definitions(rhi_placement_tests PRIVATE RHI_BACKEND_METAL=1)
else()
    target_compile_definitions(rhi_placement_tests PRIVATE RHI_BACKEND_VULKAN=1)
endif()

add_test(NAME rhi_placement_tests COMMAND rhi_placement_tests)
set_tests_properties(rhi_placement_tests PROPERTIES
    LABELS "placement"
    # Bounded so a GPU wait that never returns fails the run instead of hanging it.
    TIMEOUT 120
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHI>;PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHIBackend>")

# Presentation: surfaces, acquired frames and present-on-submit. LightRHI never creates a surface,
# so the harness supplies one - and which surface that is, is the one fact that differs per backend.
# It is written here rather than in the test source, so the test exercises the same contract on
# every backend and names none of them.
#
# Both are surfaces with no window behind them, so the cases run on a machine with no desktop
# session. `kTestTextureSourceHonoursExtent` says whether this surface hands out the extent it was asked
# for; a layer does, a headless swapchain picks its own, and the extent assertions follow that.
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

set(_presentation_test_texture_owner_header "${CMAKE_CURRENT_BINARY_DIR}/generated/test_texture_owner.generated.h")
file(GENERATE OUTPUT "${_presentation_test_texture_owner_header}" CONTENT
"// Generated by tests/CMakeLists.txt - do not edit.
#pragma once

#include <cstdint>

${_presentation_test_texture_owner}")

add_executable(rhi_presentation_tests rhi_presentation_tests.cpp)
target_compile_features(rhi_presentation_tests PRIVATE cxx_std_23)
target_compile_options(rhi_presentation_tests PRIVATE -fno-rtti)
target_include_directories(rhi_presentation_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(rhi_presentation_tests PRIVATE LightRHI::LightRHI LightRHI::Backend)
target_warnings(rhi_presentation_tests)
# The generated surface reaches for the selected backend's own headers.
if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    target_link_libraries(rhi_presentation_tests PRIVATE
        metal-cpp::metal-cpp
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK})
else()
    target_link_libraries(rhi_presentation_tests PRIVATE Vulkan::Headers)
endif()

add_test(NAME rhi_presentation_tests COMMAND rhi_presentation_tests)
set_tests_properties(rhi_presentation_tests PROPERTIES
    LABELS "presentation"
    # Bounded so a GPU wait that never returns fails the run instead of hanging it.
    TIMEOUT 120
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHI>;PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHIBackend>")

# Focused bindless integration coverage. Kept separate from rhi_tests so it can
# be selected in CI without running unrelated raster tests.
add_executable(rhi_bindless_tests rhi_bindless_tests.cpp)
target_compile_features(rhi_bindless_tests PRIVATE cxx_std_23)
target_compile_options(rhi_bindless_tests PRIVATE -fno-rtti)
target_link_libraries(rhi_bindless_tests PRIVATE LightRHI::LightRHI LightRHI::Backend)
target_warnings(rhi_bindless_tests)
add_dependencies(rhi_bindless_tests rhi_tests_shaders)

if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    target_compile_definitions(rhi_bindless_tests PRIVATE RHI_BACKEND_METAL=1)
else()
    target_compile_definitions(rhi_bindless_tests PRIVATE RHI_BACKEND_VULKAN=1)
endif()
target_compile_definitions(rhi_bindless_tests PRIVATE RHI_TEST_SHADER_DIR="${RHI_TEST_SHADER_DIR}")

add_test(NAME rhi_bindless_tests COMMAND rhi_bindless_tests)
set_tests_properties(rhi_bindless_tests PROPERTIES
    LABELS "bindless"
    # Bounded so a GPU wait that never returns fails the run instead of hanging it.
    TIMEOUT 120
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHI>;PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHIBackend>")

# -------------------------------------------------------------------
# Ray tracing tests — separate executable (own int main()), same
# shader manifest/output dir as rhi_tests (already compiled above;
# reuse rhi_tests_shaders instead of re-invoking light_rhi_compile_shaders,
# which would register a second custom rule for the same output files).
# -------------------------------------------------------------------
add_executable(rhi_raytracing_tests rhi_raytracing_tests.cpp)
target_compile_features(rhi_raytracing_tests PRIVATE cxx_std_23)
target_compile_options(rhi_raytracing_tests PRIVATE -fno-rtti)
target_link_libraries(rhi_raytracing_tests PRIVATE LightRHI::LightRHI LightRHI::Backend)
target_warnings(rhi_raytracing_tests)

if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    target_compile_definitions(rhi_raytracing_tests PRIVATE RHI_BACKEND_METAL=1)
else()
    target_compile_definitions(rhi_raytracing_tests PRIVATE RHI_BACKEND_VULKAN=1)
endif()

add_dependencies(rhi_raytracing_tests rhi_tests_shaders)
target_compile_definitions(rhi_raytracing_tests PRIVATE
    RHI_TEST_SHADER_DIR="${RHI_TEST_SHADER_DIR}"
)

add_test(NAME rhi_raytracing_tests COMMAND rhi_raytracing_tests)
# Not tagged smoke: builds/dispatches real acceleration structures on a GPU.
# Same Windows DLL-path caveat as rhi_unit_tests above.
set_tests_properties(rhi_raytracing_tests PROPERTIES
    LABELS "raytracing"
    # Bounded so a GPU wait that never returns fails the run instead of hanging it.
    TIMEOUT 120
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHI>;PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHIBackend>")
