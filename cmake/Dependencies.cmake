# cmake/Dependencies.cmake
# Resolves all external dependencies for LightRHI.
#
# Pattern: try find_package() first (picks up vcpkg or system installs),
# fall back to FetchContent if not found. This makes LightRHI fully
# self-contained when used as a submodule — the parent project does NOT
# need to know or install LightRHI's dependencies.

include(FetchContent)

# ============================================================================
# macOS — Metal backend via metal-cpp (header-only, pure C++)
# ============================================================================
if(APPLE)
    find_library(METAL_FRAMEWORK    Metal        REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore  REQUIRED)

    # metal-cpp — Apple's official C++ wrapper for Metal
    find_package(metal-cpp CONFIG QUIET)
    if(NOT TARGET metal-cpp::metal-cpp)
        if(EXISTS "${CMAKE_BINARY_DIR}/_deps/metal-cpp-src")
            message(STATUS "[LightRHI] metal-cpp not found via find_package — reusing cached FetchContent fetch")
        else()
            message(STATUS "[LightRHI] metal-cpp not found via find_package — fetching via FetchContent")
        endif()
        FetchContent_Declare(metal-cpp
            GIT_REPOSITORY https://github.com/bkaradzic/metal-cpp.git
            GIT_TAG        metal-cpp_27
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(metal-cpp)

        # bkaradzic/metal-cpp is header-only with no CMakeLists.
        # Create an interface target manually.
        if(NOT TARGET metal-cpp::metal-cpp)
            add_library(metal-cpp INTERFACE)
            add_library(metal-cpp::metal-cpp ALIAS metal-cpp)
            target_include_directories(metal-cpp SYSTEM INTERFACE
                $<BUILD_INTERFACE:${metal-cpp_SOURCE_DIR}>)
        endif()
    endif()

    message(STATUS "[LightRHI] metal-cpp: ${metal-cpp_SOURCE_DIR}")

# ============================================================================
# Linux / Windows — Vulkan backend
# ============================================================================
else()
    # Vulkan headers — NOT the loader import library.
    #
    # The backend loads Vulkan through volk at runtime (LoadLibrary/dlopen of
    # the driver-provided loader, which ships with every GPU driver), so we
    # never link the SDK's vulkan-1.lib / libvulkan.so. That means we only
    # need the *headers*: prefer a system SDK when one is installed, otherwise
    # fetch Vulkan-Headers into _deps — no SDK install and no loader compile.
    find_package(Vulkan QUIET)
    if(TARGET Vulkan::Headers)
        message(STATUS "[LightRHI] Vulkan headers: system SDK (${Vulkan_VERSION})")
    else()
        message(STATUS "[LightRHI] Vulkan SDK not found — fetching Vulkan-Headers via FetchContent")
        FetchContent_Declare(VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG        vulkan-sdk-1.3.290.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(VulkanHeaders)   # provides Vulkan::Headers

        # volk and VMA don't consume the Vulkan::Headers target — they resolve
        # includes through the FindVulkan variables (Vulkan_INCLUDE_DIR(S)),
        # which are empty when no SDK was found. Point them at the fetched
        # headers so both compile against them.
        set(Vulkan_INCLUDE_DIR  "${vulkanheaders_SOURCE_DIR}/include" CACHE PATH "Vulkan headers" FORCE)
        set(Vulkan_INCLUDE_DIRS "${vulkanheaders_SOURCE_DIR}/include" CACHE PATH "Vulkan headers" FORCE)
    endif()

    # VulkanMemoryAllocator (links Vulkan::Headers)
    find_package(VulkanMemoryAllocator CONFIG QUIET)
    if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
        message(STATUS "[LightRHI] VulkanMemoryAllocator not found — fetching via FetchContent")
        FetchContent_Declare(VulkanMemoryAllocator
            GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
            GIT_TAG        v3.1.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(VulkanMemoryAllocator)
    endif()

    # volk — optional dynamic Vulkan loader
    find_package(volk CONFIG QUIET)
    if(NOT TARGET volk::volk)
        message(STATUS "[LightRHI] volk not found — fetching via FetchContent")
        FetchContent_Declare(volk
            GIT_REPOSITORY https://github.com/zeux/volk.git
            GIT_TAG        1.3.270
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(volk)
    endif()
endif()
