cmake_minimum_required(VERSION 3.28)   # required for C++23 module scanning
project(LightRHI VERSION 0.1.0 LANGUAGES CXX)

# -------------------------------------------------------------------
# Standalone vs submodule detection
# When used via add_subdirectory(), CMAKE_SOURCE_DIR != CMAKE_CURRENT_SOURCE_DIR.
# We use this to gate things that only make sense when building standalone.
# -------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_LIGHTRHI_STANDALONE ON)
else()
    set(_LIGHTRHI_STANDALONE OFF)
endif()

# -------------------------------------------------------------------
# C++ standard
# -------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------------------------
# Build type default (standalone only — don't override parent project)
# -------------------------------------------------------------------
if(_LIGHTRHI_STANDALONE AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo"
        CACHE STRING "Build type" FORCE)
endif()

# -------------------------------------------------------------------
# CMake module path
# -------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Warnings)
include(Sanitizers)

# -------------------------------------------------------------------
# Backend selection — driven by target OS, not by user option.

# A development build also compiles the backend this platform does not build, so a change to it
# fails here rather than on someone else's machine. Shipping builds compile only what they ship.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(_light_rhi_check_default OFF)
else()
    set(_light_rhi_check_default ON)
endif()
option(LIGHT_RHI_CHECK_INACTIVE_BACKEND "Compile the backend this platform does not build"
       ${_light_rhi_check_default})

# One backend per platform; the selection is not a knob.
# -------------------------------------------------------------------
if(APPLE)
    set(LIGHT_RHI_BACKEND "Metal" CACHE INTERNAL "Active LightRHI backend (Metal/Vulkan) — not user-settable")
else()
    set(LIGHT_RHI_BACKEND "Vulkan" CACHE INTERNAL "Active LightRHI backend (Metal/Vulkan) — not user-settable")
endif()
message(STATUS "[LightRHI] Backend: ${LIGHT_RHI_BACKEND}")

# -------------------------------------------------------------------
# Options
# Examples and tests default ON when standalone, OFF as a submodule
# so the parent build isn't cluttered with our executables.
# -------------------------------------------------------------------
option(LIGHT_RHI_BUILD_EXAMPLES "Build example programs"            ${_LIGHTRHI_STANDALONE})
option(LIGHT_RHI_BUILD_TESTS    "Build tests"                       ${_LIGHTRHI_STANDALONE})
option(LIGHT_RHI_ENABLE_ASAN    "Enable AddressSanitizer"           OFF)
option(LIGHT_RHI_ENABLE_UBSAN   "Enable UndefinedBehaviorSanitizer" OFF)
option(LIGHT_RHI_ENABLE_TSAN    "Enable ThreadSanitizer"            OFF)

if(NOT DEFINED DEBUG_ENABLED)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(_light_rhi_debug_default ON)
    else()
        set(_light_rhi_debug_default OFF)
    endif()
    option(DEBUG_ENABLED "Enable LightRHI debug-only host and shader facilities" ${_light_rhi_debug_default})
endif()
# What LightRHI measures for its caller, such as when a presented frame reached the screen. A
# shipping build leaves it out. A debug build always measures too; a parent project's choice wins.
if(NOT DEFINED METRICS_ENABLED)
    option(METRICS_ENABLED "Compile what LightRHI measures for its caller" ${DEBUG_ENABLED})
endif()
if(DEBUG_ENABLED AND NOT METRICS_ENABLED)
    set(METRICS_ENABLED ON CACHE BOOL "Compile what LightRHI measures for its caller" FORCE)
endif()

# -------------------------------------------------------------------
# External dependencies (platform-specific)
# -------------------------------------------------------------------
include(Dependencies)
include(Slang)
include(Shaders)

# -------------------------------------------------------------------
# Shared-library RPATH setup (macOS)
# Ensures dylibs find each other both in the build tree and after install.
# CMAKE_MACOSX_RPATH is ON by default since CMake 3.28, listed here
# explicitly for clarity.  BUILD_WITH_INSTALL_RPATH stays OFF so the
# build-tree executables (run_tests.sh) resolve dylibs from the build dir.
# -------------------------------------------------------------------
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

# -------------------------------------------------------------------
# Core library — module interface, backend-neutral
# -------------------------------------------------------------------

add_library(LightRHI SHARED)
add_library(LightRHI::LightRHI ALIAS LightRHI)
if(WIN32)
    set_target_properties(LightRHI PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_compile_features(LightRHI PUBLIC cxx_std_23)
# Failure is reported through return values, never by unwinding. These language settings must reach
# every C++ consumer because a named module cannot be imported across a configuration mismatch.
# The language guard keeps the options away from Swift or C sources in a mixed-language consumer.
target_compile_options(LightRHI PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)
target_compile_definitions(LightRHI PUBLIC
    $<$<BOOL:${DEBUG_ENABLED}>:DEBUG_ENABLED=1>
    $<$<NOT:$<BOOL:${DEBUG_ENABLED}>>:DEBUG_ENABLED=0>
    $<$<BOOL:${METRICS_ENABLED}>:METRICS_ENABLED=1>
    $<$<NOT:$<BOOL:${METRICS_ENABLED}>>:METRICS_ENABLED=0>)
target_include_directories(LightRHI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shaders>)

# Module partitions. CMake 3.28 scans these with clang-scan-deps and derives
# the build order from the module graph itself, so the file list need not be
# hand-ordered — glob the whole module directory.
file(GLOB _lightrhi_modules CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/rhi/*.cppm")
target_sources(LightRHI
    PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/source"
    FILES ${_lightrhi_modules}
)

target_warnings(LightRHI)

# -------------------------------------------------------------------
# Vulkan backend  (Linux / Windows)
# -------------------------------------------------------------------
if(LIGHT_RHI_BACKEND STREQUAL "Vulkan")
    add_library(LightRHIBackend SHARED)
    add_library(LightRHI::Backend ALIAS LightRHIBackend)
    if(WIN32)
        set_target_properties(LightRHIBackend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()

    target_compile_features(LightRHIBackend PUBLIC cxx_std_23)
    target_compile_options(LightRHIBackend PRIVATE -fno-rtti)

    # A Vulkan build may expose every window-system integration whose native headers are present.
    # These target-scoped definitions build the matching backend-owned descriptors and also let a
    # consumer select one without guessing from the host environment.
    set(_light_rhi_vulkan_surface_definitions)
    if(WIN32)
        list(APPEND _light_rhi_vulkan_surface_definitions LIGHT_RHI_VULKAN_HAS_WIN32_SURFACE=1)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        include(CheckIncludeFileCXX)
        check_include_file_cxx("wayland-client.h" LIGHT_RHI_HAVE_WAYLAND_CLIENT_H)
        check_include_file_cxx("xcb/xcb.h" LIGHT_RHI_HAVE_XCB_H)
        check_include_file_cxx("X11/Xlib.h" LIGHT_RHI_HAVE_XLIB_H)

        if(LIGHT_RHI_HAVE_WAYLAND_CLIENT_H)
            list(APPEND _light_rhi_vulkan_surface_definitions LIGHT_RHI_VULKAN_HAS_WAYLAND_SURFACE=1)
        endif()
        if(LIGHT_RHI_HAVE_XCB_H)
            list(APPEND _light_rhi_vulkan_surface_definitions LIGHT_RHI_VULKAN_HAS_XCB_SURFACE=1)
        endif()
        if(LIGHT_RHI_HAVE_XLIB_H)
            list(APPEND _light_rhi_vulkan_surface_definitions LIGHT_RHI_VULKAN_HAS_XLIB_SURFACE=1)
        endif()
    endif()
    target_compile_definitions(LightRHIBackend PUBLIC ${_light_rhi_vulkan_surface_definitions})

    # Glob the backend tree: *.cppm are module interfaces, *.cpp are regular
    # TUs (GLOB_RECURSE picks up external/vma/vulkan_vma.cpp).
    file(GLOB _vk_modules CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/*.cppm")
    file(GLOB_RECURSE _vk_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/*.cpp")
    target_sources(LightRHIBackend
        PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/source"
        FILES ${_vk_modules}
        PRIVATE ${_vk_sources}
    )

    # Both private: this backend's module interface names no Vulkan type, so a consumer creates a
    # provider without ever having the Vulkan headers. The platform object reaches it as an opaque
    # rhi::NativeTextureOwner and is cast here, where the real type is known.
    target_link_libraries(LightRHIBackend
        PUBLIC  LightRHI::LightRHI
        PRIVATE volk::volk Vulkan::Headers
    )

    # LightRHIBackend.so looks for LightRHI.so in the same directory at runtime
    set_target_properties(LightRHIBackend PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )

    if(TARGET GPUOpen::VulkanMemoryAllocator)
        target_link_libraries(LightRHIBackend PRIVATE GPUOpen::VulkanMemoryAllocator)
    endif()

    target_warnings(LightRHIBackend)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(LightRHIBackend PRIVATE
            -Wno-include-angled-in-module-purview
            -Wno-nullability-extension)
        set_source_files_properties(source/backend_vulkan/external/vma/vulkan_vma.cpp PROPERTIES COMPILE_OPTIONS -w)
    endif()
endif()

# -------------------------------------------------------------------
# Metal backend  (macOS)
# -------------------------------------------------------------------
if(LIGHT_RHI_BACKEND STREQUAL "Metal")
    # CAMetalDisplayLink, which paces a surface to the display, exists only in Objective-C and
    # Swift. A small Swift library reaches it and exposes plain functions to the C++ backend.
    if(POLICY CMP0157)
        cmake_policy(SET CMP0157 NEW) # compile and link Swift as separate steps
    endif()
    enable_language(Swift)
    execute_process(
        COMMAND xcrun swiftc -print-target-info
        OUTPUT_VARIABLE _light_rhi_swift_target_info
        COMMAND_ERROR_IS_FATAL ANY)
    string(JSON _light_rhi_swift_resource_path GET "${_light_rhi_swift_target_info}" paths runtimeResourcePath)
    set(_light_rhi_swift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${_light_rhi_swift_generated_dir}")
    file(GLOB _light_rhi_swift_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_metal/*.swift")
    add_library(LightRHIMetalDisplayLink STATIC ${_light_rhi_swift_sources})
    set_target_properties(LightRHIMetalDisplayLink PROPERTIES
        Swift_MODULE_NAME LightRHIMetalDisplayLink
        Swift_MODULE_DIRECTORY "${_light_rhi_swift_generated_dir}")
    target_compile_options(LightRHIMetalDisplayLink PRIVATE
        -cxx-interoperability-mode=default
        "SHELL:-Xfrontend -clang-header-expose-decls=has-expose-attr"
        -emit-objc-header-path "${_light_rhi_swift_generated_dir}/LightRHIMetalDisplayLink-Swift.h")
    target_link_libraries(LightRHIMetalDisplayLink PRIVATE "-framework QuartzCore")

    add_library(LightRHIBackend SHARED)
    add_library(LightRHI::Backend ALIAS LightRHIBackend)
    if(WIN32)
        set_target_properties(LightRHIBackend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()

    target_compile_features(LightRHIBackend PUBLIC cxx_std_23)
    target_compile_options(LightRHIBackend PRIVATE -fno-rtti)
    # metal-cpp's NS::SharedPtr destructor calls release() on its pointer without checking it,
    # because Objective-C lets a message to nil do nothing - true of the machine, not of C++, where
    # it is a member call on a null pointer (the header marks it no_sanitize("undefined") for the
    # same reason). A moved-from or empty SharedPtr takes that path, and an optimizing build that
    # may assume the pointer is not null then treats everything after it as unreachable: every
    # failure return and every return after a move became a trap instruction. This keeps those
    # checks, which is what the interop the headers were written for actually requires.
    target_compile_options(LightRHIBackend PRIVATE -fno-delete-null-pointer-checks)

    file(GLOB _mtl_modules CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_metal/*.cppm")
    file(GLOB_RECURSE _mtl_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_metal/*.cpp")
    target_sources(LightRHIBackend
        PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/source"
        FILES ${_mtl_modules}
        PRIVATE ${_mtl_sources}
    )

    # All private: this backend's module interface names no Metal type, so a consumer creates a
    # provider without ever having metal-cpp. The layer reaches it as an opaque
    # rhi::NativeTextureOwner and is cast here, where the real type is known.
    target_link_libraries(LightRHIBackend
        PUBLIC  LightRHI::LightRHI
        PRIVATE
            metal-cpp::metal-cpp
            LightRHIMetalDisplayLink
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${QUARTZCORE_FRAMEWORK}
    )
    add_dependencies(LightRHIBackend LightRHIMetalDisplayLink)
    # The generated header, and the Swift runtime headers it includes.
    target_include_directories(LightRHIBackend SYSTEM PRIVATE
        "${_light_rhi_swift_generated_dir}" "${_light_rhi_swift_resource_path}")
    set_source_files_properties(${_mtl_sources} PROPERTIES OBJECT_DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}LightRHIMetalDisplayLink${CMAKE_STATIC_LIBRARY_SUFFIX}")

    # LightRHIBackend.dylib looks for LightRHI.dylib in the same directory at runtime
    set_target_properties(LightRHIBackend PROPERTIES
        INSTALL_RPATH "@loader_path"
    )

    # Which Metal shader libraries the build ships, decided once by light_rhi_compile_shaders' option.
    if(LIGHT_RHI_METAL_SHADER_FORMAT STREQUAL "MetalLib")
        target_compile_definitions(LightRHIBackend PRIVATE LIGHT_RHI_METAL_PRECOMPILED_SHADERS=1)
    else()
        target_compile_definitions(LightRHIBackend PRIVATE LIGHT_RHI_METAL_PRECOMPILED_SHADERS=0)
    endif()

    target_warnings(LightRHIBackend)
endif()

# -------------------------------------------------------------------
# Sanitizers
# -------------------------------------------------------------------
foreach(_san IN ITEMS asan ubsan tsan)
    string(TOUPPER ${_san} _SAN)
    if(LIGHT_RHI_ENABLE_${_SAN})
        enable_sanitizer(LightRHI ${_san})
        enable_sanitizer(LightRHIBackend ${_san})
    endif()
endforeach()

# -------------------------------------------------------------------
# Examples  (standalone only by default — see option above)
# -------------------------------------------------------------------
if(LIGHT_RHI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -------------------------------------------------------------------
# Tests  (standalone only by default)
# -------------------------------------------------------------------
if(LIGHT_RHI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(InactiveBackendCheck)
light_rhi_check_inactive_backend()
