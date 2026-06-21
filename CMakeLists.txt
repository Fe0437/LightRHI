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
target_compile_options(LightRHI PUBLIC -fno-rtti)
target_compile_definitions(LightRHI PUBLIC
    $<$<BOOL:${DEBUG_ENABLED}>:DEBUG_ENABLED=1>
    $<$<NOT:$<BOOL:${DEBUG_ENABLED}>>:DEBUG_ENABLED=0>)
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
    add_library(LightRHIBackend SHARED)
    add_library(LightRHI::Backend ALIAS LightRHIBackend)
    if(WIN32)
        set_target_properties(LightRHIBackend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()

    target_compile_features(LightRHIBackend PUBLIC cxx_std_23)
    target_compile_options(LightRHIBackend PRIVATE -fno-rtti)

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

    target_link_libraries(LightRHIBackend
        PUBLIC  LightRHI::LightRHI
        PRIVATE
            metal-cpp::metal-cpp
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${QUARTZCORE_FRAMEWORK}
    )

    # LightRHIBackend.dylib looks for LightRHI.dylib in the same directory at runtime
    set_target_properties(LightRHIBackend PROPERTIES
        INSTALL_RPATH "@loader_path"
    )

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
