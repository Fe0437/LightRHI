include_guard(GLOBAL)

# Compile the backend this platform does not build, as part of building the one it does.
#
# One backend is selected per platform, so a change to the other one is not compiled where it was
# written and only fails somewhere else. This makes that impossible: every development build also
# syntax-checks the inactive backend's translation units against the same `rhi` module, with the
# same standard, dialect, warnings and switches the real build uses. It is incremental - a stamp per unit,
# rebuilt only when that unit or the module changes - and it is off for shipping builds, which
# compile only what they ship.
#
# Today this means checking Vulkan while building Metal. The reverse is not possible: Metal's
# headers need Apple's SDK and Objective-C runtime, which a Linux or Windows machine does not have.
function(light_rhi_check_inactive_backend)
    if(NOT LIGHT_RHI_CHECK_INACTIVE_BACKEND)
        return()
    endif()
    if(NOT LIGHT_RHI_BACKEND STREQUAL "Metal")
        return() # the inactive backend here is Metal, which only Apple's SDK can compile
    endif()

    set(_check_dir "${CMAKE_CURRENT_BINARY_DIR}/inactive-backend-check")
    file(MAKE_DIRECTORY "${_check_dir}")

    # Headers only: nothing is linked, so the Vulkan loader and its libraries are not needed.
    include(FetchContent)
    # SOURCE_SUBDIR points at no CMake project because this check needs only headers. This keeps
    # FetchContent provider-aware without configuring or adding the dependencies' build targets.
    FetchContent_Declare(VulkanHeaders
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG        vulkan-sdk-1.3.290.0
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  light_rhi_headers_only)
    FetchContent_Declare(volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG        1.3.270
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  light_rhi_headers_only)
    FetchContent_Declare(VulkanMemoryAllocator
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        v3.1.0
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  light_rhi_headers_only)
    FetchContent_MakeAvailable(VulkanHeaders volk VulkanMemoryAllocator)

    # The `rhi` module is backend-neutral, so the build's own module files are what the inactive
    # backend is checked against. A unit's module name is what it declares rather than what its file
    # is called, and the build names each module file after the module it holds.
    set(_module_flags "")
    file(GLOB _rhi_units "${CMAKE_CURRENT_SOURCE_DIR}/source/rhi/*.cppm")
    foreach(_unit IN LISTS _rhi_units)
        file(READ "${_unit}" _source)
        string(REGEX MATCH "export module rhi(:[A-Za-z_0-9]+)?;" _declaration "${_source}")
        if(NOT _declaration)
            message(FATAL_ERROR "[LightRHI] ${_unit} declares no rhi module")
        endif()
        string(REGEX REPLACE "export module rhi:?([A-Za-z_0-9]*);" "\\1" _partition "${_declaration}")
        if(_partition STREQUAL "")
            list(APPEND _module_flags "-fmodule-file=rhi=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LightRHI.dir/rhi.pcm")
        else()
            list(APPEND _module_flags
                "-fmodule-file=rhi:${_partition}=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LightRHI.dir/rhi-${_partition}.pcm")
        endif()
    endforeach()

    set(_flags
        -std=c++23 -fno-rtti -fno-exceptions
        -Wall -Wextra -Wpedantic -Werror -Wshadow -Wnon-virtual-dtor -Wcast-align -Wunused
        -Woverloaded-virtual -Wconversion -Wsign-conversion -Wmisleading-indentation
        -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wno-missing-field-initializers
        -isystem "${vulkanheaders_SOURCE_DIR}/include"
        -isystem "${volk_SOURCE_DIR}"
        -isystem "${vulkanmemoryallocator_SOURCE_DIR}/include"
        -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        # The module was built with these, and parts of it exist only under them.
        "-DDEBUG_ENABLED=$<BOOL:${DEBUG_ENABLED}>"
        "-DMETRICS_ENABLED=$<BOOL:${METRICS_ENABLED}>"
        ${_module_flags})
    if(APPLE AND CMAKE_OSX_SYSROOT)
        list(APPEND _flags -isysroot "${CMAKE_OSX_SYSROOT}")
    endif()

    # The inactive backend's module interface, precompiled so its implementation units can import it.
    set(_interface_pcm "${_check_dir}/lightRHI.pcm")
    add_custom_command(
        OUTPUT "${_interface_pcm}"
        COMMAND "${CMAKE_CXX_COMPILER}" ${_flags} -x c++-module --precompile
                "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/vulkan_backend.cppm"
                -o "${_interface_pcm}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/vulkan_backend.cppm" LightRHI
        COMMENT "Checking the inactive backend: vulkan_backend.cppm"
        VERBATIM)

    set(_stamps "")
    file(GLOB _vulkan_units "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/*.cpp")
    list(FILTER _vulkan_units EXCLUDE REGEX "/external/")
    foreach(_unit IN LISTS _vulkan_units)
        get_filename_component(_name "${_unit}" NAME)
        set(_stamp "${_check_dir}/${_name}.checked")
        add_custom_command(
            OUTPUT "${_stamp}"
            COMMAND "${CMAKE_CXX_COMPILER}" ${_flags} "-fmodule-file=lightRHI=${_interface_pcm}"
                    -fsyntax-only "${_unit}"
            COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
            DEPENDS "${_unit}" "${_interface_pcm}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/vulkan_internal.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/source/backend_vulkan/vulkan_platform.h"
            COMMENT "Checking the inactive backend: ${_name}"
            VERBATIM)
        list(APPEND _stamps "${_stamp}")
    endforeach()

    # Attached to the backend rather than to ALL: a project that adds this one with EXCLUDE_FROM_ALL
    # still gets the check whenever it builds the backend, which is the moment it matters.
    add_custom_target(LightRHIInactiveBackendCheck ALL DEPENDS ${_stamps})
    add_dependencies(LightRHIBackend LightRHIInactiveBackendCheck)
endfunction()
