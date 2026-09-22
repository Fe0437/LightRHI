include_guard(GLOBAL)

# Compile a shaders_registry.generated.json manifest into backend-specific
# shader artifacts using LightRHI's shared Slang pipeline.
#
# Example:
#   light_rhi_compile_shaders(
#       TARGET      my_target
#       SHADERS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders_registry.generated.json"
#       OUTPUT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders"
#       INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/source/core"
#       DEFINES MY_SHADER_FEATURE=1
#       ARTIFACTS_VAR artifacts
#       BACKEND_DIR_VAR backend_dir
#       EXTENSION_VAR extension
#       FORMAT_VAR format)
#
# INCLUDE_DIRS is optional: extra directories slangc resolves #include "..."
# against, beyond each shader's own source directory (its always-searched
# default). Needed when a .slang file #includes a header living elsewhere —
# e.g. a project's shared CPU/GPU struct-definition headers.
#
# Each .slang file becomes one library holding all of its entry points, in the backend's own format:
#   <OUTPUT_DIR>/<source>.spv        Vulkan
#   <OUTPUT_DIR>/<source>.metallib   Metal, LIGHT_RHI_METAL_SHADER_FORMAT=MetalLib (default)
#   <OUTPUT_DIR>/<source>.metal      Metal, LIGHT_RHI_METAL_SHADER_FORMAT=MslSource
# Anything else the compile goes through is kept under <OUTPUT_DIR>/intermediate/ and is not shipped.
#
# The backend reads them back with rhi::ReadShaderLibrary(directory, "<source>"): what the files are
# called and what they hold stays the backend's business, so a consumer never names either.
#
# The generated custom target is named "${TARGET}_shaders" by default, which
# collides if this function is called more than once against the same
# TARGET (e.g. one target compiling two independent shader manifests into two
# separate output directories). Pass NAME_PREFIX to name the custom target
# "${NAME_PREFIX}_shaders" instead — TARGET still receives the
# add_dependencies() wiring, only the custom target's own name changes, so no
# extra indirection (dummy INTERFACE target, manual add_dependencies) is
# needed at the call site.

# How Metal shaders ship. MetalLib compiles them ahead of time with Apple's toolchain, so the device
# loads a finished library; MslSource ships the source and the device compiles it at startup, which
# needs no Apple toolchain at build time and is kept for that case.
if(APPLE)
    execute_process(COMMAND xcrun -f metal OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _light_rhi_no_metal_toolchain)
    if(_light_rhi_no_metal_toolchain EQUAL 0)
        set(_light_rhi_metal_format_default MetalLib)
    else()
        set(_light_rhi_metal_format_default MslSource)
    endif()
    set(LIGHT_RHI_METAL_SHADER_FORMAT "${_light_rhi_metal_format_default}" CACHE STRING
        "How Metal shaders are shipped: MetalLib (precompiled) or MslSource (compiled at startup)")
    set_property(CACHE LIGHT_RHI_METAL_SHADER_FORMAT PROPERTY STRINGS MetalLib MslSource)
endif()

# The Apple SDK a .metallib is compiled for: the one this build targets.
function(_light_rhi_metal_sdk out)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
            set(${out} iphonesimulator PARENT_SCOPE)
        else()
            set(${out} iphoneos PARENT_SCOPE)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "tvOS")
        set(${out} appletvos PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
        set(${out} xros PARENT_SCOPE)
    else()
        set(${out} macosx PARENT_SCOPE)
    endif()
endfunction()

function(light_rhi_compile_shaders)
    set(_options "")
    set(_one_value_args TARGET NAME_PREFIX SHADERS_JSON OUTPUT_DIR BACKEND ARTIFACTS_VAR BACKEND_DIR_VAR EXTENSION_VAR FORMAT_VAR)
    set(_multi_value_args INCLUDE_DIRS DEFINES)
    cmake_parse_arguments(LIGHT_RHI_SHADER
        "${_options}" "${_one_value_args}" "${_multi_value_args}" ${ARGN})

    foreach(_required TARGET SHADERS_JSON OUTPUT_DIR)
        if(NOT LIGHT_RHI_SHADER_${_required})
            message(FATAL_ERROR "light_rhi_compile_shaders requires ${_required}")
        endif()
    endforeach()

    if(NOT TARGET "${LIGHT_RHI_SHADER_TARGET}")
        message(FATAL_ERROR "light_rhi_compile_shaders TARGET does not exist: ${LIGHT_RHI_SHADER_TARGET}")
    endif()

    if(NOT LIGHT_RHI_SLANG_AVAILABLE)
        message(FATAL_ERROR "[LightRHI] light_rhi_compile_shaders requires slangc — set SLANG_DIR")
    endif()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    get_filename_component(_manifest_dir "${LIGHT_RHI_SHADER_SHADERS_JSON}" DIRECTORY)
    get_filename_component(_light_rhi_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)

    # The manifest is generated, never authored: tools/generate_shader_registry.py
    # scans the .slang sources for their [shader(...)] entry points. The glob is
    # CONFIGURE_DEPENDS so adding or deleting a shader re-runs configure, which
    # regenerates the manifest before anything reads it below.
    file(GLOB _shader_sources CONFIGURE_DEPENDS "${_manifest_dir}/*.slang")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
                "${_light_rhi_root}/tools/generate_shader_registry.py"
                --shaders-dir "${_manifest_dir}"
                --output      "${LIGHT_RHI_SHADER_SHADERS_JSON}"
        RESULT_VARIABLE _generate_result
        OUTPUT_VARIABLE _generate_output
        ERROR_VARIABLE  _generate_error)
    if(NOT _generate_result EQUAL 0)
        message(FATAL_ERROR "[LightRHI] failed to generate ${LIGHT_RHI_SHADER_SHADERS_JSON}:\n${_generate_error}")
    endif()
    if(_generate_output)
        message(STATUS "${_generate_output}")
    endif()

    if(LIGHT_RHI_SHADER_BACKEND)
        set(_backend "${LIGHT_RHI_SHADER_BACKEND}")
    elseif(LIGHT_RHI_BACKEND STREQUAL "Metal")
        set(_backend metal)
    else()
        set(_backend vulkan)
    endif()

    set(_metal_sdk_args "")
    if(_backend STREQUAL "metal" AND LIGHT_RHI_METAL_SHADER_FORMAT STREQUAL "MetalLib")
        set(_target metal)
        set(_extension ".metallib")
        set(_format "rhi::ShaderFormat::MetalLib")
        _light_rhi_metal_sdk(_metal_sdk)
        set(_metal_sdk_args --metal-sdk "${_metal_sdk}")
    elseif(_backend STREQUAL "metal")
        set(_target metal)
        set(_extension ".metal")
        set(_format "rhi::ShaderFormat::MslSource")
    elseif(_backend STREQUAL "vulkan")
        set(_target spirv)
        set(_extension ".spv")
        set(_format "rhi::ShaderFormat::Spirv")
    else()
        message(FATAL_ERROR "light_rhi_compile_shaders unsupported backend: ${_backend}")
    endif()

    set(_compile_shaders_py "${_light_rhi_root}/tools/compile_shaders.py")

    # LightRHI's own debug helper is available to every Slang compilation.
    # Consumers can include "shader_debug.slangh" without knowing where the
    # submodule lives.
    list(APPEND LIGHT_RHI_SHADER_INCLUDE_DIRS "${_light_rhi_root}/shaders")

    set(_define_args "")
    if(DEBUG_ENABLED)
        list(APPEND _define_args --define "DEBUG_ENABLED=1")
    else()
        list(APPEND _define_args --define "DEBUG_ENABLED=0")
    endif()
    foreach(_define IN LISTS LIGHT_RHI_SHADER_DEFINES)
        list(APPEND _define_args --define "${_define}")
    endforeach()

    # Turn each INCLUDE_DIRS entry into a --include-dir slangc flag, and glob
    # its headers into DEPENDS too — otherwise editing a header a shader
    # #includes from outside its own directory (e.g. a shared CPU/GPU struct
    # header) silently leaves stale, pre-edit shader binaries in place: slangc
    # has no dependency file of its own for CMake to read, so without this,
    # CMake has no way to know the shader needs recompiling at all.
    set(_include_dir_args "")
    set(_include_dir_headers "")
    foreach(_include_dir IN LISTS LIGHT_RHI_SHADER_INCLUDE_DIRS)
        list(APPEND _include_dir_args --include-dir "${_include_dir}")
        file(GLOB _this_include_dir_headers CONFIGURE_DEPENDS "${_include_dir}/*.h" "${_include_dir}/*.slangh")
        list(APPEND _include_dir_headers ${_this_include_dir_headers})
    endforeach()

    # A .slang file's own directory (_manifest_dir) is always on slangc's
    # search path implicitly — no explicit --include-dir needed for it — but
    # its *.slangh siblings (e.g. path_trace_core.slangh, environment.slangh)
    # are just as capable of being silently-stale-on-edit as an
    # INCLUDE_DIRS header, so they need the same DEPENDS treatment.
    file(GLOB _manifest_dir_headers CONFIGURE_DEPENDS "${_manifest_dir}/*.slangh")
    list(APPEND _include_dir_headers ${_manifest_dir_headers})

    # One library per .slang file: the manifest lists entry points, so collect their sources.
    file(READ "${LIGHT_RHI_SHADER_SHADERS_JSON}" _manifest_content)
    string(JSON _shader_count LENGTH "${_manifest_content}" shaders)
    set(_sources "")
    if(_shader_count GREATER 0)
        math(EXPR _shader_last "${_shader_count} - 1")
        foreach(_i RANGE ${_shader_last})
            string(JSON _shader_source GET "${_manifest_content}" shaders ${_i} source)
            list(APPEND _sources "${_shader_source}")
        endforeach()
        list(REMOVE_DUPLICATES _sources)
    endif()

    set(_artifacts "")
    foreach(_shader_source IN LISTS _sources)
        get_filename_component(_library "${_shader_source}" NAME_WE)
        set(_shader_out "${LIGHT_RHI_SHADER_OUTPUT_DIR}/${_library}${_extension}")
        add_custom_command(
            OUTPUT "${_shader_out}"
            COMMAND "${Python3_EXECUTABLE}"
                    "${_compile_shaders_py}"
                    --shaders-json "${LIGHT_RHI_SHADER_SHADERS_JSON}"
                    --output-dir   "${LIGHT_RHI_SHADER_OUTPUT_DIR}"
                    --slangc       "${SLANGC}"
                    --backend      "${_backend}"
                    --only-source  "${_shader_source}"
                    ${_include_dir_args}
                    ${_define_args}
                    ${_metal_sdk_args}
            DEPENDS "${LIGHT_RHI_SHADER_SHADERS_JSON}" "${_manifest_dir}/${_shader_source}" ${_include_dir_headers}
                    "${_compile_shaders_py}"
            COMMENT "Slang -> ${_backend}: ${_library}${_extension}"
            VERBATIM
        )
        list(APPEND _artifacts "${_shader_out}")
    endforeach()

    if(LIGHT_RHI_SHADER_NAME_PREFIX)
        set(_shader_target "${LIGHT_RHI_SHADER_NAME_PREFIX}_shaders")
    else()
        set(_shader_target "${LIGHT_RHI_SHADER_TARGET}_shaders")
    endif()
    add_custom_target("${_shader_target}" DEPENDS ${_artifacts})
    add_dependencies("${LIGHT_RHI_SHADER_TARGET}" "${_shader_target}")
    # Which target builds each library, so light_rhi_ship_shaders can wait for it from any directory.
    foreach(_artifact IN LISTS _artifacts)
        set_property(GLOBAL PROPERTY "_LIGHT_RHI_BUILT_BY:${_artifact}" "${_shader_target}")
    endforeach()

    if(LIGHT_RHI_SHADER_ARTIFACTS_VAR)
        set("${LIGHT_RHI_SHADER_ARTIFACTS_VAR}" "${_artifacts}" PARENT_SCOPE)
    endif()
    if(LIGHT_RHI_SHADER_BACKEND_DIR_VAR)
        set("${LIGHT_RHI_SHADER_BACKEND_DIR_VAR}" "${_backend}" PARENT_SCOPE)
    endif()
    if(LIGHT_RHI_SHADER_EXTENSION_VAR)
        set("${LIGHT_RHI_SHADER_EXTENSION_VAR}" "${_extension}" PARENT_SCOPE)
    endif()
    if(LIGHT_RHI_SHADER_FORMAT_VAR)
        set("${LIGHT_RHI_SHADER_FORMAT_VAR}" "${_format}" PARENT_SCOPE)
    endif()
endfunction()

# Put compiled shader libraries where an application finds them at run time.
#
#   light_rhi_ship_shaders(
#       TARGET    my_app                        # an executable
#       ARTIFACTS ${artifacts}                  # from light_rhi_compile_shaders(ARTIFACTS_VAR ...)
#       [DIRECTORY shaders])                    # the folder name, "shaders" by default
#
# The libraries are copied into DIRECTORY beside the executable, or into the bundle's resources
# for an Apple app bundle (Contents/Resources on macOS, the bundle root on iOS) - where SDL's
# SDL_GetBasePath(), NSBundle's resourcePath and a plain executable-relative path all look. The
# copy is refreshed on every build of TARGET, so an edited shader never runs stale. Install rules
# stay with the application, which knows its own layout.
function(light_rhi_ship_shaders)
    cmake_parse_arguments(SHIP "" "TARGET;DIRECTORY" "ARTIFACTS" ${ARGN})
    foreach(_required TARGET ARTIFACTS)
        if(NOT SHIP_${_required})
            message(FATAL_ERROR "light_rhi_ship_shaders requires ${_required}")
        endif()
    endforeach()
    if(NOT SHIP_DIRECTORY)
        set(SHIP_DIRECTORY shaders)
    endif()

    get_target_property(_bundle "${SHIP_TARGET}" MACOSX_BUNDLE)
    if(_bundle AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(_destination "$<TARGET_BUNDLE_CONTENT_DIR:${SHIP_TARGET}>/Resources/${SHIP_DIRECTORY}")
    elseif(_bundle)
        set(_destination "$<TARGET_BUNDLE_DIR:${SHIP_TARGET}>/${SHIP_DIRECTORY}")
    else()
        set(_destination "$<TARGET_FILE_DIR:${SHIP_TARGET}>/${SHIP_DIRECTORY}")
    endif()

    set(_copy_target "${SHIP_TARGET}_ship_shaders")
    add_custom_target("${_copy_target}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_destination}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${SHIP_ARTIFACTS} "${_destination}"
        COMMENT "Ship shader libraries with ${SHIP_TARGET}"
        VERBATIM)
    foreach(_artifact IN LISTS SHIP_ARTIFACTS)
        get_property(_built_by GLOBAL PROPERTY "_LIGHT_RHI_BUILT_BY:${_artifact}")
        if(NOT _built_by)
            message(FATAL_ERROR "light_rhi_ship_shaders: ${_artifact} was not compiled by light_rhi_compile_shaders")
        endif()
        add_dependencies("${_copy_target}" "${_built_by}")
    endforeach()
    add_dependencies("${SHIP_TARGET}" "${_copy_target}")
endfunction()
