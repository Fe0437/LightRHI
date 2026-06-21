# cmake/Slang.cmake — find or fetch a working slangc.
#
# slangc is the Slang shader compiler (https://github.com/shader-slang/slang),
# invoked by tools/compile_shaders.py to produce external shader artifacts
# (.spv / .metal). CMake itself no longer wraps shader output as C++ headers.
#
# Resolution order (first working one wins):
#   1. SLANGC       — path to the compiler executable directly
#   2. SLANG_DIR     — a Slang install prefix (checks SLANG_DIR/bin/slangc)
#   3. PATH          — every directory on PATH, not just the first match
#   4. FetchContent  — a prebuilt release downloaded from GitHub, same as
#                      LightRHI's other dependencies (see Dependencies.cmake)
#                      that aren't found on the system. This makes a working
#                      Slang toolchain available out of the box to anyone who
#                      pulls this repo, with no manual install step required.
#
# Candidates from (1)-(3) are verified by actually invoking them (`slangc
# -v`), not just checked for existence — a broken or partially-installed
# slangc (e.g. missing a co-located shared library) can shadow a working one
# elsewhere on PATH, and find_program alone can't tell the difference.

include(FetchContent)

function(_light_rhi_try_slangc candidate)
    if(SLANGC OR NOT candidate OR NOT EXISTS "${candidate}")
        return()
    endif()
    execute_process(COMMAND "${candidate}" -v
        OUTPUT_VARIABLE _out ERROR_VARIABLE _out
        RESULT_VARIABLE _result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_result EQUAL 0)
        set(SLANGC "${candidate}" CACHE FILEPATH "Path to the Slang shader compiler" FORCE)
        set(_light_rhi_slangc_version "${_out}" CACHE INTERNAL "")
    endif()
endfunction()

# Re-verify on every configure instead of trusting an already-cached SLANGC
# blindly: the whole point of this file is that a path existing isn't proof
# it still works (an install can go stale between configures).
unset(SLANGC CACHE)

set(_slangc_candidates "")
if(DEFINED ENV{SLANGC})
    list(APPEND _slangc_candidates "$ENV{SLANGC}")
endif()
if(DEFINED ENV{SLANG_DIR})
    list(APPEND _slangc_candidates "$ENV{SLANG_DIR}/bin/slangc")
endif()
if(DEFINED ENV{PATH})
    file(TO_CMAKE_PATH "$ENV{PATH}" _path_dirs)
    foreach(_dir IN LISTS _path_dirs)
        find_program(_light_rhi_slangc_in_dir NAMES slangc PATHS "${_dir}" NO_DEFAULT_PATH NO_CACHE)
        if(_light_rhi_slangc_in_dir)
            list(APPEND _slangc_candidates "${_light_rhi_slangc_in_dir}")
        endif()
    endforeach()
endif()

list(REMOVE_DUPLICATES _slangc_candidates)
foreach(_candidate IN LISTS _slangc_candidates)
    _light_rhi_try_slangc("${_candidate}")
endforeach()

# ----------------------------------------------------------------------------
# Fall back to a prebuilt release when nothing on the system works.
# ----------------------------------------------------------------------------
if(NOT SLANGC)
    set(_light_rhi_slang_version "2026.13")

    if(APPLE)
        set(_light_rhi_slang_os "macos")
    elseif(WIN32)
        set(_light_rhi_slang_os "windows")
    elseif(UNIX)
        set(_light_rhi_slang_os "linux")
    else()
        set(_light_rhi_slang_os "")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
        set(_light_rhi_slang_arch "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_light_rhi_slang_arch "x86_64")
    else()
        set(_light_rhi_slang_arch "")
    endif()

    # SHA256 of each supported release asset, pinned alongside the version
    # above so a version bump can't silently start fetching an unverified
    # archive — bump both together and refresh these from the new release.
    set(_light_rhi_slang_hash_linux-aarch64   "3b951f7180b4234ee248e8dc9e7724b127533eec861e40728a54d968e2cc1adb")
    set(_light_rhi_slang_hash_linux-x86_64    "3563957e4a0d4cfed6d6b92ea088103e93533f02181018fcfbe56d7b831ff619")
    set(_light_rhi_slang_hash_macos-aarch64   "85a70992c978b17ae95660377afe00ff1dd9b9f9302ca19ac8190165f8dab77b")
    set(_light_rhi_slang_hash_macos-x86_64    "38ab5df7545c3eaddefc9983e5cfa27ebd49df4d6b82f2093686bb0c2888252b")
    set(_light_rhi_slang_hash_windows-aarch64 "7b5b32e1aacc308dfd42bf4e3b379de52f2ee67a92866f895084d3bc5146146a")
    set(_light_rhi_slang_hash_windows-x86_64  "6bae110faa3b51ea00316e0a1dd4f40b11eb196a2e3a135074300bf2d5d4c777")

    if(_light_rhi_slang_os AND _light_rhi_slang_arch AND
       DEFINED _light_rhi_slang_hash_${_light_rhi_slang_os}-${_light_rhi_slang_arch})
        set(_light_rhi_slang_asset "${_light_rhi_slang_os}-${_light_rhi_slang_arch}")
        if(EXISTS "${CMAKE_BINARY_DIR}/_deps/slang_binaries-src")
            message(STATUS "[LightRHI] slangc not found or not runnable on this system — "
                           "reusing cached Slang ${_light_rhi_slang_version} (${_light_rhi_slang_asset}) fetch")
        else()
            message(STATUS "[LightRHI] slangc not found or not runnable on this system — "
                           "fetching Slang ${_light_rhi_slang_version} (${_light_rhi_slang_asset}) via FetchContent")
        endif()

        FetchContent_Declare(slang_binaries
            URL      "https://github.com/shader-slang/slang/releases/download/v${_light_rhi_slang_version}/slang-${_light_rhi_slang_version}-${_light_rhi_slang_asset}.zip"
            URL_HASH "SHA256=${_light_rhi_slang_hash_${_light_rhi_slang_asset}}"
        )
        FetchContent_MakeAvailable(slang_binaries)

        if(WIN32)
            set(_light_rhi_fetched_slangc "${slang_binaries_SOURCE_DIR}/bin/slangc.exe")
        else()
            set(_light_rhi_fetched_slangc "${slang_binaries_SOURCE_DIR}/bin/slangc")
        endif()
        if(NOT WIN32)
            file(CHMOD "${_light_rhi_fetched_slangc}" PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)
        endif()

        _light_rhi_try_slangc("${_light_rhi_fetched_slangc}")
    else()
        message(STATUS "[LightRHI] no prebuilt Slang release available for "
                       "os='${_light_rhi_slang_os}' arch='${CMAKE_SYSTEM_PROCESSOR}' — cannot fetch automatically")
    endif()
endif()

if(NOT SLANGC)
    message(FATAL_ERROR "[LightRHI] slangc not found, not runnable, and could not be fetched automatically. "
                        "Put a working slangc on PATH, set SLANG_DIR, or set SLANGC.")
else()
    message(STATUS "[LightRHI] slangc ${_light_rhi_slangc_version}: ${SLANGC}")
    set(LIGHT_RHI_SLANG_AVAILABLE ON CACHE BOOL "slangc was found during configure" FORCE)
endif()
