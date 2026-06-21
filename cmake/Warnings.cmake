# cmake/Warnings.cmake
# Attaches strict compile warnings to a target.
# Usage: target_warnings(<target>)

function(target_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /WX          # warnings as errors
            /wd4100      # unreferenced formal parameter (too noisy with interfaces)
            /wd4324      # structure padded due to alignment
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wnon-virtual-dtor
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wconversion
            -Wsign-conversion
            -Wmisleading-indentation
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
            # Silence noise from third-party Vulkan headers
            -Wno-missing-field-initializers
        )
        # GCC-only warnings
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE
                -Wduplicated-cond
                -Wduplicated-branches
                -Wlogical-op
            )
        endif()
    endif()
endfunction()
