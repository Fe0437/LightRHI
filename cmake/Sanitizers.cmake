# cmake/Sanitizers.cmake
# Usage:
#   include(Sanitizers)
#   enable_sanitizer(<target> asan|ubsan|tsan)
# Multiple sanitizers can be stacked (asan + ubsan work together; tsan is exclusive).

function(enable_sanitizer target sanitizer)
    if(MSVC)
        if(sanitizer STREQUAL "asan")
            target_compile_options(${target} PRIVATE /fsanitize=address)
        else()
            message(WARNING "[LightRHI] Sanitizer '${sanitizer}' not supported on MSVC — skipping")
        endif()
        return()
    endif()

    set(_flags "")

    if(sanitizer STREQUAL "asan")
        list(APPEND _flags -fsanitize=address -fno-omit-frame-pointer)
    elseif(sanitizer STREQUAL "ubsan")
        list(APPEND _flags -fsanitize=undefined -fno-omit-frame-pointer)
    elseif(sanitizer STREQUAL "tsan")
        list(APPEND _flags -fsanitize=thread)
    else()
        message(FATAL_ERROR "[LightRHI] Unknown sanitizer: ${sanitizer}")
    endif()

    target_compile_options(${target} PRIVATE ${_flags})
    # PUBLIC so the sanitizer runtime is also linked into any consumer (e.g. the test executable).
    target_link_options(${target}    PUBLIC  ${_flags})
endfunction()
