function(add_rhi_example name)
    add_executable(${name} ${name}/main.cpp)
    target_compile_features(${name} PRIVATE cxx_std_23)
    target_link_libraries(${name} PRIVATE LightRHI::LightRHI LightRHI::Backend)
    target_warnings(${name})

    # If the example ships Slang shaders, compile them to this backend's format
    # (Slang -> SPIR-V / MSL) and point the binary at the generated artifacts.
    set(_shader_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/${name}/shaders")
    file(GLOB _shader_sources CONFIGURE_DEPENDS "${_shader_source_dir}/*.slang")
    if(_shader_sources)
        set(_shaders_json "${_shader_source_dir}/shaders_registry.generated.json")
        set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders/${name}")
        light_rhi_compile_shaders(
            TARGET       ${name}
            SHADERS_JSON "${_shaders_json}"
            OUTPUT_DIR   "${_shader_dir}")

        target_compile_definitions(${name} PRIVATE RHI_EXAMPLE_SHADER_DIR="${_shader_dir}")
        if(LIGHT_RHI_BACKEND STREQUAL "Metal")
            target_compile_definitions(${name} PRIVATE RHI_BACKEND_METAL=1)
        else()
            target_compile_definitions(${name} PRIVATE RHI_BACKEND_VULKAN=1)
        endif()
    endif()
endfunction()

add_rhi_example(00_device_info)
add_rhi_example(01_clear_texture)
add_rhi_example(02_compute_triangle)
