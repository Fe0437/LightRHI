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
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHI>;PATH=path_list_prepend:$<TARGET_FILE_DIR:LightRHIBackend>")
