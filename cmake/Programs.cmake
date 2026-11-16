include_guard(GLOBAL)

# A program is one folder: every source in it, so adding a file to the folder never changes the build.
# `folder` is relative to the calling CMakeLists.txt.
function(light_rhi_add_program target folder)
  file(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${folder}/*.cpp")
  if(NOT sources)
    message(FATAL_ERROR "[LightRHI] ${target} has no source under ${folder}")
  endif()
  add_executable(${target} ${sources})
endfunction()
