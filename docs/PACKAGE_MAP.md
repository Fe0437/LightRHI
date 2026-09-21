# Package map

| Package or target | Path | Responsibility |
|---|---|---|
| `LightRHI::LightRHI` · module `rhi` | `source/rhi` | Backend-neutral handles, descriptors, interfaces, errors, and synchronization contracts |
| `LightRHI::Backend` · module `lightRHI` | `source/backend_metal` or `source/backend_vulkan` | Platform device factory and implementation of the public interfaces |
| Metal backend | `source/backend_metal` | Metal 4 resource, command, pipeline, synchronization, and presentation implementation |
| Vulkan backend | `source/backend_vulkan` | Vulkan resource, command, pipeline, synchronization, and presentation implementation |
| Shader support | `shaders`, `cmake/Shaders.cmake`, `tools/compile_shaders.py` | Shared Slang declarations and backend-specific artifact compilation |
| Examples | `examples` | Small consumer programs that use only public targets and modules |
| Tests | `tests` | API, resource, placement, presentation, bindless, and ray-tracing behavior |
| Documentation tooling | `tools/docs` | Public API extraction and Sphinx site generation |

The public `rhi` module must not import backend packages or platform types.
Backend code depends on the public contract. The selected backend's `lightRHI`
module owns its typed native surface descriptors and device factory overloads.

Only one runtime backend is selected for a target platform. Development builds
may compile-check the inactive backend, but they do not expose two runtime
factories.
