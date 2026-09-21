# Test strategy

LightRHI separates fast public-contract checks from GPU feature checks.

| Test | Label | Responsibility |
|---|---|---|
| `rhi_unit_tests` | `smoke` | Public values, device creation, basic resources, commands, and shader artifacts |
| `rhi_placement_tests` | `placement` | Heap requirements, placed resources, alignment, overlap, and reuse |
| `rhi_presentation_tests` | `presentation` | Native surfaces, frame acquisition, resize, skipped frames, and present submission |
| `rhi_bindless_tests` | `bindless` | Shader-visible resource slots and bindless access |
| `rhi_raytracing_tests` | `raytracing` | Acceleration structures and ray-tracing dispatch on supported hardware |

Run the fast gate with:

```sh
ctest --test-dir build -L smoke --output-on-failure
```

Run the complete enabled suite with:

```sh
ctest --test-dir build --output-on-failure
```

Every GPU wait has a timeout. A feature that the device does not support may be
reported as a skip only when the test registers that skip explicitly. A missing
GPU does not provide evidence for placement, presentation, bindless, or
ray-tracing behavior.

Backend-neutral tests import the public modules and must not name Metal or
Vulkan implementation types. Backend component tests are allowed only when the
behavior cannot be observed through the public contract.
