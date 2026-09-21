# Sanitizer support

Standalone options enable sanitizers on LightRHI targets:

| Option | Sanitizer |
|---|---|
| `LIGHT_RHI_ENABLE_ASAN=ON` | AddressSanitizer |
| `LIGHT_RHI_ENABLE_UBSAN=ON` | UndefinedBehaviorSanitizer |
| `LIGHT_RHI_ENABLE_TSAN=ON` | ThreadSanitizer |

Example:

```sh
cmake -S . -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLIGHT_RHI_BUILD_TESTS=ON \
  -DLIGHT_RHI_ENABLE_ASAN=ON \
  -DLIGHT_RHI_ENABLE_UBSAN=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

Do not combine ThreadSanitizer with AddressSanitizer. Use a separate build
directory for each sanitizer configuration.

Sanitizers observe host memory, undefined behavior, and host thread races. They
do not validate GPU memory access or synchronization. Use Metal validation,
Vulkan validation layers, and the focused GPU tests for those failures.
