# Developer commands

LightRHI builds standalone with CMake 3.28 or newer, Ninja, and a compiler with
C++23 module dependency scanning. Install `slangc` before building shader tests
or examples.

Configure, build, and test:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DLIGHT_RHI_BUILD_TESTS=ON \
  -DLIGHT_RHI_BUILD_EXAMPLES=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On macOS, provide the active SDK when the selected compiler does not find it
automatically:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)"
```

Build this documentation site:

```sh
python3 tools/docs/build_docs.py
```

The output is `build/docs/html/index.html`. Warnings fail the build.

When LightRHI is consumed as a submodule, its tests and examples default off.
The parent project owns those options and the build directory.
