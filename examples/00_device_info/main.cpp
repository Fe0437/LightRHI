/**
 * {file} main.cpp
 * {brief} Creates a device and prints its adapter and bindless limits.
 */

// Standard headers before module imports (required by Homebrew LLVM / libc++).
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

import lightRHI;

#include "../example_required.h"

int main()
{
    auto opened{rhi::CreateDevice({
        .EnableValidation    = true,
        .EnableGpuValidation = false,
        .AppName             = "00_device_info",
    })};
    if (!opened)
    {
        std::fprintf(stderr, "no device (reason %d)\n", static_cast<int>(opened.error()));
        return EXIT_FAILURE;
    }
    auto &device{*opened};
    std::printf("Adapter : %s\n", std::string{device->AdapterName()}.c_str());
    std::printf("VRAM    : %.1f MB\n", static_cast<double>(device->VideoMemoryBytes()) / (1024.0 * 1024.0));

    auto &heap{device->BindlessHeap()};
    std::printf("Bindless: buffers=%u  textures=%u  samplers=%u\n", heap.MaxBuffers(), heap.MaxTextures(),
                heap.MaxSamplers());

    device->WaitIdle();
    std::printf("OK\n");
    return EXIT_SUCCESS;
}
