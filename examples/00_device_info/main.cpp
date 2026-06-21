// 00_device_info — create a device, print adapter info, destroy.
// The simplest possible LightRHI program.

// Standard headers before module imports (required by Homebrew LLVM / libc++).
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

import lightRHI;

int main()
{
    try
    {
        auto device{rhi::CreateDevice({
            .EnableValidation    = true,
            .EnableGpuValidation = false,
            .AppName             = "00_device_info",
        })};

        std::printf("Adapter : %s\n", std::string{device->AdapterName()}.c_str());
        std::printf("VRAM    : %.1f MB\n", static_cast<double>(device->VideoMemoryBytes()) / (1024.0 * 1024.0));

        auto &heap = device->BindlessHeap();
        std::printf("Bindless: buffers=%u  textures=%u  samplers=%u\n", heap.MaxBuffers(), heap.MaxTextures(),
                    heap.MaxSamplers());

        device->WaitIdle();
        std::printf("OK\n");
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
