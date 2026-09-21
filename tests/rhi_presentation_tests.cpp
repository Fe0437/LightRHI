// rhi_presentation_tests.cpp — surface targets and presentation.
//
// A surface hands out the texture to draw this frame into, shows it without any readback, and
// follows a window that changes size. A device presents to as many surfaces as an application
// makes targets for, and knows nothing about any of them.
//
// The surface is supplied by this harness, never created by LightRHI. Which surface that is, is the
// one fact that differs per backend, so the build writes it into test_texture_source.generated.h and it is
// named nowhere below. Both are surfaces with no window behind them, so these cases run on a
// machine with no desktop session.

// Standard headers must precede module imports to avoid include-guard
// isolation issues (__promote_t redefinition) with LLVM libc++ and C++23 modules.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <memory>
#include <new>
#include <string>


import lightRHI;

#include "test_texture_owner.generated.h"

#include "device_result.h"

#define REQUIRE(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "FAIL: %s  [%s:%d]\n", #expr, __FILE__, __LINE__);                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (false)

#define TEST(name) void test_##name()

namespace
{
    /**
     * {brief} Counts heap allocations, separating LightRHI's own from the platform's.
     *
     * An allocation is attributed to the nearest real frame above `operator new`. Waiting for the
     * next frame takes LightRHI down into the platform's own presentation stack, which allocates
     * inside itself; those belong to the platform and no library above it can avoid them.
     */
    std::size_t gLightRhiAllocations{0};
    std::size_t gPlatformAllocations{0};
    bool        gCounting{false};

    [[nodiscard]] bool AllocationIsLightRhis()
    {
        // This executable's own frames are the counting machinery itself, and libc++/libsystem
        // frames are the allocator. The first frame outside all of them is the real caller.
        Dl_info     self{};
        const void *selfImage =
            dladdr(reinterpret_cast<const void *>(&AllocationIsLightRhis), &self) != 0 ? self.dli_fbase : nullptr;

        void     *frames[16];
        const int count{backtrace(frames, 16)};
        for (int i{1}; i < count; ++i)
        {
            Dl_info info{};
            if (dladdr(frames[i], &info) == 0 || info.dli_fname == nullptr)
            {
                continue;
            }
            if (info.dli_fbase == selfImage)
            {
                continue;
            }
            const char *slash = std::strrchr(info.dli_fname, '/');
            const char *image = slash != nullptr ? slash + 1 : info.dli_fname;
            if (std::strncmp(image, "libc++", 6) == 0 || std::strncmp(image, "libsystem", 9) == 0)
            {
                continue;
            }
            return std::strstr(image, "LightRHI") != nullptr;
        }
        return false;
    }
} // namespace

void *operator new(std::size_t size)
{
    if (gCounting)
    {
        gCounting = false; // the attribution walk must not re-enter this counter
        if (AllocationIsLightRhis())
        {
            ++gLightRhiAllocations;
        }
        else
        {
            ++gPlatformAllocations;
        }
        gCounting = true;
    }
    if (void *memory = std::malloc(size == 0 ? 1 : size))
    {
        return memory;
    }
    // This replacement stands in for the library's allocator, which no longer unwinds either: a
    // test that cannot get memory has nothing left to measure.
    std::fprintf(stderr, "FAIL: out of memory\n");
    std::exit(1);
}

void operator delete(void *memory) noexcept
{
    std::free(memory);
}

void operator delete(void *memory, std::size_t) noexcept
{
    std::free(memory);
}

namespace
{
    /** {brief} Names the build a measurement was taken in, so its numbers can be compared. */
#if defined(NDEBUG)
    constexpr const char *kBuildConfiguration{"optimized"};
#else
    constexpr const char *kBuildConfiguration{"debug"};
#endif

    constexpr uint32_t kWidth{256};
    constexpr uint32_t kHeight{192};

    /**
     * {brief} What the textures this harness presents come from, released with it.
     *
     * What it is made of is the build's to decide; this owns it and hands it over on request.
     */
    class HarnessTextureOwner
    {
      public:
        HarnessTextureOwner(rhi::IDevice &device, uint32_t width, uint32_t height)
            : _owner{rhitest::MakeTestTextureOwner(device, width, height)}
        {
        }

        HarnessTextureOwner(const HarnessTextureOwner &)            = delete;
        HarnessTextureOwner(HarnessTextureOwner &&)                 = delete;
        HarnessTextureOwner &operator=(const HarnessTextureOwner &) = delete;
        HarnessTextureOwner &operator=(HarnessTextureOwner &&)      = delete;

        ~HarnessTextureOwner() { rhitest::ReleaseTestTextureOwner(_owner); }

        /** {brief} Resizes what the textures come from, or says this one cannot be driven. */
        [[nodiscard]] bool Resize(uint32_t width, uint32_t height)
        {
            return rhitest::ResizeTestTextureOwner(_owner, width, height);
        }

        [[nodiscard]] rhitest::TestTextureOwner Get() const noexcept
        {
            return _owner;
        }

      private:
        rhitest::TestTextureOwner _owner{};
    };

    /** {brief} Clears this frame's texture and shows it. No readback anywhere in this path. */
    void DrawAndPresent(rhi::IDevice &device, rhi::IExternalTextureProvider &provider, rhi::TextureHandle texture)
    {
        // The size of the frame, asked of the frame: the window may have changed since this image
        // was made, and this image is what is being drawn into.
        const rhi::TextureInfo info{device.GetTextureInfo(texture)};
        auto cmd{rhitest::Required(device.CreateCommandList(rhi::QueueType::Graphics, "present_cmd"))};
        cmd->Begin();
        cmd->Transition(texture, rhi::ResourceState::Undefined, rhi::ResourceState::RenderTarget);
        cmd->FlushBarriers();
        cmd->BeginRendering({
            .Color      = {{
                .Texture    = texture,
                .LoadOp     = rhi::LoadOp::Clear,
                .StoreOp    = rhi::StoreOp::Store,
                .ClearValue = {0.1F, 0.3F, 0.8F, 1.F},
            }},
            .RenderArea = {info.Extent.Width, info.Extent.Height},
        });
        cmd->EndRendering();
        cmd->Transition(texture, rhi::ResourceState::RenderTarget, rhi::ResourceState::Present);
        cmd->FlushBarriers();
        // Showing the frame is recorded like the rest of it, so one submission carries everything.
        cmd->Present(provider.Handle());
        cmd->End();

        device.WaitForFence(device.Submit(*cmd));
    }

    /** {brief} Opens a provider on what this harness owns. */
    std::unique_ptr<rhi::IExternalTextureProvider> MakeProvider(rhi::IDevice &device, const HarnessTextureOwner &owner,
                                                               rhi::Format format = rhi::Format::Undefined)
    {
        // No cast and no callback: the owner is already the type this build's factory takes.
        return rhitest::Required(rhitest::MakeProviderFor(device, owner.Get(), format));
    }
} // namespace

TEST(a_provider_hands_out_frames_and_shows_them)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    HarnessTextureOwner surface{*device, kWidth, kHeight};
    auto        target{MakeProvider(*device, surface, rhi::Format::BGRA8Unorm)};
    REQUIRE(target != nullptr);
    REQUIRE(target->TextureFormat() == rhi::Format::BGRA8Unorm);

    for (uint32_t i{0}; i < 8; ++i)
    {
        const auto texture{target->NextTexture()};
        REQUIRE(texture.has_value());
        REQUIRE(texture->Valid());
        // Every frame answers for itself, with the surface's format and its own size. The size
        // asked for is only the size handed back when this surface is one that can be told.
        const rhi::TextureInfo info{device->GetTextureInfo(*texture)};
        REQUIRE(info.Format == rhi::Format::BGRA8Unorm);
        REQUIRE(info.Extent.Width > 0);
        REQUIRE(info.Extent.Height > 0);
        if constexpr (rhitest::kTestTextureOwnerHonoursExtent)
        {
            REQUIRE(info.Extent.Width == kWidth);
            REQUIRE(info.Extent.Height == kHeight);
        }
        DrawAndPresent(*device, *target, *texture);
    }
    std::printf("  presented 8 frames with no readback\n");
}

TEST(one_device_serves_several_providers)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    HarnessTextureOwner firstSurface{*device, kWidth, kHeight};
    HarnessTextureOwner secondSurface{*device, kWidth / 2, kHeight / 2};

    auto first{MakeProvider(*device, firstSurface)};
    auto second{MakeProvider(*device, secondSurface)};
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);

    // Each target answers for its own surface; neither is the device's single opinion of "the"
    // surface, which is the whole reason a target exists.
    for (uint32_t i{0}; i < 4; ++i)
    {
        const auto firstTexture{first->NextTexture()};
        REQUIRE(firstTexture.has_value());
        const auto secondTexture{second->NextTexture()};
        REQUIRE(secondTexture.has_value());

        // Each target keeps its own surface's size, rather than both reporting one the device
        // holds - which is the whole reason a target exists and not a device-wide surface.
        if constexpr (rhitest::kTestTextureOwnerHonoursExtent)
        {
            REQUIRE(device->GetTextureInfo(*firstTexture).Extent.Width == kWidth);
            REQUIRE(device->GetTextureInfo(*secondTexture).Extent.Width == kWidth / 2);
        }
        DrawAndPresent(*device, *first, *firstTexture);
        DrawAndPresent(*device, *second, *secondTexture);
    }
    std::printf("  one device presented 4 frames each to two surfaces\n");
}

TEST(a_resized_source_hands_out_resized_frames)
{
    auto device{rhitest::Required(rhi::CreateDevice({}))};
    HarnessTextureOwner surface{*device, kWidth, kHeight};
    auto        target{MakeProvider(*device, surface)};

    const auto before{target->NextTexture()};
    REQUIRE(before.has_value());
    DrawAndPresent(*device, *target, *before);

    constexpr uint32_t kResizedWidth{128};
    constexpr uint32_t kResizedHeight{96};
    if (!surface.Resize(kResizedWidth, kResizedHeight))
    {
        // Nothing out here drives this surface's size, so there is no resize to follow. The
        // contract is unchanged; only this harness cannot stage it.
        std::printf("  this surface's size cannot be driven from the harness; resize not staged\n");
        return;
    }

    // The frame after the resize is the new size, which is the whole of what a caller needs to
    // know: it sizes its work by the image it was handed, so it cannot be drawing at yesterday's
    // size. Nothing has to be told about the resize, and nothing caches it.
    const auto after{target->NextTexture()};
    REQUIRE(after.has_value());
    REQUIRE(after->Valid());
    const rhi::TextureInfo info{device->GetTextureInfo(*after)};
    REQUIRE(info.Extent.Width == kResizedWidth);
    REQUIRE(info.Extent.Height == kResizedHeight);
    DrawAndPresent(*device, *target, *after);

    // A surface with no window behind it keeps whatever size it has rather than going to zero, so
    // the Unavailable skip path cannot be induced from here: it needs a real window, minimized or
    // mid-resize.
    std::printf("  frames followed a resize from %ux%u to %ux%u; the unavailable path needs a "
                "windowed runner\n",
                kWidth, kHeight, kResizedWidth, kResizedHeight);
}

TEST(a_steady_frame_loop_allocates_nothing)
{
    // Validation off: this measures the configuration a realtime application ships. With validation
    // on, each submission looks its capture scope up by name, which allocates deliberately.
    auto device{rhitest::Required(rhi::CreateDevice(rhi::DeviceDesc{.EnableValidation = false}))};
    HarnessTextureOwner surface{*device, kWidth, kHeight};
    auto        target{MakeProvider(*device, surface)};

    // Warm up so first-use allocations inside the device are not counted as steady state.
    for (uint32_t i{0}; i < 4; ++i)
    {
        const auto texture{target->NextTexture()};
        REQUIRE(texture.has_value());
        DrawAndPresent(*device, *target, *texture);
    }

    gLightRhiAllocations = 0;
    gPlatformAllocations = 0;
    for (uint32_t i{0}; i < 32; ++i)
    {
        // Only the two calls this test is about are counted. Recording a frame needs a command
        // list, which allocates by design, so that part of the loop stays outside the window.
        gCounting = true;
        const auto texture{target->NextTexture()};
        gCounting = false;
        REQUIRE(texture.has_value());
        REQUIRE(texture->Valid());

        auto cmd{rhitest::Required(device->CreateCommandList(rhi::QueueType::Graphics, "steady_cmd"))};
        cmd->Begin();
        cmd->Transition(*texture, rhi::ResourceState::Undefined, rhi::ResourceState::Present);
        cmd->FlushBarriers();
        cmd->Present(target->Handle());
        cmd->End();

        gCounting = true;
        const rhi::FenceHandle fence{device->Submit(*cmd)};
        gCounting = false;
        device->WaitForFence(fence);
    }

    // LightRHI must add nothing per frame. The platform's own cost is reported, not asserted: those
    // allocations happen inside the platform's presentation stack while it makes the next frame
    // ready to be drawn into, which is below anything LightRHI chooses.
    std::printf("  on %s (%s build): 32 present frames made LightRHI %zu allocations, platform %zu\n",
                std::string{device->AdapterName()}.c_str(), kBuildConfiguration, gLightRhiAllocations,
                gPlatformAllocations);
    REQUIRE(gLightRhiAllocations == 0);
}

int main()
{
    std::printf("\n── Presentation tests ──────────────────────────\n");
    test_a_provider_hands_out_frames_and_shows_them();
    test_one_device_serves_several_providers();
    test_a_resized_source_hands_out_resized_frames();
    test_a_steady_frame_loop_allocates_nothing();
    std::printf("\nAll tests passed.\n");
    return 0;
}
