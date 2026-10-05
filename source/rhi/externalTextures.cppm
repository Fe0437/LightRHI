/**
 * {file}
 * {brief} Textures this library did not make, and whatever owns them.
 *
 * Two words are used here and they are not the same thing:
 *   - a *source* is the platform's own object - a CAMetalLayer, a VkSurfaceKHR - created and owned
 *     by whatever opened the window. This library never makes one; it is handed one.
 *   - a *provider* is this library's object, built from a source, which lends out the textures
 *     that source owns, one per frame.
 *
 * Those textures belong to the presentation engine, not to the application: it owns a small set of
 * them, hands one over when it is free, and shows it again once the work that drew it is done.
 * This is that handover and nothing else - no formats to negotiate, no image count, no present
 * mode. The application owns the window; this is the texture it may draw into right now.
 *
 * Showing a texture is recorded like any other work, with ICommandList::Present(), so a frame is
 * submitted the same way every other piece of work is. That is not a convenience: both backends
 * order presentation with the submission that drew it, and a submission that knows it ends in a
 * present carries that ordering itself instead of paying for a second one.
 */
module;

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

export module rhi:externalTextures;

import :handles;
import :types;

export namespace rhi
{
#if METRICS_ENABLED
    /**
     * {brief} What became of one presented frame: when it reached the screen, or that it never did.
     *
     * Compiled only with METRICS_ENABLED, like {apiref} IExternalTextureProvider::TakeShownFrames.
     *
     * ```cpp
     * std::array<rhi::ShownFrame, 16> shown{};
     * for (const auto &frame : std::span{shown}.first(provider.TakeShownFrames(shown)))
     * {
     *     if (frame.ShownNanoseconds != 0) { latency.Add(frame.ShownNanoseconds - presentedAt[frame.Present]); }
     * }
     * ```
     */
    struct ShownFrame
    {
        std::uint64_t Present{}; ///< Which present: 1 for the provider's first, then one more for each.
        /// When it reached the screen, in nanoseconds of the process's `std::chrono::steady_clock`;
        /// zero when a newer frame replaced it before the screen took it.
        std::uint64_t ShownNanoseconds{};
    };
#endif // METRICS_ENABLED

    /** {brief} Why a provider has no texture to give. */
    enum class ExternalTextureError : std::uint8_t
    {
        Unavailable, ///< Nothing is free this instant, or the source has no area; skip the frame.
        Lost         ///< The source or its device is gone; build a new one.
    };

    /**
     * {brief} Whatever owns the textures a provider lends out.
     *
     * *External* is the same word used by everything else here, and means the same thing: this
     * library did not make it. A CAMetalLayer on one build, a VkSurfaceKHR on another, and which
     * it is stays inside the backend that was compiled - naming the type here would put a vendor
     * header in this library's public interface, and every consumer would then need metal-cpp or
     * the Vulkan headers to call a function that only passes the value along.
     * {note} The one cast this costs is written by the backend, against the type it was compiled
     * for, so a value of the wrong kind cannot reach it from a build that does not have that type.
     */
    struct ExternalTextureOwner
    {
        void *Value{}; ///< The owner, as the toolkit that made it returns it.
    };

    /**
     * {brief} When a presented frame may reach the screen.
     *
     * Neither choice ever tears. `OnRefresh` queues every frame for its own refresh, so a frame
     * presented while others wait is shown only after them. `AsSoonAsReady` lets the newest frame
     * replace any that is still waiting, which keeps what is on screen closest to the latest input;
     * a drawing surface wants this. Where a platform cannot do it, `OnRefresh` is used instead.
     *
     * ```cpp
     * auto provider{rhi::CreateExternalTextureProvider(device, owner, rhi::Format::Undefined,
     *                                                  rhi::PresentTiming::AsSoonAsReady)};
     * ```
     */
    enum class PresentTiming : uint8_t
    {
        OnRefresh,     ///< Every frame waits for its own display refresh, in order.
        AsSoonAsReady, ///< The newest finished frame is shown at the next refresh; older ones are skipped.
    };

    /**
     * {brief} What a caller needs from the device before it can make an owner at all.
     *
     * Empty where the window has the owner already. Where it does not - a Vulkan surface is made
     * from the instance inside the device - this carries what the toolkit must be given.
     */
    struct ExternalTextureOwnerContext
    {
        void *Value{}; ///< Opaque to everything but the toolkit call that consumes it.
    };

    /**
     * {brief} What every texture from a provider will be, answerable before any frame exists.
     *
     * Separated because it is asked at a different time, by different code, than textures are: a
     * pipeline is built once, at start-up, from the format alone. It cannot change for the life of
     * the provider, so nothing that only builds pipelines needs to be handed a source of textures.
     */
    class ITextureFormatProvider
    {
      public:
        ITextureFormatProvider(const ITextureFormatProvider &)            = delete;
        ITextureFormatProvider(ITextureFormatProvider &&)                 = delete;
        ITextureFormatProvider &operator=(const ITextureFormatProvider &) = delete;
        ITextureFormatProvider &operator=(ITextureFormatProvider &&)      = delete;

        virtual ~ITextureFormatProvider() = default;

        /**
         * {brief} The format of every texture this provider hands out.
         *
         * It is the format requested when the source could provide it and the source's own
         * choice otherwise, so it is what will actually arrive rather than what was asked for.
         */
        [[nodiscard]] virtual Format TextureFormat() const noexcept = 0;

      protected:
        ITextureFormatProvider() = default;
    };

    /**
     * {brief} Images that belong to the display, lent back one at a time.
     *
     * Not textures this library made: the system owns a small set of them, hands one over when it
     * is free, and takes it back once the work that drew it has been shown. That borrowing is the
     * whole of this ability, and it is why there is no single texture to keep hold of.
     */
    class ITextureProvider
    {
      public:
        ITextureProvider(const ITextureProvider &)            = delete;
        ITextureProvider(ITextureProvider &&)                 = delete;
        ITextureProvider &operator=(const ITextureProvider &) = delete;
        ITextureProvider &operator=(ITextureProvider &&)      = delete;

        virtual ~ITextureProvider() = default;

        /**
         * {brief} Takes the texture to draw this frame into.
         *
         * The handle belongs to the provider and stays valid until the submission that presents it
         * completes, so it is taken once per frame and not stored. Unavailable is an ordinary
         * answer - the display is still using every image, or the window has no area - and the
         * frame is skipped rather than the call being retried in a loop.
         */
        [[nodiscard]] virtual std::expected<TextureHandle, ExternalTextureError> NextTexture() = 0;

      protected:
        ITextureProvider() = default;
    };

    /**
     * {brief} Textures this library did not make, offered one at a time to be drawn and shown.
     *
     * External is the whole distinction, and it is what separates these from a texture you create:
     *   - the presentation engine allocates them, not this library;
     *   - there are several, rotating, and which one arrives is not yours to choose;
     *   - one is yours only until the submission that presents it completes, so it is taken per
     *     frame and never stored;
     *   - their format and size are the source's answer, not your request.
     *
     * One provider is one surface. A device serves as many as the application creates, and a
     * device that serves none has nothing extra to answer for.
     *
     * The two abilities above are stated separately because they are used separately: setting up
     * asks only the format, drawing asks only for textures. This names their union, which is what
     * a presentable surface actually offers, so a caller that needs both takes one object and a
     * caller that needs one takes only what it uses.
     */
    class IExternalTextureProvider : public ITextureFormatProvider, public ITextureProvider
    {
      public:
        ~IExternalTextureProvider() override = default;

        /**
         * {brief} Names this provider to a device, so a present can be recorded against it.
         *
         * Presenting takes this value rather than this object: a recorded present then carries an
         * opaque slot the device resolves, exactly as drawing carries a buffer or texture handle,
         * and no backend has to recover a concrete type from a base it was handed.
         */
        [[nodiscard]] virtual ExternalTextureProviderHandle Handle() const noexcept = 0;

#if METRICS_ENABLED
        /**
         * {brief} Hands over what became of presented frames since the last call, oldest first.
         *
         * Compiled only with METRICS_ENABLED: it exists to measure, so a shipping build neither
         * records nor offers it.
         * A frame's fate is known only once the platform reports it, a refresh or two after its
         * present, so this answers for earlier presents than the latest. Call it often: at most a
         * fixed number of fates are kept, and the oldest are dropped first.
         * {param frames} Receives the fates; the number written is returned.
         * {returns} How many were written. Always zero where the platform cannot tell when a frame
         * reached the screen, as on Vulkan without a display-timing extension.
         * {note} Safe to call from the thread that presents while the platform reports from another.
         */
        [[nodiscard]] virtual std::size_t TakeShownFrames(std::span<ShownFrame> frames) noexcept = 0;
#endif // METRICS_ENABLED

      protected:
        IExternalTextureProvider() = default;
    };

} // namespace rhi
