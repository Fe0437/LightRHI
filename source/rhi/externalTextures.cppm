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

#include <cstdint>
#include <expected>

export module rhi:externalTextures;

import :handles;
import :types;

export namespace rhi
{
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

      protected:
        IExternalTextureProvider() = default;
    };

} // namespace rhi
