/**
 * {file}
 * {brief} A compiled shader library, holding every entry point of one shader source.
 *
 * What a library file is called and what format it holds are the backend's to say: each backend
 * module exports ReadShaderLibrary() for its own files. This type only keeps what was read and
 * describes an entry point from it, so it names no backend.
 */
module;
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

export module rhi:shaderLibrary;
import :types;
import :pipeline;

export namespace rhi
{
    /** {brief} One compiled shader library and the format its bytes are in. */
    class ShaderLibrary
    {
      public:
        /**
         * {brief} The largest library Read() accepts unless told otherwise.
         *
         * Far larger than any real shader library - even large ray-tracing libraries are a few
         * megabytes - and far short of what would exhaust memory, so a wrong or damaged file is
         * refused before anything is allocated for it.
         */
        static constexpr std::uint64_t DefaultMaximumBytes{std::uint64_t{64} << 20U};

        /**
         * {brief} Reads the library file at `path`.
         * {param format} What the file holds; the backend reading it knows.
         * {param maximumBytes} The largest file accepted; the size is checked before any allocation.
         * {returns} The library, or NotFound / InvalidSize / Unreadable.
         * {note} For backends. An application calls its backend's ReadShaderLibrary().
         */
        [[nodiscard]] static std::expected<ShaderLibrary, ShaderArtifactError>
        Read(const std::string &path, const ShaderFormat format, const std::uint64_t maximumBytes = DefaultMaximumBytes)
        {
            std::error_code error;
            const auto      size{std::filesystem::file_size(path, error)};
            std::ifstream   file{path, std::ios::binary};
            if (error || !file)
            {
                return std::unexpected{ShaderArtifactError::NotFound};
            }
            if (size == 0U || size > maximumBytes)
            {
                return std::unexpected{ShaderArtifactError::InvalidSize};
            }
            std::vector<std::byte> bytes(static_cast<std::size_t>(size));
            // A stream reads only into char, and char may view any object's bytes, std::byte included.
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            if (!file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size)))
            {
                return std::unexpected{ShaderArtifactError::Unreadable};
            }
            return ShaderLibrary{format, std::move(bytes)};
        }

        /**
         * {brief} The entry point `entryPoint`, described for pipeline creation.
         * {returns} A descriptor viewing this library's bytes: keep the library while it is used.
         *
         * ```cpp
         * const auto library{rhi::ReadShaderLibrary(shaderDir, "blur")};
         * const auto shader{library->Describe("horizontal", rhi::ShaderStage::Compute)};
         * ```
         */
        [[nodiscard]] std::expected<ShaderDesc, ShaderArtifactError> Describe(const std::string_view entryPoint,
                                                                              const ShaderStage      stage) const
        {
            return ToShaderDesc({.Format = _format, .Stage = stage, .EntryPoint = entryPoint, .Data = _bytes});
        }

      private:
        ShaderLibrary(const ShaderFormat format, std::vector<std::byte> bytes)
            : _format{format}, _bytes{std::move(bytes)}
        {
        }

        ShaderFormat           _format;
        std::vector<std::byte> _bytes;
    };
} // namespace rhi
