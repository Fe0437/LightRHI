/**
 * {file}
 * {brief} The fields of the structs a shader library takes as parameters, such as its push
 * constants, as the shader compiler laid them out.
 *
 * tools/compile_shaders.py writes `<library>.layout` beside each library from the compiler's own
 * reflection, so a caller fills a parameter by field name, at the offset the compiler chose, and
 * never repeats the struct in C++. Which fields a caller fills, and with what, is the caller's.
 */
module;
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

export module rhi:shaderLayout;
import :pipeline;

export namespace rhi
{
    /** {brief} One field of a parameter struct, where the compiler placed it. */
    struct ShaderLayoutField
    {
        std::string   Name;     ///< As the shader names it.
        std::uint32_t Offset{}; ///< Bytes from the start of the struct.
        std::uint32_t Size{};   ///< Bytes it takes.
        /**
         * The type as a layout names it: a scalar ("float32", "uint32"), a vector of one
         * ("float32x2"), or a pointer to a type as the shader language names that type
         * ("pointer float2").
         */
        std::string Type;
    };

    /**
     * {brief} The fields of every parameter struct of one shader library.
     *
     * ```cpp
     * const auto layout{rhi::ReadShaderLayout(shaderDir, "dab_stamp")};
     * const auto *tileX{layout->Field("g_pc", "TileX")};   // where to write it, and how big
     * ```
     */
    class ShaderLayout
    {
      public:
        /** {brief} The first line of every layout file, which says what it is and its version. */
        static constexpr std::string_view Header{"lightrhi shader layout 1"};

        /**
         * {brief} Read a layout from its text: the header, then `parameter <name>` lines each
         * followed by its `field <name> <offset> <size> <type>` lines.
         * {returns} The layout, or MalformedLayout.
         */
        [[nodiscard]] static std::expected<ShaderLayout, ShaderArtifactError> Parse(const std::string_view text)
        {
            ShaderLayout layout;
            bool         first{true};
            for (std::string_view rest{text}; !rest.empty();)
            {
                const auto end{rest.find('\n')};
                const auto line{rest.substr(0, end)};
                rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 1U);
                if (first)
                {
                    if (line != Header)
                    {
                        return std::unexpected{ShaderArtifactError::MalformedLayout};
                    }
                    first = false;
                    continue;
                }
                if (line.empty())
                {
                    continue;
                }
                if (!layout._read(line))
                {
                    return std::unexpected{ShaderArtifactError::MalformedLayout};
                }
            }
            if (first)
            {
                return std::unexpected{ShaderArtifactError::MalformedLayout};
            }
            return layout;
        }

        /** {brief} Read the layout file at `path`. {returns} The layout, or NotFound / Unreadable / MalformedLayout. */
        [[nodiscard]] static std::expected<ShaderLayout, ShaderArtifactError> Read(const std::string &path)
        {
            std::ifstream file{path};
            if (!file)
            {
                return std::unexpected{ShaderArtifactError::NotFound};
            }
            const std::string text{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
            if (file.bad())
            {
                return std::unexpected{ShaderArtifactError::Unreadable};
            }
            return Parse(text);
        }

        /** {brief} The fields of the parameter `parameter`, in the order the struct declares them; empty when it has
         * none. */
        [[nodiscard]] std::span<const ShaderLayoutField> Fields(const std::string_view parameter) const noexcept
        {
            const auto found{std::ranges::find(_parameters, parameter, &Parameter::Name)};
            return found == _parameters.end() ? std::span<const ShaderLayoutField>{} : found->Fields;
        }

        /** {brief} The field `name` of the parameter `parameter`, or nullptr when there is none. */
        [[nodiscard]] const ShaderLayoutField *Field(const std::string_view parameter,
                                                     const std::string_view name) const noexcept
        {
            const auto fields{Fields(parameter)};
            const auto found{std::ranges::find(fields, name, &ShaderLayoutField::Name)};
            return found == fields.end() ? nullptr : &*found;
        }

      private:
        /** {brief} One parameter struct and its fields. */
        struct Parameter
        {
            std::string                    Name;
            std::vector<ShaderLayoutField> Fields;
        };

        /** {brief} Read one line after the header; false when it is neither a parameter nor a field of one. */
        [[nodiscard]] bool _read(std::string_view line)
        {
            const auto word{[&line]
                            {
                                const auto space{line.find(' ')};
                                const auto taken{line.substr(0, space)};
                                line = space == std::string_view::npos ? std::string_view{} : line.substr(space + 1U);
                                return taken;
                            }};
            const auto number{[&word](std::uint32_t &value)
                              {
                                  const auto text{word()};
                                  const auto [end,
                                              error]{std::from_chars(text.data(), text.data() + text.size(), value)};
                                  return error == std::errc{} && end == text.data() + text.size();
                              }};
            const auto kind{word()};
            if (kind == "parameter" && !line.empty())
            {
                _parameters.push_back({.Name = std::string{line}, .Fields = {}});
                return true;
            }
            if (kind != "field" || _parameters.empty())
            {
                return false;
            }
            ShaderLayoutField field{.Name = std::string{word()}};
            if (field.Name.empty() || !number(field.Offset) || !number(field.Size) || line.empty())
            {
                return false;
            }
            field.Type = std::string{line};
            _parameters.back().Fields.push_back(std::move(field));
            return true;
        }

        std::vector<Parameter> _parameters;
    };

    /**
     * {brief} The layout of the library `library` in `directory`: the file `<library>.layout` that
     * tools/compile_shaders.py writes beside it, whatever the backend.
     */
    [[nodiscard]] inline std::expected<ShaderLayout, ShaderArtifactError>
    ReadShaderLayout(const std::string_view directory, const std::string_view library)
    {
        std::string path{directory};
        path.append("/").append(library).append(".layout");
        return ShaderLayout::Read(path);
    }
} // namespace rhi
