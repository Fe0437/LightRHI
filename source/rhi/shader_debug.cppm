module;
#include <cstddef>
#include <cstdint>

export module rhi:shaderDebug;

export namespace rhi
{

#if DEBUG_ENABLED
    /// \brief Number of symbolic values available in one shader debug record.
    inline constexpr std::size_t ShaderDebugSlotCount{16};

    /// \brief CPU-readable record written by one shader invocation selected for debugging.
    ///
    /// Bind a buffer of this layout through the matching helpers in
    /// `shaders/shader_debug.slangh`, then read it only after the dispatch completes.
    /// \note This layout is a host/shader ABI and must remain identical to the Slang definition.
    struct ShaderDebugRecord
    {
        std::uint32_t ThreadIndex{0};                  ///< Linear invocation index that wrote the record.
        std::uint32_t Symbols[ShaderDebugSlotCount]{}; ///< Application-defined identifiers for captured values.
        std::uint32_t Values[ShaderDebugSlotCount]{};  ///< Bit patterns captured for the matching symbols.
    };

    static_assert(sizeof(ShaderDebugRecord) == 132);
#endif

} // namespace rhi
