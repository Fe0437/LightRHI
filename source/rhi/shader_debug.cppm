module;
#include <cstddef>
#include <cstdint>

export module rhi:shaderDebug;

export namespace rhi
{

#if DEBUG_ENABLED
    inline constexpr std::size_t ShaderDebugSlotCount{16};

    // Must match shaders/shader_debug.slangh. One selected shader thread
    // writes one record; keeping this POD in LightRHI gives host code a
    // backend-neutral layout for direct mapping/readback.
    struct ShaderDebugRecord
    {
        std::uint32_t ThreadIndex{0};
        std::uint32_t Symbols[ShaderDebugSlotCount]{};
        std::uint32_t Values[ShaderDebugSlotCount]{};
    };

    static_assert(sizeof(ShaderDebugRecord) == 132);
#endif

} // namespace rhi
