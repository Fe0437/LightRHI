module;
#include <cstdint>

export module rhi:sync;
import :types;
import :handles;

export namespace rhi
{

    // ---- Texture subresources ----

    /// \brief Selects contiguous mip levels and array layers within a texture.
    struct SubresourceRange
    {
        uint32_t BaseMip{0};      ///< First selected mip level.
        uint32_t MipCount{~0U};   ///< Number of mips, or `~0U` for all remaining mips.
        uint32_t BaseLayer{0};    ///< First selected array layer.
        uint32_t LayerCount{~0U}; ///< Number of layers, or `~0U` for all remaining layers.
    };

    // ---- Explicit resource barriers ----

    /// \brief Describes an explicit state transition for selected texture subresources.
    /// \note The caller tracks the current state; LightRHI does not infer `Before`.
    ///
    /// \code{.cpp}
    /// cmd->Transition(texture, ResourceState::Undefined, ResourceState::TransferDst);
    /// cmd->FlushBarriers();
    /// cmd->CopyBufferToTexture(staging, 0, texture, region);
    /// cmd->Transition(texture, ResourceState::TransferDst, ResourceState::ShaderRead);
    /// cmd->FlushBarriers();
    /// \endcode
    struct TextureBarrier
    {
        TextureHandle    Texture; ///< Texture being transitioned.
        ResourceState    Before;  ///< State of earlier access.
        ResourceState    After;   ///< State required by later access.
        SubresourceRange Range{}; ///< Affected subresources; defaults to the whole texture.
    };

    /// \brief Describes an explicit state transition for a byte range of a buffer.
    struct BufferBarrier
    {
        BufferHandle  Buffer;      ///< Buffer being transitioned.
        ResourceState Before;      ///< State of earlier access.
        ResourceState After;       ///< State required by later access.
        uint64_t      Offset{0};   ///< First affected byte.
        uint64_t      Size{~0ULL}; ///< Number of bytes, or `~0ULL` for the rest of the buffer.
    };

    /// \brief Orders all accesses in one state class before all accesses in another.
    /// \note Prefer resource-scoped barriers when the dependency concerns a known resource.
    struct MemoryBarrier
    {
        ResourceState Before; ///< State of accesses that must complete first.
        ResourceState After;  ///< State of accesses that may begin afterward.
    };

    // ---- Timeline points ----

    /// \brief Represents an optional value on a GPU submission timeline.
    struct TimelinePoint
    {
        uint64_t Value{0}; ///< Timeline value; zero means no point was selected.

        /// \brief Reports whether a non-zero timeline value is present.
        [[nodiscard]] constexpr bool Valid() const noexcept
        {
            return Value != 0;
        }
    };

} // namespace rhi
