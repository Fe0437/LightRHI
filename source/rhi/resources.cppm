module;
#include <cstdint>

export module rhi:resources;
import :types;
import :handles;
import :sync;
import :descriptors;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // Texture view — a window into a texture's mip/layer range.
    // Used for binding a specific mip as a render target, or a single
    // array layer as a shader resource.
    // ---------------------------------------------------------------------------

    struct TextureView
    {
        TextureHandle    Texture;
        SubresourceRange Range{};
        TextureDimension ViewDimension{TextureDimension::Tex2D};
        Format           Format{Format::Undefined}; // Undefined = inherit from texture
    };

    // ---------------------------------------------------------------------------
    // Buffer info — static layout of a GPU Buffer (for stride / element math).
    // ---------------------------------------------------------------------------

    struct BufferInfo
    {
        uint64_t    Size{0};
        uint32_t    Stride{0}; // 0 = raw/unstructured
        BufferUsage Usage{};
        MemoryType  MemoryType{};
        GpuAddress  DeviceAddress{}; // 0 if BDA not requested
    };

    // ---------------------------------------------------------------------------
    // Indirect draw argument formats (std140 compatible).
    // Mirrors VkDrawIndirectCommand / VkDrawIndexedIndirectCommand.
    // Use GpuAddress to push a buffer holding an array of these.
    // ---------------------------------------------------------------------------

    struct DrawIndirectArgs
    {
        uint32_t VertexCount{0};
        uint32_t InstanceCount{1};
        uint32_t FirstVertex{0};
        uint32_t FirstInstance{0};
    };

    struct DrawIndexedIndirectArgs
    {
        uint32_t IndexCount{0};
        uint32_t InstanceCount{1};
        uint32_t FirstIndex{0};
        int32_t  VertexOffset{0};
        uint32_t FirstInstance{0};
    };

    struct DispatchIndirectArgs
    {
        uint32_t X{1};
        uint32_t Y{1};
        uint32_t Z{1};
    };

} // namespace rhi
