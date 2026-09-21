/**
 * {file} bindless.cppm
 * {brief} Defines the backend-neutral bindless heap contract and its limits.
 */
module;
#include <cstdint>

export module rhi:bindless;
import :types;
import :handles;

export namespace rhi
{

    /**
     * {brief} Requested capacities for a device-wide bindless heap.
     * {note} Device creation may impose lower hardware limits; query IBindlessHeap afterward.
     */
    struct BindlessLimits
    {
        uint32_t MaxBuffers{1U << 20U};  ///< Maximum simultaneously live bindless buffers.
        uint32_t MaxTextures{1U << 20U}; ///< Maximum simultaneously live bindless textures.
        uint32_t MaxSamplers{2048};    ///< Maximum simultaneously live bindless samplers.
    };

    /**
     * {brief} Reports capacity and occupancy of the device-wide bindless resource table.
     *
     * IDevice registers resources automatically at creation and releases their
     * slots at destruction. Callers normally pass values returned by the device
     * address methods to shaders and use this interface only for limits or diagnostics.
     *
     * \par Slang usage
     * ```hlsl
     * struct RootConstants
     * {
     *     DescriptorHandle<Texture2D> Texture;
     * };
     *
     * #include "light_rhi_shader_abi.slangh"
     * [[vk::push_constant]] ConstantBuffer<RootConstants> constants
     *     : register(LIGHTRHI_PUSH_CONSTANT_REGISTER);
     * Texture2D texture = constants.Texture;
     * ```
     * Populate the matching host field with IDevice::TextureAddress(). The
     * resource must remain live until every dispatch using the value completes.
     */
    class IBindlessHeap
    {
      public:
        /** {brief} Releases the interface with its owning device. */
        virtual ~IBindlessHeap() = default;

        /** {brief} Returns the maximum number of simultaneously live bindless buffers. */
        [[nodiscard]] virtual uint32_t MaxBuffers() const noexcept = 0;

        /** {brief} Returns the maximum number of simultaneously live bindless textures. */
        [[nodiscard]] virtual uint32_t MaxTextures() const noexcept = 0;

        /** {brief} Returns the maximum number of simultaneously live bindless samplers. */
        [[nodiscard]] virtual uint32_t MaxSamplers() const noexcept = 0;

        /**
         * {brief} Returns a shader-visible heap address when the active binding model requires one.
         * {returns} An invalid address when shaders use resource values directly.
         */
        [[nodiscard]] virtual GpuAddress HeapAddress() const noexcept = 0;

        /** {brief} Returns the number of buffer slots currently occupied. */
        [[nodiscard]] virtual uint32_t UsedBuffers() const noexcept = 0;

        /** {brief} Returns the number of texture slots currently occupied. */
        [[nodiscard]] virtual uint32_t UsedTextures() const noexcept = 0;

        /** {brief} Returns the number of sampler slots currently occupied. */
        [[nodiscard]] virtual uint32_t UsedSamplers() const noexcept = 0;
    };

} // namespace rhi
