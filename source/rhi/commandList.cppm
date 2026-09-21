/**
 * {file} commandList.cppm
 * {brief} Defines commands recorded for later GPU submission. Device ownership is defined elsewhere.
 */
module;
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

export module rhi:commandList;
import :types;
import :handles;
import :externalTextures;
import :descriptors;
import :pipeline;
import :sync;
import :resources;
import :raytracing;

export namespace rhi
{

    /**
     * {brief} Records an ordered sequence of commands for one queue submission.
     *
     * Record commands between Begin() and End(), then pass the list to
     * IDevice::Submit(). A command list is not thread-safe and must not be
     * recorded or reused while an earlier submission of it is still running.
     *
     * ```cpp
     * auto cmd{device->CreateCommandList(QueueType::Compute, "simulation")};
     * cmd->Begin();
     * cmd->SetPipeline(simulationPipeline);
     * cmd->SetPushConstants(SimulationConstants{});
     * cmd->Dispatch(groupCountX, groupCountY);
     * cmd->End();
     * const FenceHandle fence = device->Submit(*cmd);
     * // Do not call Begin() again until fence completes.
     * ```
     */
    class ICommandList
    {
      public:
        /** {brief} Releases the command-list object after any submission using it has completed. */
        virtual ~ICommandList() = default;

        // ---- Lifecycle ----

        /**
         * {brief} Starts a new recording and discards commands from the previous completed use.
         * {pre} The command list is not recording and is not in flight.
         */
        virtual void Begin() = 0;

        /**
         * {brief} Finishes the current recording so it can be submitted.
         * {pre} Begin() was called and all rendering and debug groups have been ended.
         */
        virtual void End() = 0;

        /**
         * {brief} Shows the named provider's current texture once this submission has drawn it.
         *
         * Recorded like any other work, so a frame needs no second submission path. The provider is
         * named by handle for the same reason a buffer or texture is: the device resolves it, and
         * recording carries a value rather than an object.
         * {param target} An external texture provider of this device, from its Handle().
         * {pre} A texture was taken from that target for this frame, not yet shown, and
         * transitioned to ResourceState::Present by the work recorded before this.
         * {note} The submission's fence completing proves the GPU finished the commands, not that
         * the display has shown them. Do not measure presentation latency from it.
         */
        virtual void Present(ExternalTextureProviderHandle target) = 0;

        /**
         * {brief} Enqueues a state transition for a texture subresource range.
         * {note} `barrier.Before` must match the state tracked by the caller.
         * {note} Enqueue related transitions together, then call FlushBarriers() once.
         */
        virtual void Transition(const TextureBarrier &barrier) = 0;

        /** {brief} Enqueues a state transition for a byte range of a buffer. */
        virtual void Transition(const BufferBarrier &barrier) = 0;

        /** {brief} Enqueues a global ordering dependency between two classes of access. */
        virtual void Transition(const MemoryBarrier &barrier) = 0;

        /** {brief} Enqueues a state transition for an acceleration structure. */
        virtual void Transition(const AccelerationStructureBarrier &barrier) = 0;

        /** {brief} Enqueues a whole-texture transition. */
        void Transition(TextureHandle tex, ResourceState before, ResourceState after)
        {
            Transition(TextureBarrier{.Texture = tex, .Before = before, .After = after});
        }

        /** {brief} Enqueues a whole-buffer transition. */
        void Transition(BufferHandle buf, ResourceState before, ResourceState after)
        {
            Transition(BufferBarrier{.Buffer = buf, .Before = before, .After = after});
        }

        /** {brief} Enqueues a transition for one acceleration structure. */
        void Transition(AccelerationStructureHandle as, ResourceState before, ResourceState after)
        {
            Transition(AccelerationStructureBarrier{.AccelerationStructure = as, .Before = before, .After = after});
        }

        /**
         * {brief} Records all transitions enqueued since the previous flush.
         * {note} Flush before recording work that relies on those transitions.
         */
        virtual void FlushBarriers() = 0;

        /**
         * {brief} Begins drawing to the attachments in `desc`.
         * {pre} Color attachments are in ResourceState::RenderTarget.
         * {pre} A writable depth attachment is in ResourceState::DepthWrite.
         * {note} Pipeline attachment formats and sample counts must match `desc`.
         */
        virtual void BeginRendering(const RenderingDesc &desc) = 0;

        /**
         * {brief} Ends the current rendering region.
         * {pre} BeginRendering() is active.
         */
        virtual void EndRendering() = 0;

        /** {brief} Sets the viewport used by subsequent draws in the current rendering region. */
        virtual void SetViewport(const Viewport &vp) = 0;

        /** {brief} Sets the pixel scissor used by subsequent draws in the current rendering region. */
        virtual void SetScissor(const Scissor &scissor) = 0;

        // ---- Pipeline ----

        /** {brief} Binds the graphics or compute pipeline used by subsequent draw or dispatch commands. */
        virtual void SetPipeline(PipelineHandle pipeline) = 0;

        /**
         * {brief} Binds a texture to the explicit `t` register used by the current shader.
         * {note} Call after SetPipeline() and before each draw or dispatch that needs the binding.
         * {note} Bindless `DescriptorHandle<Texture2D>` values do not use this method.
         * For `Texture2D input : register(t3)`, pass `index == 3`.
         */
        virtual void BindTexture(TextureHandle texture, uint32_t index) = 0;

        /**
         * {brief} Binds a sampler to the explicit `s` register used by the current shader.
         * {note} Call after SetPipeline() and before each draw or dispatch that needs the binding.
         * {note} Bindless `DescriptorHandle<SamplerState>` values do not use this method.
         * For `SamplerState sampler : register(s1)`, pass `index == 1`.
         */
        virtual void BindSampler(SamplerHandle sampler, uint32_t index) = 0;

        /**
         * {brief} Updates bytes in the current pipeline's push-constant block.
         * {param data} Source bytes copied during this call; its size is the number of bytes updated.
         * {param offset} Destination byte offset within the block.
         * {pre} A pipeline is bound and `offset + data.size()` does not exceed its declared range.
         */
        virtual void SetPushConstants(std::span<const std::byte> data, uint32_t offset = 0) = 0;

        /**
         * {brief} Updates push constants from one trivially copyable value.
         * {param data} Value whose object representation is copied.
         * {param offset} Destination byte offset within the push-constant block.
         *
         * ```cpp
         * struct RootConstants
         * {
         *     GpuAddress Vertices;
         *     GpuAddress Albedo;
         *     uint32_t MaterialIndex;
         * };
         * cmd->SetPushConstants(RootConstants{
         *     .Vertices      = device->BufferAddress(vertexBuffer),
         *     .Albedo        = device->TextureAddress(albedo),
         *     .MaterialIndex = materialIndex,
         * });
         * ```
         */
        template <typename T>
        requires std::is_trivially_copyable_v<T> && (!std::convertible_to<const T &, std::span<const std::byte>>)
        void SetPushConstants(const T &data, uint32_t offset = 0)
        {
            // A span is itself trivially copyable, so the constraint keeps one from being uploaded
            // as its own pointer and size instead of the bytes it names.
            SetPushConstants(std::as_bytes(std::span{&data, 1}), offset);
        }

        // ---- Index buffer ----

        /**
         * {brief} Binds the index buffer used by subsequent indexed draws.
         * {param offset} Byte offset of the first index in `buffer`.
         */
        virtual void BindIndexBuffer(BufferHandle buffer, uint64_t offset = 0,
                                     IndexType indexType = IndexType::Uint32) = 0;

        // ---- Draw ----

        /** {brief} Records a non-indexed draw using the current graphics state. */
        virtual void Draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0,
                          uint32_t firstInstance = 0) = 0;

        /** {brief} Records an indexed draw using the current graphics state and index buffer. */
        virtual void DrawIndexed(uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0,
                                 int32_t vertexOffset = 0, uint32_t firstInstance = 0) = 0;

        /**
         * {brief} Records one or more draws described by DrawIndirectArgs entries in `argsBuffer`.
         * {pre} `argsBuffer` has BufferUsage::IndirectArgs and is in ResourceState::IndirectArgument.
         */
        virtual void DrawIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                                  uint32_t stride = sizeof(DrawIndirectArgs)) = 0;

        /**
         * {brief} Records one or more indexed draws described by DrawIndexedIndirectArgs entries.
         * {pre} An index buffer is bound and `argsBuffer` is ready for indirect access.
         */
        virtual void DrawIndexedIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                                         uint32_t stride = sizeof(DrawIndexedIndirectArgs)) = 0;

        /**
         * {brief} Records up to `maxDrawCount` indirect draws, with the actual count read from `countBuffer`.
         * {pre} Both buffers are ready for indirect-argument access.
         */
        virtual void DrawIndirectCount(BufferHandle argsBuffer, uint64_t argsOffset, BufferHandle countBuffer,
                                       uint64_t countOffset, uint32_t maxDrawCount,
                                       uint32_t stride = sizeof(DrawIndirectArgs)) = 0;

        // ---- Compute ----

        /**
         * {brief} Dispatches `x` by `y` by `z` compute thread groups.
         * {pre} A compute pipeline is bound and no rendering region is active.
         */
        virtual void Dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) = 0;

        /**
         * {brief} Dispatches compute using one DispatchIndirectArgs value in `argsBuffer`.
         * {pre} A compute pipeline is bound and the buffer is ready for indirect access.
         */
        virtual void DispatchIndirect(BufferHandle argsBuffer, uint64_t argsOffset) = 0;

        /**
         * {brief} Records the current GPU timestamp into one query-pool slot.
         * {pre} IDevice::SupportsComputeTimestamps() is true.
         * {pre} `index` was reset after its previous use and is within `pool`.
         * {note} Read the result only after the submission fence completes.
         */
        virtual void WriteComputeTimestamp(TimestampQueryPoolHandle pool, uint32_t index) = 0;

        /**
         * {brief} Records a BLAS or TLAS build into a previously allocated handle.
         * {param desc} Must match the descriptor used to query sizes and create `handle`.
         * {param scratchBuffer} Storage buffer at least as large as `BuildScratchSize`.
         * {param scratchOffset} Byte offset of suitably aligned scratch storage.
         * {pre} Referenced geometry buffers and BLAS instances remain live until completion.
         * {note} Order a TLAS after its BLAS builds with an acceleration-structure transition.
         * {note} Acceleration structures are bindless: pass the value returned by
         * IDevice::AccelerationStructureAddress() through a Slang
         * `DescriptorHandle<RaytracingAccelerationStructure>`; no separate bind call is required.
         */
        virtual void BuildAccelerationStructure(AccelerationStructureHandle      handle,
                                                const AccelerationStructureDesc &desc, BufferHandle scratchBuffer,
                                                uint64_t scratchOffset = 0) = 0;

        // ---- Copy ----

        /**
         * {brief} Copies a byte range between two buffers.
         * {pre} Source and destination are in ResourceState::TransferSrc and TransferDst respectively.
         */
        virtual void CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &region) = 0;

        /**
         * {brief} Copies one texture region between compatible textures.
         * {pre} Source and destination are in ResourceState::TransferSrc and TransferDst respectively.
         */
        virtual void CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &region) = 0;

        /**
         * {brief} Copies tightly described pixel data from a buffer into one texture region.
         * {note} The buffer layout is the packed layout required by the destination format and extent.
         */
        virtual void CopyBufferToTexture(BufferHandle src, uint64_t srcOffset, TextureHandle dst,
                                         const TextureCopyRegion &region) = 0;

        /**
         * {brief} Copies one texture region into a buffer at `dstOffset`.
         * {note} The destination must have enough space for the packed region.
         */
        virtual void CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &region, BufferHandle dst,
                                         uint64_t dstOffset) = 0;

        /**
         * {brief} Clears the selected color-texture subresources.
         * {pre} `texture` supports transfer-destination use and is ready for that access.
         */
        virtual void ClearTexture(TextureHandle texture, const ClearColor &color,
                                  const SubresourceRange &range = {}) = 0;

        /**
         * {brief} Clears the selected depth/stencil texture subresources.
         * {pre} `texture` is a compatible depth format and is ready for transfer-destination access.
         */
        virtual void ClearDepthTexture(TextureHandle texture, const ClearDepthStencil &clear,
                                       const SubresourceRange &range = {}) = 0;

        /**
         * {brief} Fills a buffer byte range with a repeated 32-bit value.
         * {pre} The range is valid and the buffer is ready for transfer-destination access.
         */
        virtual void FillBuffer(BufferHandle buffer, uint64_t offset, uint64_t size, uint32_t value) = 0;

        // ---- Debug groups (reflected in GPU debuggers: RenderDoc, PIX, Xcode) ----

        /** {brief} Begins a nested command region visible in supporting GPU debugging tools. */
        virtual void BeginDebugGroup(std::string_view name, float r = 0.F, float g = 0.8F, float b = 0.F) = 0;

        /** {brief} Ends the innermost debug group. */
        virtual void EndDebugGroup() = 0;

        /** {brief} Inserts a point label at the current recording position. */
        virtual void InsertDebugLabel(std::string_view label) = 0;

        /**
         * {brief} Makes a bindless buffer discoverable to supporting GPU debugging tools.
         * {param unusedBindingSlot} A slot not declared or accessed by the shader.
         * {note} This does not change shader binding or execution.
         */
        virtual void DebugExposeBuffer(BufferHandle buffer, uint32_t unusedBindingSlot) = 0;

        /**
         * {brief} Makes a bindless acceleration structure discoverable to supporting GPU debugging tools.
         * {param unusedBindingSlot} A slot not declared or accessed by the shader.
         * {note} This does not change shader binding or execution.
         */
        virtual void DebugExposeAccelerationStructure(AccelerationStructureHandle accelerationStructure,
                                                      uint32_t                    unusedBindingSlot) = 0;

        /**
         * {brief} Makes a bindless texture discoverable to supporting GPU debugging tools.
         * {param unusedBindingSlot} A slot not declared or accessed by the shader.
         * {note} This does not change shader binding or execution.
         */
        virtual void DebugExposeTexture(TextureHandle texture, uint32_t unusedBindingSlot) = 0;
    };

} // namespace rhi
