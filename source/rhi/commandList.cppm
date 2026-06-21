module;
#include <cstdint>
#include <string_view>

export module rhi:commandList;
import :types;
import :handles;
import :descriptors;
import :pipeline;
import :sync;
import :resources;
import :raytracing;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // ICommandList
    //
    // Records GPU commands for a single queue submission.
    // Not thread-safe — create one per thread if recording in parallel.
    //
    // Lifetime:
    //   auto cmd = device.CreateCommandList(QueueType::Graphics);
    //   cmd->Begin();
    //   ...
    //   cmd->End();
    //   device.Submit(*cmd);
    //
    // Do NOT reuse a command list while the GPU is still executing it.
    // Use FenceHandle from Submit() to know when it's safe to reset.
    // ---------------------------------------------------------------------------

    class ICommandList
    {
      public:
        virtual ~ICommandList() = default;

        // ---- Lifecycle ----

        virtual void Begin() = 0;
        virtual void End()   = 0;

        // ---- Barriers / transitions ----
        //
        // Explicit state transitions. The caller is responsible for tracking
        // current resource state. The backend does NOT insert implicit transitions.
        //
        // Batch barriers when possible — Submit them together and call
        // FlushBarriers() once; the backend may coalesce them into one
        // vkCmdPipelineBarrier2 call.

        virtual void Transition(const TextureBarrier &barrier)               = 0;
        virtual void Transition(const BufferBarrier &barrier)                = 0;
        virtual void Transition(const MemoryBarrier &barrier)                = 0;
        virtual void Transition(const AccelerationStructureBarrier &barrier) = 0;

        // Convenience: transition a whole texture in one call
        void Transition(TextureHandle tex, ResourceState before, ResourceState after)
        {
            Transition(TextureBarrier{.Texture = tex, .Before = before, .After = after});
        }

        void Transition(BufferHandle buf, ResourceState before, ResourceState after)
        {
            Transition(BufferBarrier{.Buffer = buf, .Before = before, .After = after});
        }

        void Transition(AccelerationStructureHandle as, ResourceState before, ResourceState after)
        {
            Transition(AccelerationStructureBarrier{.AccelerationStructure = as, .Before = before, .After = after});
        }

        // Flush all enqueued barriers (call after batching transitions)
        virtual void FlushBarriers() = 0;

        // ---- Dynamic rendering (no render passes) ----
        //
        // Maps to vkCmdBeginRendering / vkCmdEndRendering (Vulkan 1.3 core)
        //        MTLRenderCommandEncoder on Metal.
        //
        // Color attachments must be in ResourceState::RenderTarget before BeginRendering.
        // Depth attachment must be in ResourceState::DepthWrite / DepthRead.

        virtual void BeginRendering(const RenderingDesc &desc) = 0;
        virtual void EndRendering()                            = 0;

        virtual void SetViewport(const Viewport &vp)    = 0;
        virtual void SetScissor(const Scissor &scissor) = 0;

        // ---- Pipeline ----

        virtual void SetPipeline(PipelineHandle pipeline) = 0;

        // ---- Texture/sampler binding ----
        //
        // Binds a texture or sampler to a fixed, explicit shader slot —
        // matching a shader-declared `Texture2D tex : register(t<index>);`
        // / `SamplerState smp : register(s<index>);` (NOT a
        // DescriptorHandle<Texture2D>/<SamplerState> field embedded in a
        // push-constant struct). Each kernel author picks the index to
        // match their own shader source; call once per dispatch/draw,
        // after SetPipeline and before Dispatch/Draw.
        //
        // Metal: backed by MTL4::ArgumentTable::setTexture/setSamplerState.
        // This is separate from the preferred bindless texture path, where a
        // DescriptorHandle<Texture2D> carries the resident texture's native
        // resource ID in root/scene data. Vulkan uses its global descriptor
        // buffer for DescriptorHandle<T>; these explicit calls are a no-op.
        virtual void BindTexture(TextureHandle texture, uint32_t index) = 0;
        virtual void BindSampler(SamplerHandle sampler, uint32_t index) = 0;

        // ---- Push constants (bindless root data) ----
        //
        // LightRHI uses a single push constant block for all per-draw root data:
        //   GpuAddress, TextureHandle indices, material IDs, etc.
        //
        //   struct MyRootConstants {
        //       GpuAddress  vertices;
        //       TextureHandle albedo;
        //       uint32_t     materialId;
        //   };
        //   cmd->SetPushConstants(MyRootConstants{...});

        virtual void SetPushConstants(const void *data, uint32_t size, uint32_t offset = 0) = 0;

        template <typename T> void SetPushConstants(const T &data, uint32_t offset = 0)
        {
            SetPushConstants(&data, static_cast<uint32_t>(sizeof(T)), offset);
        }

        // ---- Index buffer ----

        virtual void BindIndexBuffer(BufferHandle buffer, uint64_t offset = 0,
                                     IndexType indexType = IndexType::Uint32) = 0;

        // ---- Draw ----

        virtual void Draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0,
                          uint32_t firstInstance = 0) = 0;

        virtual void DrawIndexed(uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0,
                                 int32_t vertexOffset = 0, uint32_t firstInstance = 0) = 0;

        // GPU-driven draw: argsBuffer holds an array of DrawIndirectArgs
        virtual void DrawIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                                  uint32_t stride = sizeof(DrawIndirectArgs)) = 0;

        virtual void DrawIndexedIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                                         uint32_t stride = sizeof(DrawIndexedIndirectArgs)) = 0;

        // Multi-draw indirect with GPU-side count
        virtual void DrawIndirectCount(BufferHandle argsBuffer, uint64_t argsOffset, BufferHandle countBuffer,
                                       uint64_t countOffset, uint32_t maxDrawCount,
                                       uint32_t stride = sizeof(DrawIndirectArgs)) = 0;

        // ---- Compute ----

        virtual void Dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1)           = 0;
        virtual void DispatchIndirect(BufferHandle argsBuffer, uint64_t argsOffset) = 0;

        // ---- Ray tracing: acceleration structure build ----
        //
        // Records a BLAS or TLAS build into `handle` (already allocated via
        // IDevice::CreateAccelerationStructure — this call only encodes the
        // geometry build, it does not allocate). For TLAS builds, `desc`'s
        // Instances must reference already-built BLAS handles: this call does
        // not order BLAS builds for you — record them first, insert a
        // Transition() barrier (AccelerationStructureWrite -> Read) between
        // BLAS and TLAS builds, then call FlushBarriers().
        //
        // `scratchBuffer` must be at least as large as the BuildScratchSize
        // returned by IDevice::QueryAccelerationStructureBuildSizes(desc).
        virtual void BuildAccelerationStructure(AccelerationStructureHandle      handle,
                                                const AccelerationStructureDesc &desc, BufferHandle scratchBuffer,
                                                uint64_t scratchOffset = 0) = 0;

        // There is deliberately no bindAccelerationStructure() call:
        // acceleration structures are bindless, like everything else in this
        // RHI. Shaders receive them as a
        // DescriptorHandle<RaytracingAccelerationStructure> in push constants
        // — the 8-byte value from IDevice::AccelerationStructureAddress() —
        // exactly like a buffer BDA pointer. See raytracing.cppm and
        // tests/shaders/ray_query_triangle.slang for the shader-side pattern.

        // ---- Copy ----

        virtual void CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &region) = 0;

        virtual void CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &region) = 0;

        virtual void CopyBufferToTexture(BufferHandle src, uint64_t srcOffset, TextureHandle dst,
                                         const TextureCopyRegion &region) = 0;

        virtual void CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &region, BufferHandle dst,
                                         uint64_t dstOffset) = 0;

        virtual void ClearTexture(TextureHandle texture, const ClearColor &color,
                                  const SubresourceRange &range = {}) = 0;

        virtual void ClearDepthTexture(TextureHandle texture, const ClearDepthStencil &clear,
                                       const SubresourceRange &range = {}) = 0;

        virtual void FillBuffer(BufferHandle buffer, uint64_t offset, uint64_t size, uint32_t value) = 0;

        // ---- Debug groups (reflected in GPU debuggers: RenderDoc, PIX, Xcode) ----

        virtual void BeginDebugGroup(std::string_view name, float r = 0.F, float g = 0.8F, float b = 0.F) = 0;
        virtual void EndDebugGroup()                                                                      = 0;
        virtual void InsertDebugLabel(std::string_view label)                                             = 0;

        // Adds otherwise-bindless resources to backend debugger metadata.
        // These calls don't change how shaders access a resource and must use
        // binding slots that the shader itself doesn't declare. Backends that
        // don't need this additional metadata may no-op.
        virtual void DebugExposeBuffer(BufferHandle buffer, uint32_t unusedBindingSlot)              = 0;
        virtual void DebugExposeAccelerationStructure(AccelerationStructureHandle accelerationStructure,
                                                      uint32_t                    unusedBindingSlot) = 0;
        virtual void DebugExposeTexture(TextureHandle texture, uint32_t unusedBindingSlot)           = 0;
    };

} // namespace rhi
