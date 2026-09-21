/**
 * {file} metal_command_list.cpp
 * {brief} Records and submits Metal command lists. Device resource creation is implemented elsewhere.
 */
//
// Targets Metal 4 exclusively — see docs/API_GUIDELINES.md and metal_internal.h's
// header comment. MTL4::ComputeCommandEncoder has no setBytes/setBuffer/
// setTexture/setSamplerState at all ("all binding goes through the argument
// table"); MTL4::RenderCommandEncoder likewise has no setVertexBuffer/
// setFragmentBuffer. Every command list owns one persistent
// MTL4::ArgumentTable: push constants bind via setAddress(kPushConstantSlot).
// Native buffer addresses and texture resource IDs inside that root data are
// fully bindless through the queue residency set. Explicit fixed-slot shaders
// may still use BindTexture/BindSampler. There is also no
// separate blit encoder under MTL4 — copyFromBuffer/copyFromTexture/
// fillBuffer live directly on MTL4::ComputeCommandEncoder, so all copy/clear
// operations here route through the same compute encoder as Dispatch.
//
// Exception: BuildAccelerationStructure uses the CLASSIC (non-MTL4) Metal
// raytracing API via MetalDevice::LegacyQueue() — see that accessor's doc
// comment in metal_internal.h for why (MTL4 acceleration structures require
// real RT hardware and refuse to run on this project's non-RT dev GPU).

module;
// Global module fragment: all system + metal-cpp headers included before
// "module rhi.metal;" to keep libc++ include-guard state consistent and
// avoid __promote_t redefinition with Clang's C++23 module support.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

module lightRHI;
import rhi;
#include "metal_internal.h"

namespace rhi::metal
{

    // ============================================================================
    // Helpers — bytes per pixel (used for CopyBufferToTexture row-pitch calculation)
    // ============================================================================

    static uint64_t bytesPerPixel(Format f) noexcept
    {
        switch (f)
        {
            case Format::R8Unorm:
                return 1;
            case Format::RG8Unorm:
                return 2;
            case Format::RGBA8Unorm:
            case Format::RGBA8Srgb:
            case Format::BGRA8Unorm:
            case Format::BGRA8Srgb:
            case Format::R32Float:
            case Format::R32Uint:
            case Format::D32Float:
            case Format::D16Unorm:
                return 4;
            case Format::RG32Float:
            case Format::RGBA16Float:
                return 8;
            case Format::RGBA32Float:
                return 16;
            default:
                return 4; // safe fallback
        }
    }

    // ============================================================================
    // MetalCommandList
    // ============================================================================

    class MetalCommandList final : public ICommandList
    {
        /** Restricts construction to Create(), while still letting it use std::make_unique. */
        struct ConstructionToken
        {
            explicit ConstructionToken() = default;
        };

      public:
        // Push constants and texture/sampler binding both go through this
        // command list's own MTL4::ArgumentTable — see this file's header
        // comment. kPushConstantSlot matches Slang's [[vk::push_constant]]
        // -> [[buffer(30)]] convention (unchanged from before MTL4).
        static constexpr NS::UInteger kPushConstantSlot{30};
        static constexpr NS::UInteger kMaxTextureBinds{32};
        static constexpr NS::UInteger kMaxSamplerBinds{16};

        static constexpr uint64_t kPushConstantSliceSize{256};
        static constexpr uint64_t kPushConstantSlicesPerBlock{64};
        static constexpr uint64_t kPushConstantBlockSize{kPushConstantSliceSize * kPushConstantSlicesPerBlock};

        /**
         * Creates a command list ready to record, or reports why the device cannot provide one.
         * Everything recording needs exists before the list does, so no recording call checks for it.
         */
        [[nodiscard]] static std::expected<std::unique_ptr<MetalCommandList>, DeviceError>
        Create(MetalDevice &device, QueueType queueType, std::string_view debugName)
        {
            auto resources{device.AcquireCommandResources()};
            if (!resources)
            {
                return std::unexpected{resources.error()};
            }

            auto tableDesc{NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init())};
            tableDesc->setMaxBufferBindCount(kPushConstantSlot + 1);
            tableDesc->setMaxTextureBindCount(kMaxTextureBinds);
            tableDesc->setMaxSamplerStateBindCount(kMaxSamplerBinds);
            if (device.DebugCaptureEnabled())
            {
                tableDesc->setInitializeBindings(true);
                if (!debugName.empty())
                {
                    const std::string label{std::string{debugName} + " / debug resource table"};
                    tableDesc->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
                }
            }
            NS::Error *err{nullptr};
            auto       argTable{adoptCreated(device.MtlDevice().newArgumentTable(tableDesc.get(), &err),
                                             "MTLDevice::newArgumentTable refused the table", err)};
            if (!argTable)
            {
                return std::unexpected{argTable.error()};
            }

            // Created last, so a refusal above has no buffer to give back.
            auto firstBlock{_createPushConstantBlock(device)};
            if (!firstBlock)
            {
                return std::unexpected{firstBlock.error()};
            }

            return std::make_unique<MetalCommandList>(ConstructionToken{}, device, std::move(*resources),
                                                      std::move(*argTable), *firstBlock, queueType, debugName);
        }

        MetalCommandList(ConstructionToken, MetalDevice &device, MetalCommandResources &&resources,
                         NS::SharedPtr<MTL4::ArgumentTable> argTable, BufferHandle firstPushConstantBlock,
                         QueueType queueType, std::string_view debugName)
            : _device{device}, _allocator{std::move(resources.Allocator)}, _cmd{std::move(resources.CommandBuffer)},
              _argTable{std::move(argTable)}, _pushConstantBlocks{firstPushConstantBlock}, _debugName{debugName},
              _queueType{queueType}
        {
            if (!_debugName.empty())
            {
                _cmd->setLabel(NS::String::string(_debugName.c_str(), NS::UTF8StringEncoding));
            }
            _allocator->reset();
            _cmd->beginCommandBuffer(_allocator.get());
        }

        ~MetalCommandList() override
        {
            _endActiveEncoder();
            _endCommandBuffer();
            for (const BufferHandle block : _pushConstantBlocks)
            {
                _device.DestroyBuffer(block);
            }
            _device.RecycleCommandResources(MetalCommandResources{
                .Allocator       = std::move(_allocator),
                .CommandBuffer   = std::move(_cmd),
                .CompletionFence = _completionFence,
            });
        }

        // ---- Lifecycle ----

        void Begin() override
        { /* MTL4::CommandBuffer is opened (beginCommandBuffer) immediately after creation */
        }
        void End() override
        {
            _endActiveEncoder();
            _endCommandBuffer();
        }

        void Present(ExternalTextureProviderHandle target) override
        {
            // Recorded, not performed: Metal 4 shows a drawable by signalling it on the queue after
            // the work that drew it, so the present happens where the commit does. The handle is
            // kept rather than an object, and resolved through the device at commit.
            _presentTarget = target;
        }

        /** The target a recorded present named, cleared as it is taken. */
        [[nodiscard]] IFramePresenter *takeRecordedPresentTarget() noexcept
        {
            return _device.ResolveExternalTextureProvider(
                std::exchange(_presentTarget, ExternalTextureProviderHandle{}));
        }

        // ---- Barriers ----
        //
        // Coarse, conservative barriers (StageAll <-> StageAll) — matching the
        // classic-Metal implementation this replaces, which also used
        // blanket buffer/texture barrier scopes rather than fine-grained
        // per-resource tracking.

        void Transition(const TextureBarrier &b) override
        {
            _enqueueBarrier(b.Before, b.After);
        }

        void Transition(const BufferBarrier &b) override
        {
            _enqueueBarrier(b.Before, b.After);
        }

        void Transition(const MemoryBarrier &b) override
        {
            // A global memory barrier must be at least as strong as any resource one, so it takes
            // the widest scope regardless of the states named.
            _enqueueBarrier(b.Before, b.After, MTL::StageAll, MTL::StageAll);
        }

        void Transition(const AccelerationStructureBarrier &b) override
        {
            // Metal has no separate acceleration-structure barrier scope; StageAll also covers
            // them for read-after-write ordering between builds (BLAS build -> TLAS build).
            _enqueueBarrier(b.Before, b.After, MTL::StageAll, MTL::StageAll);
        }

        void FlushBarriers() override
        {
            _recordPendingBarrier();
        }

        // ---- Dynamic rendering ----

        void BeginRendering(const RenderingDesc &desc) override
        {
            _endActiveEncoder();

            auto rpd{NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init())};

            for (NS::UInteger i{0}; i < static_cast<NS::UInteger>(desc.Color.size()); ++i)
            {
                const auto &ca{desc.Color[i]};
                auto       *att{rpd->colorAttachments()->object(i)};
                att->setTexture(&_lookupTexture(ca.Texture));
                att->setLoadAction(_toLoadAction(ca.LoadOp));
                att->setStoreAction(_toStoreAction(ca.StoreOp));
                att->setClearColor(MTL::ClearColor::Make((double)ca.ClearValue.R, (double)ca.ClearValue.G,
                                                         (double)ca.ClearValue.B, (double)ca.ClearValue.A));
                if (ca.ResolveTexture.Valid())
                {
                    att->setResolveTexture(&_lookupTexture(ca.ResolveTexture));
                    att->setStoreAction(MTL::StoreActionMultisampleResolve);
                }
            }

            if (desc.Depth.Texture.Valid())
            {
                auto *d{rpd->depthAttachment()};
                d->setTexture(&_lookupTexture(desc.Depth.Texture));
                d->setLoadAction(_toLoadAction(desc.Depth.LoadOp));
                d->setStoreAction(_toStoreAction(desc.Depth.StoreOp));
                d->setClearDepth((double)desc.Depth.ClearValue.Depth);
            }

            const auto &render{_encoder.emplace<RenderEncoder>(NS::RetainPtr(_cmd->renderCommandEncoder(rpd.get())))};
            // A barrier flushed outside a pass belongs to the first work that could see
            // what it orders, which is this pass.
            _recordPendingBarrier();
            render->setArgumentTable(_argTable.get(), MTL::RenderStageVertex | MTL::RenderStageFragment);
        }

        void EndRendering() override
        {
            if (std::holds_alternative<RenderEncoder>(_encoder))
            {
                _endActiveEncoder();
            }
        }

        void SetViewport(const Viewport &vp) override
        {
            _renderEncoder("SetViewport")
                .setViewport(MTL::Viewport{.originX = (double)vp.X,
                                           .originY = (double)vp.Y,
                                           .width   = (double)vp.Width,
                                           .height  = (double)vp.Height,
                                           .znear   = (double)vp.MinDepth,
                                           .zfar    = (double)vp.MaxDepth});
        }

        void SetScissor(const Scissor &s) override
        {
            _renderEncoder("SetScissor")
                .setScissorRect(MTL::ScissorRect{.x      = static_cast<NS::UInteger>(s.X),
                                                 .y      = static_cast<NS::UInteger>(s.Y),
                                                 .width  = s.Width,
                                                 .height = s.Height});
        }

        // ---- Pipeline ----

        void SetPipeline(PipelineHandle handle) override
        {
            if (!handle.Valid())
            {
                return;
            }
            auto &p{_device.Pipeline(handle)};
            if (p.isCompute)
            {
                auto &compute{_computeEncoder()};
                if (auto *label{p.computePso->label()})
                {
                    compute.setLabel(label);
                }
                compute.setComputePipelineState(p.computePso.get());
                _tgX = p.threadGroupSizeX;
                _tgY = p.threadGroupSizeY;
                _tgZ = p.threadGroupSizeZ;
            }
            else
            {
                auto &render{_renderEncoder("SetPipeline(graphics)")};
                render.setRenderPipelineState(p.renderPso.get());
                if (p.depthStencilState)
                {
                    render.setDepthStencilState(p.depthStencilState.get());
                }
                render.setFrontFacingWinding(p.winding);
                render.setCullMode(p.cullMode);
                render.setTriangleFillMode(p.fillMode);
                if (p.depthBiasConstant != 0.F || p.depthBiasSlope != 0.F)
                {
                    render.setDepthBias(p.depthBiasConstant, p.depthBiasSlope, 0.F);
                }
            }
        }

        // ---- Texture/sampler binding ----

        void BindTexture(TextureHandle texture, uint32_t index) override
        {
            if (!texture.Valid())
            {
                return;
            }
            _argTable->setTexture(_device.Texture(texture).texture->gpuResourceID(), index);
        }

        void BindSampler(SamplerHandle sampler, uint32_t index) override
        {
            if (!sampler.Valid())
            {
                return;
            }
            _argTable->setSamplerState(_device.Sampler(sampler).state->gpuResourceID(), index);
        }

        // ---- Push constants ----

        void SetPushConstants(std::span<const std::byte> data, uint32_t offset) override
        {
            if (data.size_bytes() + offset > kPushConstantSliceSize)
            {
                FailContract("push constants exceed the per-command scratch slice");
            }
            if (_pushConstantCursor == kPushConstantBlockSize)
            {
                auto block{_createPushConstantBlock(_device)};
                if (!block)
                {
                    // Reported where it was refused. The update is dropped rather than written past
                    // the end of the full block.
                    return;
                }
                _pushConstantBlocks.push_back(*block);
                _pushConstantCursor = 0;
            }

            const std::span<std::byte> slice{_device.MapBuffer(_pushConstantBlocks.back())
                                                 .Bytes()
                                                 .subspan(_pushConstantCursor, kPushConstantSliceSize)};
            // A caller may update part of the root data and expect the rest to stand, so a new
            // slice starts as a copy of the one in force before it.
            if (_pushConstantCursor > 0)
            {
                std::memcpy(slice.data(), slice.data() - kPushConstantSliceSize, kPushConstantSliceSize);
            }
            else if (_pushConstantBlocks.size() > 1)
            {
                const std::span<std::byte> previous{
                    _device.MapBuffer(_pushConstantBlocks[_pushConstantBlocks.size() - 2])
                        .Bytes()
                        .last(kPushConstantSliceSize)};
                std::memcpy(slice.data(), previous.data(), kPushConstantSliceSize);
            }
            std::memcpy(slice.data() + offset, data.data(), data.size_bytes());

            _argTable->setAddress(_device.BufferAddress(_pushConstantBlocks.back()).Address + _pushConstantCursor,
                                  kPushConstantSlot);
            _pushConstantCursor += kPushConstantSliceSize;
        }

        // ---- Index buffer ----

        void BindIndexBuffer(BufferHandle buf, uint64_t offset, IndexType type) override
        {
            _indexBuffer = buf;
            _indexOffset = offset;
            _indexType   = type;
        }

        // ---- Draw ----

        void Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override
        {
            _renderEncoder("Draw").drawPrimitives(_primitiveType, firstVertex, vertexCount, instanceCount,
                                                  firstInstance);
        }

        void DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                         uint32_t firstInstance) override
        {
            const uint32_t stride{_indexType == IndexType::Uint16 ? 2U : 4U};
            auto           idxType{_indexType == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32};
            const auto     indexStart{_indexOffset + static_cast<uint64_t>(firstIndex) * stride};
            const MTL::GPUAddress addr{_device.BufferAddress(_indexBuffer).Address + indexStart};
            auto                  len{static_cast<NS::UInteger>(_device.GetBufferInfo(_indexBuffer).Size - indexStart)};
            _renderEncoder("DrawIndexed")
                .drawIndexedPrimitives(_primitiveType, indexCount, idxType, addr, len, instanceCount, vertexOffset,
                                       static_cast<NS::UInteger>(firstInstance));
        }

        void DrawIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                          uint32_t /*stride*/) override
        {
            const auto base{_device.BufferAddress(argsBuffer).Address};
            auto      &render{_renderEncoder("DrawIndirect")};
            for (uint32_t i{0}; i < drawCount; ++i)
            {
                render.drawPrimitives(_primitiveType, base + argsOffset + i * sizeof(DrawIndirectArgs));
            }
        }

        void DrawIndexedIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                                 uint32_t /*stride*/) override
        {
            auto idxType{_indexType == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32};
            const MTL::GPUAddress idxAddr{_device.BufferAddress(_indexBuffer).Address + _indexOffset};
            auto       idxLen{static_cast<NS::UInteger>(_device.GetBufferInfo(_indexBuffer).Size - _indexOffset)};
            const auto argBase{_device.BufferAddress(argsBuffer).Address};
            auto      &render{_renderEncoder("DrawIndexedIndirect")};
            for (uint32_t i{0}; i < drawCount; ++i)
            {
                render.drawIndexedPrimitives(_primitiveType, idxType, idxAddr, idxLen,
                                             argBase + argsOffset + i * sizeof(DrawIndexedIndirectArgs));
            }
        }

        void DrawIndirectCount(BufferHandle, uint64_t, BufferHandle, uint64_t, uint32_t, uint32_t) override
        {
            // True GPU-driven multi-draw needs MTLIndirectCommandBuffer.
            // Not implemented; use DrawIndirect with a fixed draw count.
        }

        // ---- Compute ----

        void Dispatch(uint32_t x, uint32_t y, uint32_t z) override
        {
            _computeEncoder().dispatchThreadgroups(MTL::Size::Make(x, y, z), MTL::Size::Make(_tgX, _tgY, _tgZ));
        }

        void DispatchIndirect(BufferHandle argsBuffer, uint64_t argsOffset) override
        {
            const MTL::GPUAddress addr{_device.BufferAddress(argsBuffer).Address + argsOffset};
            _computeEncoder().dispatchThreadgroups(addr, MTL::Size::Make(_tgX, _tgY, _tgZ));
        }

        // ---- Ray tracing ----

        void BuildAccelerationStructure(AccelerationStructureHandle handle, const AccelerationStructureDesc &desc,
                                        BufferHandle scratchBuffer, uint64_t scratchOffset) override
        {
            // Deliberate exception to the MTL4-everywhere design of this
            // file: acceleration structures build via the CLASSIC Metal
            // raytracing API, on a separate classic MTL::CommandQueue
            // (MetalDevice::LegacyQueue()) — see that accessor's doc comment
            // in metal_internal.h. Classic MTL4::CommandBuffer has no
            // accelerationStructureCommandEncoder() at all, so this can't
            // live on this command list's own MTL4 command buffer; it's
            // submitted and waited on synchronously instead; the descriptor
            // and its auxiliary objects (e.g. a TLAS instance descriptor
            // buffer written host-side) only need to stay alive until that
            // wait returns, not until this command list's own Submit().
            std::vector<NS::SharedPtr<NS::Object>> keepAlive{};
            auto descriptor{_device.MakeAccelerationStructureDescriptor(desc, keepAlive)};

            auto *legacyCmd{_device.LegacyQueue().commandBuffer()};
            auto *enc{legacyCmd->accelerationStructureCommandEncoder()};
            if (!desc.DebugName.empty())
            {
                auto *label{MakeLabel(desc.DebugName)};
                legacyCmd->setLabel(label);
                enc->setLabel(label);
            }
            enc->buildAccelerationStructure(_device.AccelStruct(handle).as.get(), descriptor.get(),
                                            &_lookupBuffer(scratchBuffer), scratchOffset);
            enc->endEncoding();

            // Suspended around the synchronous cross-queue wait below — see
            // SuspendActiveCaptureScope's doc comment in metal_internal.h.
            _device.SuspendActiveCaptureScope();
            legacyCmd->commit();
            legacyCmd->waitUntilCompleted();
            _device.ResumeActiveCaptureScope();
        }

        // ---- Copy ----
        //
        // MTL4 has no blit encoder — copyFromBuffer/copyFromTexture/fillBuffer
        // live directly on MTL4::ComputeCommandEncoder, so these all share
        // the same compute encoder Dispatch() uses.

        void CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &region) override
        {
            _computeEncoder().copyFromBuffer(&_lookupBuffer(src), region.SrcOffset, &_lookupBuffer(dst),
                                             region.DstOffset, region.Size);
        }

        void CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &region) override
        {
            _computeEncoder().copyFromTexture(
                &_lookupTexture(src), region.ArrayLayer, region.MipLevel, MTL::Origin::Make(0, 0, 0),
                MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth), &_lookupTexture(dst),
                region.ArrayLayer, region.MipLevel,
                MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                  static_cast<NS::UInteger>(region.DstOffset.Y),
                                  static_cast<NS::UInteger>(region.DstOffset.Z)));
        }

        void CopyBufferToTexture(BufferHandle src, uint64_t srcOffset, TextureHandle dst,
                                 const TextureCopyRegion &region) override
        {
            auto          &t{_device.Texture(dst)};
            const uint64_t bpp{bytesPerPixel(t.desc.Format)};
            const uint64_t rowP{bpp * region.Extent.Width};
            const uint64_t sliceP{rowP * region.Extent.Height};
            _computeEncoder().copyFromBuffer(
                &_lookupBuffer(src), srcOffset, rowP, sliceP,
                MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth), &_lookupTexture(dst),
                region.ArrayLayer, region.MipLevel,
                MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                  static_cast<NS::UInteger>(region.DstOffset.Y),
                                  static_cast<NS::UInteger>(region.DstOffset.Z)));
        }

        void CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &region, BufferHandle dst,
                                 uint64_t dstOffset) override
        {
            auto          &t{_device.Texture(src)};
            const uint64_t bpp{bytesPerPixel(t.desc.Format)};
            const uint64_t rowP{bpp * region.Extent.Width};
            const uint64_t sliceP{rowP * region.Extent.Height};
            _computeEncoder().copyFromTexture(
                &_lookupTexture(src), region.ArrayLayer, region.MipLevel,
                MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                  static_cast<NS::UInteger>(region.DstOffset.Y),
                                  static_cast<NS::UInteger>(region.DstOffset.Z)),
                MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth), &_lookupBuffer(dst),
                dstOffset, rowP, sliceP);
        }

        void ClearTexture(TextureHandle handle, const ClearColor &color, const SubresourceRange & /*range*/) override
        {
            // Metal has no standalone clear — use a transient render pass with loadAction=clear.
            _endActiveEncoder();
            auto  rpd{NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init())};
            auto *att{rpd->colorAttachments()->object(0)};
            att->setTexture(&_lookupTexture(handle));
            att->setLoadAction(MTL::LoadActionClear);
            att->setStoreAction(MTL::StoreActionStore);
            att->setClearColor(
                MTL::ClearColor::Make((double)color.R, (double)color.G, (double)color.B, (double)color.A));
            auto *enc{_cmd->renderCommandEncoder(rpd.get())};
            enc->endEncoding();
        }

        void ClearDepthTexture(TextureHandle handle, const ClearDepthStencil &clear,
                               const SubresourceRange & /*range*/) override
        {
            _endActiveEncoder();
            auto  rpd{NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init())};
            auto *d{rpd->depthAttachment()};
            d->setTexture(&_lookupTexture(handle));
            d->setLoadAction(MTL::LoadActionClear);
            d->setStoreAction(MTL::StoreActionStore);
            d->setClearDepth((double)clear.Depth);
            auto *enc{_cmd->renderCommandEncoder(rpd.get())};
            enc->endEncoding();
        }

        void FillBuffer(BufferHandle buf, uint64_t offset, uint64_t size, uint32_t value) override
        {
            // fillBuffer fills with a single byte value. For multi-byte
            // fill, a compute kernel is needed; we use the low byte for now.
            _computeEncoder().fillBuffer(&_lookupBuffer(buf), NS::Range::Make(offset, size),
                                         static_cast<uint8_t>(value & 0xFF));
        }

        void WriteComputeTimestamp(TimestampQueryPoolHandle pool, uint32_t index) override
        {
            const auto &record{_device.TimestampQueryPool(pool)};
            assert(index < record.count);
            _computeEncoder().writeTimestamp(MTL4::TimestampGranularityPrecise, record.heap.get(), index);
        }

        // ---- Debug ----

        void BeginDebugGroup(std::string_view name, float, float, float) override
        {
            auto *ns{MakeLabel(name)};
            _pushDebugGroup(ns);
        }

        void EndDebugGroup() override
        {
            _popDebugGroup();
        }

        void InsertDebugLabel(std::string_view label) override
        {
            auto *ns{MakeLabel(label)};
            if (auto *encoder{_activeEncoder()})
            {
                encoder->insertDebugSignpost(ns);
            }
        }

        void DebugExposeBuffer(BufferHandle buffer, uint32_t unusedBindingSlot) override
        {
            if (!buffer.Valid())
            {
                return;
            }
            _checkDebugBufferSlot(unusedBindingSlot);
            _argTable->setAddress(_device.BufferAddress(buffer).Address, unusedBindingSlot);
        }

        void DebugExposeAccelerationStructure(AccelerationStructureHandle accelerationStructure,
                                              uint32_t                    unusedBindingSlot) override
        {
            if (!accelerationStructure.Valid())
            {
                return;
            }
            _checkDebugBufferSlot(unusedBindingSlot);
            _argTable->setResource(_device.AccelStruct(accelerationStructure).as->gpuResourceID(), unusedBindingSlot);
        }

        void DebugExposeTexture(TextureHandle texture, uint32_t unusedBindingSlot) override
        {
            if (!texture.Valid())
            {
                return;
            }
            if (unusedBindingSlot >= kMaxTextureBinds)
            {
                FailContract("DebugExposeTexture: binding slot " + std::to_string(unusedBindingSlot) +
                             " is out of range (max " + std::to_string(kMaxTextureBinds) + ")");
            }
            _argTable->setTexture(_device.Texture(texture).texture->gpuResourceID(), unusedBindingSlot);
        }

        // ---- Internal accessor for MetalDevice::Submit ----
        [[nodiscard]] MTL4::CommandBuffer *commandBuffer() const noexcept
        {
            return _cmd.get();
        }

        [[nodiscard]] std::string_view debugName() const noexcept
        {
            return _debugName;
        }

        void setCompletionFence(FenceHandle fence) noexcept
        {
            _completionFence = fence;
        }

      private:
        // Used only by this class, which is what the leading underscore has always meant.
        /**
         * Creates one block of root-data scratch. It goes through the public IDevice API so it joins
         * the device's persistent MTL::ResidencySet, which MetalDevice::CreateBuffer already does.
         */
        [[nodiscard]] static std::expected<BufferHandle, DeviceError> _createPushConstantBlock(MetalDevice &device)
        {
            auto block{device.CreateBuffer(BufferDesc{
                .Size = kPushConstantBlockSize, .Usage = BufferUsage::Storage, .MemoryType = MemoryType::CpuToGpu})};
            if (!block)
            {
                ReportRefusal("the device refused the root-data scratch for a command list");
            }
            return block;
        }
        using RenderEncoder  = NS::SharedPtr<MTL4::RenderCommandEncoder>;
        using ComputeEncoder = NS::SharedPtr<MTL4::ComputeCommandEncoder>;

        MetalDevice                          &_device;
        NS::SharedPtr<MTL4::CommandAllocator> _allocator{};
        NS::SharedPtr<MTL4::CommandBuffer>    _cmd{};
        NS::SharedPtr<MTL4::ArgumentTable>    _argTable{};
        // Metal allows one active encoder per command buffer, so there is exactly one of these states.
        std::variant<std::monostate, RenderEncoder, ComputeEncoder> _encoder{};
        std::vector<BufferHandle>                                   _pushConstantBlocks{}; // never empty
        uint64_t                                                    _pushConstantCursor{0};
        std::string                                                 _debugName{};
        FenceHandle                                                 _completionFence{};
        /// The surface a recorded Present() named, shown by the submission that carries this list.
        ExternalTextureProviderHandle _presentTarget{};
        bool                          _cmdEnded{false};
        [[maybe_unused]] QueueType    _queueType{QueueType::Graphics};

        // ---- Deferred barriers ----
        //
        // Transition() records what has to be ordered and FlushBarriers() writes it down, which is
        // what this library's ICommandList promises. Metal's barriers are stage-scoped rather than
        // resource-scoped, so what accumulates is the union of the stages that must finish and the
        // stages that must wait, not one entry per resource.
        //
        // A flush can land outside an active pass - between a render pass and a copy, say, which
        // is exactly where a readback needs ordering. Metal has nowhere to write a barrier then, so
        // it stays pending and is recorded by the next encoder that opens. That encoder is the
        // first thing that could observe the work being ordered, so recording it there is both
        // sufficient and the earliest it can be done.
        MTL::Stages _pendingAfterStages{};  ///< Stages that must complete before the barrier.
        MTL::Stages _pendingBeforeStages{}; ///< Stages that must wait for it.
        bool        _barrierPending{false};
        bool        _pendingBarrierConsumesPriorPass{false};

        MTL::PrimitiveType _primitiveType{MTL::PrimitiveTypeTriangle};
        NS::UInteger       _tgX{64}, _tgY{1}, _tgZ{1};

        BufferHandle _indexBuffer{};
        uint64_t     _indexOffset{0};
        IndexType    _indexType{IndexType::Uint32};

        // ---- Encoder management ----

        /** The encoder for the active render or compute pass, or null outside a pass. */
        [[nodiscard]] MTL4::CommandEncoder *_activeEncoder() const noexcept
        {
            if (const auto *render{std::get_if<RenderEncoder>(&_encoder)})
            {
                return render->get();
            }
            if (const auto *compute{std::get_if<ComputeEncoder>(&_encoder)})
            {
                return compute->get();
            }
            return nullptr;
        }

        /** The Metal stages that read or write a resource in `state`. */
        [[nodiscard]] static MTL::Stages _stagesFor(ResourceState state) noexcept
        {
            NS::UInteger stages{0};
            if (HasState(state, ResourceState::RenderTarget) || HasState(state, ResourceState::DepthWrite) ||
                HasState(state, ResourceState::DepthRead) || HasState(state, ResourceState::Present))
            {
                stages |= MTL::StageFragment | MTL::StageVertex;
            }
            if (HasState(state, ResourceState::ShaderRead) || HasState(state, ResourceState::UnorderedAccess) ||
                HasState(state, ResourceState::ConstantBuffer) || HasState(state, ResourceState::VertexBuffer) ||
                HasState(state, ResourceState::IndexBuffer) || HasState(state, ResourceState::IndirectArgument))
            {
                stages |= MTL::StageDispatch | MTL::StageFragment | MTL::StageVertex;
            }
            // MTL4 records copies on the unified compute encoder but executes them in its blit
            // stage. Do not also block independent dispatch work.
            if (HasState(state, ResourceState::TransferSrc) || HasState(state, ResourceState::TransferDst) ||
                HasState(state, ResourceState::CopySrc) || HasState(state, ResourceState::CopyDst))
            {
                stages |= MTL::StageBlit;
            }
            if (HasState(state, ResourceState::AccelerationStructureRead) ||
                HasState(state, ResourceState::AccelerationStructureWrite))
            {
                stages |= MTL::StageAccelerationStructure | MTL::StageDispatch;
            }
            // Undefined, or a state this backend does not distinguish: order against everything
            // rather than silently against nothing.
            return stages == 0 ? MTL::StageAll : static_cast<MTL::Stages>(stages);
        }

        /** Records that `before` must finish before `after` may begin. */
        void _enqueueBarrier(ResourceState before, ResourceState after, MTL::Stages afterOverride = {},
                             MTL::Stages beforeOverride = {})
        {
            const NS::UInteger produced{afterOverride != 0 ? afterOverride : _stagesFor(before)};
            const NS::UInteger consumed{beforeOverride != 0 ? beforeOverride : _stagesFor(after)};
            _pendingAfterStages              = static_cast<MTL::Stages>(_pendingAfterStages | produced);
            _pendingBeforeStages             = static_cast<MTL::Stages>(_pendingBeforeStages | consumed);
            _barrierPending                  = true;
            _pendingBarrierConsumesPriorPass = _activeEncoder() == nullptr;
        }

        /**
         * Writes any pending barrier into the active pass.
         *
         * Does nothing outside a pass: the barrier stays pending and is written into the next
         * active pass, which is the first place it could matter.
         */
        void _recordPendingBarrier()
        {
            if (!_barrierPending)
            {
                return;
            }
            auto *encoder{_activeEncoder()};
            if (encoder == nullptr)
            {
                return; // outside a pass; the next active pass will record it
            }
            if (_pendingBarrierConsumesPriorPass)
            {
                // A transition recorded between passes belongs at the consumer. This orders prior
                // passes without blocking unrelated stages in the producing pass.
                encoder->barrierAfterQueueStages(_pendingAfterStages, _pendingBeforeStages,
                                                 MTL4::VisibilityOptionDevice);
            }
            else
            {
                // Use the smallest scope for a dependency within one pass. Metal only accepts
                // stages that the current encoder can execute for an intra-pass barrier.
                const MTL::Stages supportedStages{std::holds_alternative<RenderEncoder>(_encoder)
                                                      ? static_cast<MTL::Stages>(MTL::StageVertex | MTL::StageFragment)
                                                      : static_cast<MTL::Stages>(MTL::StageDispatch | MTL::StageBlit |
                                                                                 MTL::StageAccelerationStructure)};
                const auto        intraAfter{static_cast<MTL::Stages>(_pendingAfterStages & supportedStages)};
                const auto        intraBefore{static_cast<MTL::Stages>(_pendingBeforeStages & supportedStages)};
                if (intraAfter != 0 && intraBefore != 0)
                {
                    encoder->barrierAfterEncoderStages(intraAfter, intraBefore, MTL4::VisibilityOptionDevice);
                }
                else
                {
                    // The consumer stage cannot run in this pass, so this is a dependency on a
                    // later pass even though the caller flushed before ending the current one.
                    encoder->barrierAfterStages(_pendingAfterStages, _pendingBeforeStages,
                                                MTL4::VisibilityOptionDevice);
                }
            }
            _pendingAfterStages              = {};
            _pendingBeforeStages             = {};
            _barrierPending                  = false;
            _pendingBarrierConsumesPriorPass = false;
        }

        void _endActiveEncoder()
        {
            if (auto *encoder{_activeEncoder()})
            {
                if (_barrierPending && !_pendingBarrierConsumesPriorPass)
                {
                    // No command in this pass consumed the queued transition. It therefore names
                    // a dependency on a later pass, for which Metal's producer barrier is exact.
                    encoder->barrierAfterStages(_pendingAfterStages, _pendingBeforeStages,
                                                MTL4::VisibilityOptionDevice);
                    _pendingAfterStages  = {};
                    _pendingBeforeStages = {};
                    _barrierPending      = false;
                }
                encoder->endEncoding();
            }
            _encoder = std::monostate{};
        }

        /** The open render encoder. Recording `call` outside BeginRendering() breaks the contract. */
        [[nodiscard]] MTL4::RenderCommandEncoder &_renderEncoder(std::string_view call)
        {
            auto *render{std::get_if<RenderEncoder>(&_encoder)};
            if (render == nullptr)
            {
                FailContract(std::string{call} + " was recorded outside BeginRendering()");
            }
            _recordPendingBarrier();
            return *render->get();
        }

        void _endCommandBuffer()
        {
            if (!_cmdEnded)
            {
                _cmd->endCommandBuffer();
                _cmdEnded = true;
            }
        }

        /** The open compute encoder, ending any render encoder to open one when needed. */
        [[nodiscard]] MTL4::ComputeCommandEncoder &_computeEncoder()
        {
            if (auto *compute{std::get_if<ComputeEncoder>(&_encoder)})
            {
                _recordPendingBarrier();
                return *compute->get();
            }
            _endActiveEncoder();
            const auto &compute{_encoder.emplace<ComputeEncoder>(NS::RetainPtr(_cmd->computeCommandEncoder()))};
            compute->setArgumentTable(_argTable.get());
            // A barrier flushed outside a pass belongs to the first work that could see
            // what it orders, which is this encoder.
            _recordPendingBarrier();
            return *compute.get();
        }

        // ---- Resource lookups ----

        // A command that names a resource needs one: recording it against an invalid handle is a
        // caller mistake, stopped here rather than recorded as a command that touches nothing.

        [[nodiscard]] MTL::Buffer &_lookupBuffer(BufferHandle h) const
        {
            if (!h.Valid())
            {
                FailContract("a command was recorded against an invalid buffer handle");
            }
            return *_device.Buffer(h).buffer.get();
        }

        [[nodiscard]] MTL::Texture &_lookupTexture(TextureHandle h) const
        {
            if (!h.Valid())
            {
                FailContract("a command was recorded against an invalid texture handle");
            }
            return *_device.Texture(h).texture.get();
        }

        // ---- Debug helpers ----

        // DebugExposeBuffer/DebugExposeAccelerationStructure both bind into
        // the argument table's buffer-slot namespace (shared with push
        // constants at kPushConstantSlot) — this used to be an assert, so an
        // out-of-range slot silently no-op'd the debug exposure in
        // non-assert builds with no signal. Thrown instead so a bad slot is
        // loud in every build configuration.
        void _checkDebugBufferSlot(uint32_t slot) const
        {
            if (slot >= kPushConstantSlot)
            {
                FailContract("Debug resource slot " + std::to_string(slot) +
                             " overlaps or exceeds the push-constant slot (" + std::to_string(kPushConstantSlot) + ")");
            }
        }

        void _pushDebugGroup(NS::String *name)
        {
            if (auto *encoder{_activeEncoder()})
            {
                encoder->pushDebugGroup(name);
            }
            else
            {
                _cmd->pushDebugGroup(name);
            }
        }

        void _popDebugGroup()
        {
            if (auto *encoder{_activeEncoder()})
            {
                encoder->popDebugGroup();
            }
            else
            {
                _cmd->popDebugGroup();
            }
        }

        // ---- Static helpers ----

        [[nodiscard]] static MTL::LoadAction _toLoadAction(LoadOp op) noexcept
        {
            switch (op)
            {
                case LoadOp::Load:
                    return MTL::LoadActionLoad;
                case LoadOp::Clear:
                    return MTL::LoadActionClear;
                case LoadOp::DontCare:
                    return MTL::LoadActionDontCare;
            }
            return MTL::LoadActionDontCare;
        }

        [[nodiscard]] static MTL::StoreAction _toStoreAction(StoreOp op) noexcept
        {
            switch (op)
            {
                case StoreOp::Store:
                    return MTL::StoreActionStore;
                case StoreOp::DontCare:
                    return MTL::StoreActionDontCare;
            }
            return MTL::StoreActionDontCare;
        }
    };

    // ============================================================================
    // MetalDevice methods that reference MetalCommandList
    // ============================================================================

    std::expected<std::unique_ptr<ICommandList>, DeviceError> MetalDevice::CreateCommandList(QueueType        q,
                                                                                             std::string_view name)
    {
        return MetalCommandList::Create(*this, q, name)
            .transform([](std::unique_ptr<MetalCommandList> commands)
                       { return std::unique_ptr<ICommandList>{std::move(commands)}; });
    }

    FenceHandle MetalDevice::Submit(ICommandList &cmdList, const SubmitDesc & /*desc*/)
    {
        auto                      &mcl{static_cast<MetalCommandList &>(cmdList)};
        FenceHandle                fence{NextFence()};
        MTL4::CommandBuffer *const buffers[1]{mcl.commandBuffer()};
        auto                      *captureScope{SubmissionCaptureScope(mcl.debugName())};
        if (captureScope)
        {
            captureScope->beginScope();
        }
        Mtl4Queue().commit(buffers, 1);
        Mtl4Queue().signalEvent(&TimelineEvent(), fence.Id);
        mcl.setCompletionFence(fence);

        // A recorded present is shown by the submission that drew it: Metal 4 orders that on the
        // queue, so it belongs here, after the commit and before this call returns.
        if (auto *target{mcl.takeRecordedPresentTarget()})
        {
            target->PresentHeldFrame();
        }

        if (captureScope)
        {
            captureScope->endScope();
        }
        return fence;
    }

} // namespace rhi::metal
