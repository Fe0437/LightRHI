// metal_command_list.cpp — MetalCommandList + MetalDevice::CreateCommandList / Submit
//
// Targets Metal 4 exclusively — see API_GUIDELINES.md and metal_internal.h's
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
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
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
      public:
        // Push constants and texture/sampler binding both go through this
        // command list's own MTL4::ArgumentTable — see this file's header
        // comment. kPushConstantSlot matches Slang's [[vk::push_constant]]
        // -> [[buffer(30)]] convention (unchanged from before MTL4).
        static constexpr NS::UInteger kPushConstantSlot{30};
        static constexpr NS::UInteger kMaxTextureBinds{32};
        static constexpr NS::UInteger kMaxSamplerBinds{16};
        static constexpr uint64_t     kPushConstantScratchSize{256};

        MetalCommandList(MetalDevice *device, MetalCommandResources &&resources, QueueType queueType,
                         std::string_view debugName)
            : _device{device}, _allocator{std::move(resources.Allocator)}, _cmd{std::move(resources.CommandBuffer)},
              _debugName{debugName}, _queueType{queueType}
        {
            if (!_debugName.empty())
            {
                _cmd->setLabel(NS::String::string(_debugName.c_str(), NS::UTF8StringEncoding));
            }

            // Push-constant scratch buffer — created through the public
            // IDevice API so it's automatically added to the device's
            // persistent MTL::ResidencySet (MetalDevice::CreateBuffer already
            // calls _addResident).
            _pushConstantBuf = _device->CreateBuffer(BufferDesc{
                .Size = kPushConstantScratchSize, .Usage = BufferUsage::Storage, .MemoryType = MemoryType::CpuToGpu});

            auto tableDesc{NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init())};
            tableDesc->setMaxBufferBindCount(kPushConstantSlot + 1);
            tableDesc->setMaxTextureBindCount(kMaxTextureBinds);
            tableDesc->setMaxSamplerStateBindCount(kMaxSamplerBinds);
            if (_device->DebugCaptureEnabled())
            {
                tableDesc->setInitializeBindings(true);
                if (!_debugName.empty())
                {
                    const std::string label{_debugName + " / debug resource table"};
                    tableDesc->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
                }
            }
            NS::Error *err{nullptr};
            _argTable = NS::TransferPtr(_device->MtlDevice()->newArgumentTable(tableDesc.get(), &err));
            if (!_argTable)
            {
                const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
                throw std::runtime_error(std::string{"[LightRHI] MTLDevice::newArgumentTable failed: "} + msg);
            }

            _allocator->reset();
            _cmd->beginCommandBuffer(_allocator.get());
        }

        ~MetalCommandList() override
        {
            _endActiveEncoder();
            _endCommandBuffer();
            _device->DestroyBuffer(_pushConstantBuf);
            _device->RecycleCommandResources(MetalCommandResources{
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

        // ---- Barriers ----
        //
        // Coarse, conservative barriers (StageAll <-> StageAll) — matching the
        // classic-Metal implementation this replaces, which also used
        // blanket buffer/texture barrier scopes rather than fine-grained
        // per-resource tracking.

        void Transition(const TextureBarrier &b) override
        {
            if (_encoderType == EncoderType::Render &&
                (HasState(b.After, ResourceState::ShaderRead) || HasState(b.After, ResourceState::UnorderedAccess)))
            {
                _renderEncoder->barrierAfterStages(MTL::StageFragment, MTL::StageVertex, MTL4::VisibilityOptionDevice);
            }
        }

        void Transition(const BufferBarrier &) override
        {
            if (_encoderType == EncoderType::Compute)
            {
                _computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                                           MTL4::VisibilityOptionDevice);
            }
        }

        void Transition(const MemoryBarrier &) override
        {
            if (_encoderType == EncoderType::Compute)
            {
                _computeEncoder->barrierAfterStages(MTL::StageAll, MTL::StageAll, MTL4::VisibilityOptionDevice);
            }
        }

        void Transition(const AccelerationStructureBarrier &) override
        {
            // Metal has no separate AS resource-barrier scope; StageAll also
            // covers acceleration structures for purposes of read-after-write
            // ordering between encoders (e.g. BLAS build -> TLAS build).
            if (_encoderType == EncoderType::Compute)
            {
                _computeEncoder->barrierAfterStages(MTL::StageAll, MTL::StageAll, MTL4::VisibilityOptionDevice);
            }
        }

        void FlushBarriers() override
        { /* deferred queue not needed on Metal */
        }

        // ---- Dynamic rendering ----

        void BeginRendering(const RenderingDesc &desc) override
        {
            _endActiveEncoder();

            auto rpd{NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init())};

            for (NS::UInteger i = 0; i < static_cast<NS::UInteger>(desc.Color.size()); ++i)
            {
                const auto &ca  = desc.Color[i];
                auto       *att = rpd->colorAttachments()->object(i);
                att->setTexture(_lookupTexture(ca.Texture));
                att->setLoadAction(_toLoadAction(ca.LoadOp));
                att->setStoreAction(_toStoreAction(ca.StoreOp));
                att->setClearColor(MTL::ClearColor::Make((double)ca.ClearValue.R, (double)ca.ClearValue.G,
                                                         (double)ca.ClearValue.B, (double)ca.ClearValue.A));
                if (ca.ResolveTexture.Valid())
                {
                    att->setResolveTexture(_lookupTexture(ca.ResolveTexture));
                    att->setStoreAction(MTL::StoreActionMultisampleResolve);
                }
            }

            if (desc.Depth.Texture.Valid())
            {
                auto *d = rpd->depthAttachment();
                d->setTexture(_lookupTexture(desc.Depth.Texture));
                d->setLoadAction(_toLoadAction(desc.Depth.LoadOp));
                d->setStoreAction(_toStoreAction(desc.Depth.StoreOp));
                d->setClearDepth((double)desc.Depth.ClearValue.Depth);
            }

            _renderEncoder = NS::RetainPtr(_cmd->renderCommandEncoder(rpd.get()));
            _encoderType   = EncoderType::Render;

            _renderEncoder->setArgumentTable(_argTable.get(), MTL::RenderStageVertex | MTL::RenderStageFragment);
        }

        void EndRendering() override
        {
            if (_renderEncoder)
            {
                _renderEncoder->endEncoding();
                _renderEncoder.reset();
                _encoderType = EncoderType::None;
            }
        }

        void SetViewport(const Viewport &vp) override
        {
            _renderEncoder->setViewport(MTL::Viewport{(double)vp.X, (double)vp.Y, (double)vp.Width, (double)vp.Height,
                                                      (double)vp.MinDepth, (double)vp.MaxDepth});
        }

        void SetScissor(const Scissor &s) override
        {
            _renderEncoder->setScissorRect(
                MTL::ScissorRect{static_cast<NS::UInteger>(s.X), static_cast<NS::UInteger>(s.Y), s.Width, s.Height});
        }

        // ---- Pipeline ----

        void SetPipeline(PipelineHandle handle) override
        {
            if (!handle.Valid())
            {
                return;
            }
            auto &p = _device->Pipeline(handle);

            if (p.isCompute)
            {
                _ensureComputeEncoder();
                if (auto *label = p.computePso->label())
                {
                    _computeEncoder->setLabel(label);
                }
                _computeEncoder->setComputePipelineState(p.computePso.get());
                _tgX = p.threadGroupSizeX;
                _tgY = p.threadGroupSizeY;
                _tgZ = p.threadGroupSizeZ;
            }
            else
            {
                assert(_encoderType == EncoderType::Render && "SetPipeline(graphics) called outside BeginRendering()");
                _renderEncoder->setRenderPipelineState(p.renderPso.get());
                if (p.depthStencilState)
                {
                    _renderEncoder->setDepthStencilState(p.depthStencilState.get());
                }
                _renderEncoder->setFrontFacingWinding(p.winding);
                _renderEncoder->setCullMode(p.cullMode);
                _renderEncoder->setTriangleFillMode(p.fillMode);
                if (p.depthBiasConstant != 0.f || p.depthBiasSlope != 0.f)
                {
                    _renderEncoder->setDepthBias(p.depthBiasConstant, p.depthBiasSlope, 0.f);
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
            _argTable->setTexture(_device->Texture(texture).texture->gpuResourceID(), index);
        }

        void BindSampler(SamplerHandle sampler, uint32_t index) override
        {
            if (!sampler.Valid())
            {
                return;
            }
            _argTable->setSamplerState(_device->Sampler(sampler).state->gpuResourceID(), index);
        }

        // ---- Push constants ----

        void SetPushConstants(const void *data, uint32_t size, uint32_t offset) override
        {
            auto mapped = _device->MapBuffer(_pushConstantBuf);
            std::memcpy(static_cast<uint8_t *>(mapped.Data) + offset, data, size);
            _argTable->setAddress(_device->BufferAddress(_pushConstantBuf).Address, kPushConstantSlot);
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
            _renderEncoder->drawPrimitives(_primitiveType, firstVertex, vertexCount, instanceCount, firstInstance);
        }

        void DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                         uint32_t firstInstance) override
        {
            uint32_t        stride{_indexType == IndexType::Uint16 ? 2u : 4u};
            auto            idxType = _indexType == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
            const auto      indexStart{_indexOffset + static_cast<uint64_t>(firstIndex) * stride};
            MTL::GPUAddress addr{_device->BufferAddress(_indexBuffer).Address + indexStart};
            auto            len{static_cast<NS::UInteger>(_device->GetBufferInfo(_indexBuffer).Size - indexStart)};
            _renderEncoder->drawIndexedPrimitives(_primitiveType, indexCount, idxType, addr, len, instanceCount,
                                                  vertexOffset, static_cast<NS::UInteger>(firstInstance));
        }

        void DrawIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                          uint32_t /*stride*/) override
        {
            const auto base{_device->BufferAddress(argsBuffer).Address};
            for (uint32_t i = 0; i < drawCount; ++i)
            {
                _renderEncoder->drawPrimitives(_primitiveType, base + argsOffset + i * sizeof(DrawIndirectArgs));
            }
        }

        void DrawIndexedIndirect(BufferHandle argsBuffer, uint64_t argsOffset, uint32_t drawCount,
                                 uint32_t /*stride*/) override
        {
            auto            idxType = _indexType == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
            MTL::GPUAddress idxAddr{_device->BufferAddress(_indexBuffer).Address + _indexOffset};
            auto            idxLen{static_cast<NS::UInteger>(_device->GetBufferInfo(_indexBuffer).Size - _indexOffset)};
            const auto      argBase{_device->BufferAddress(argsBuffer).Address};
            for (uint32_t i = 0; i < drawCount; ++i)
            {
                _renderEncoder->drawIndexedPrimitives(_primitiveType, idxType, idxAddr, idxLen,
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
            _ensureComputeEncoder();
            _computeEncoder->dispatchThreadgroups(MTL::Size::Make(x, y, z), MTL::Size::Make(_tgX, _tgY, _tgZ));
        }

        void DispatchIndirect(BufferHandle argsBuffer, uint64_t argsOffset) override
        {
            _ensureComputeEncoder();
            MTL::GPUAddress addr{_device->BufferAddress(argsBuffer).Address + argsOffset};
            _computeEncoder->dispatchThreadgroups(addr, MTL::Size::Make(_tgX, _tgY, _tgZ));
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
            std::vector<NS::SharedPtr<NS::Object>> keepAlive;
            auto descriptor{_device->MakeAccelerationStructureDescriptor(desc, keepAlive)};

            auto *legacyCmd = _device->LegacyQueue()->commandBuffer();
            auto *enc       = legacyCmd->accelerationStructureCommandEncoder();
            if (!desc.DebugName.empty())
            {
                auto *label = NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding);
                legacyCmd->setLabel(label);
                enc->setLabel(label);
            }
            enc->buildAccelerationStructure(_device->AccelStruct(handle).as.get(), descriptor.get(),
                                            _lookupBuffer(scratchBuffer), scratchOffset);
            enc->endEncoding();

            // Suspended around the synchronous cross-queue wait below — see
            // SuspendActiveCaptureScope's doc comment in metal_internal.h.
            _device->SuspendActiveCaptureScope();
            legacyCmd->commit();
            legacyCmd->waitUntilCompleted();
            _device->ResumeActiveCaptureScope();
        }

        // ---- Copy ----
        //
        // MTL4 has no blit encoder — copyFromBuffer/copyFromTexture/fillBuffer
        // live directly on MTL4::ComputeCommandEncoder, so these all share
        // the same compute encoder Dispatch() uses.

        void CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &region) override
        {
            _ensureComputeEncoder();
            _computeEncoder->copyFromBuffer(_lookupBuffer(src), region.SrcOffset, _lookupBuffer(dst), region.DstOffset,
                                            region.Size);
        }

        void CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &region) override
        {
            _ensureComputeEncoder();
            _computeEncoder->copyFromTexture(
                _lookupTexture(src), region.ArrayLayer, region.MipLevel, MTL::Origin::Make(0, 0, 0),
                MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth), _lookupTexture(dst),
                region.ArrayLayer, region.MipLevel,
                MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                  static_cast<NS::UInteger>(region.DstOffset.Y),
                                  static_cast<NS::UInteger>(region.DstOffset.Z)));
        }

        void CopyBufferToTexture(BufferHandle src, uint64_t srcOffset, TextureHandle dst,
                                 const TextureCopyRegion &region) override
        {
            _ensureComputeEncoder();
            auto    &t = _device->Texture(dst);
            uint64_t bpp{bytesPerPixel(t.desc.Format)};
            uint64_t rowP{bpp * region.Extent.Width};
            uint64_t sliceP{rowP * region.Extent.Height};
            _computeEncoder->copyFromBuffer(
                _lookupBuffer(src), srcOffset, rowP, sliceP,
                MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth), _lookupTexture(dst),
                region.ArrayLayer, region.MipLevel,
                MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                  static_cast<NS::UInteger>(region.DstOffset.Y),
                                  static_cast<NS::UInteger>(region.DstOffset.Z)));
        }

        void CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &region, BufferHandle dst,
                                 uint64_t dstOffset) override
        {
            _ensureComputeEncoder();
            auto    &t = _device->Texture(src);
            uint64_t bpp{bytesPerPixel(t.desc.Format)};
            uint64_t rowP{bpp * region.Extent.Width};
            uint64_t sliceP{rowP * region.Extent.Height};
            _computeEncoder->copyFromTexture(
                _lookupTexture(src), region.ArrayLayer, region.MipLevel,
                MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                  static_cast<NS::UInteger>(region.DstOffset.Y),
                                  static_cast<NS::UInteger>(region.DstOffset.Z)),
                MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth), _lookupBuffer(dst),
                dstOffset, rowP, sliceP);
        }

        void ClearTexture(TextureHandle handle, const ClearColor &color, const SubresourceRange & /*range*/) override
        {
            // Metal has no standalone clear — use a transient render pass with loadAction=clear.
            _endActiveEncoder();
            auto *tex = _lookupTexture(handle);
            if (!tex)
            {
                return;
            }
            auto  rpd{NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init())};
            auto *att = rpd->colorAttachments()->object(0);
            att->setTexture(tex);
            att->setLoadAction(MTL::LoadActionClear);
            att->setStoreAction(MTL::StoreActionStore);
            att->setClearColor(
                MTL::ClearColor::Make((double)color.R, (double)color.G, (double)color.B, (double)color.A));
            auto *enc = _cmd->renderCommandEncoder(rpd.get());
            enc->endEncoding();
            _encoderType = EncoderType::None;
        }

        void ClearDepthTexture(TextureHandle handle, const ClearDepthStencil &clear,
                               const SubresourceRange & /*range*/) override
        {
            _endActiveEncoder();
            auto *tex = _lookupTexture(handle);
            if (!tex)
            {
                return;
            }
            auto  rpd{NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init())};
            auto *d = rpd->depthAttachment();
            d->setTexture(tex);
            d->setLoadAction(MTL::LoadActionClear);
            d->setStoreAction(MTL::StoreActionStore);
            d->setClearDepth((double)clear.Depth);
            auto *enc = _cmd->renderCommandEncoder(rpd.get());
            enc->endEncoding();
            _encoderType = EncoderType::None;
        }

        void FillBuffer(BufferHandle buf, uint64_t offset, uint64_t size, uint32_t value) override
        {
            _ensureComputeEncoder();
            // fillBuffer fills with a single byte value. For multi-byte
            // fill, a compute kernel is needed; we use the low byte for now.
            _computeEncoder->fillBuffer(_lookupBuffer(buf), NS::Range::Make(offset, size),
                                        static_cast<uint8_t>(value & 0xFF));
        }

        // ---- Debug ----

        void BeginDebugGroup(std::string_view name, float, float, float) override
        {
            auto *ns = NS::String::string(name.data(), NS::UTF8StringEncoding);
            _pushDebugGroup(ns);
        }

        void EndDebugGroup() override
        {
            _popDebugGroup();
        }

        void InsertDebugLabel(std::string_view label) override
        {
            auto *ns = NS::String::string(label.data(), NS::UTF8StringEncoding);
            if (_renderEncoder)
            {
                _renderEncoder->insertDebugSignpost(ns);
            }
            if (_computeEncoder)
            {
                _computeEncoder->insertDebugSignpost(ns);
            }
        }

        void DebugExposeBuffer(BufferHandle buffer, uint32_t unusedBindingSlot) override
        {
            if (!buffer.Valid())
            {
                return;
            }
            _checkDebugBufferSlot(unusedBindingSlot);
            _argTable->setAddress(_device->BufferAddress(buffer).Address, unusedBindingSlot);
        }

        void DebugExposeAccelerationStructure(AccelerationStructureHandle accelerationStructure,
                                              uint32_t                    unusedBindingSlot) override
        {
            if (!accelerationStructure.Valid())
            {
                return;
            }
            _checkDebugBufferSlot(unusedBindingSlot);
            _argTable->setResource(_device->AccelStruct(accelerationStructure).as->gpuResourceID(), unusedBindingSlot);
        }

        void DebugExposeTexture(TextureHandle texture, uint32_t unusedBindingSlot) override
        {
            if (!texture.Valid())
            {
                return;
            }
            if (unusedBindingSlot >= kMaxTextureBinds)
            {
                throw std::runtime_error("[LightRHI] DebugExposeTexture: binding slot " +
                                         std::to_string(unusedBindingSlot) + " is out of range (max " +
                                         std::to_string(kMaxTextureBinds) + ")");
            }
            _argTable->setTexture(_device->Texture(texture).texture->gpuResourceID(), unusedBindingSlot);
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
        enum class EncoderType
        {
            None,
            Render,
            Compute,
        };

        MetalDevice                               *_device{nullptr};
        NS::SharedPtr<MTL4::CommandAllocator>      _allocator;
        NS::SharedPtr<MTL4::CommandBuffer>         _cmd;
        NS::SharedPtr<MTL4::RenderCommandEncoder>  _renderEncoder;
        NS::SharedPtr<MTL4::ComputeCommandEncoder> _computeEncoder;
        NS::SharedPtr<MTL4::ArgumentTable>         _argTable;
        BufferHandle                               _pushConstantBuf{};
        std::string                                _debugName;
        FenceHandle                                _completionFence;
        bool                                       _cmdEnded{false};
        EncoderType                                _encoderType{EncoderType::None};
        [[maybe_unused]] QueueType                 _queueType{QueueType::Graphics};

        MTL::PrimitiveType _primitiveType{MTL::PrimitiveTypeTriangle};
        NS::UInteger       _tgX{64}, _tgY{1}, _tgZ{1};

        BufferHandle _indexBuffer{};
        uint64_t     _indexOffset{0};
        IndexType    _indexType{IndexType::Uint32};

        // ---- Encoder management ----

        void _endActiveEncoder()
        {
            if (_renderEncoder)
            {
                _renderEncoder->endEncoding();
                _renderEncoder.reset();
            }
            if (_computeEncoder)
            {
                _computeEncoder->endEncoding();
                _computeEncoder.reset();
            }
            _encoderType = EncoderType::None;
        }

        void _endCommandBuffer()
        {
            if (!_cmdEnded)
            {
                _cmd->endCommandBuffer();
                _cmdEnded = true;
            }
        }

        void _ensureComputeEncoder()
        {
            if (_encoderType == EncoderType::Compute)
            {
                return;
            }
            _endActiveEncoder();
            _computeEncoder = NS::RetainPtr(_cmd->computeCommandEncoder());
            _encoderType    = EncoderType::Compute;
            _computeEncoder->setArgumentTable(_argTable.get());
        }

        // ---- Resource lookups ----

        [[nodiscard]] MTL::Buffer *_lookupBuffer(BufferHandle h) const noexcept
        {
            if (!h.Valid())
            {
                return nullptr;
            }
            return _device->Buffer(h).buffer.get();
        }

        [[nodiscard]] MTL::Texture *_lookupTexture(TextureHandle h) const noexcept
        {
            if (!h.Valid())
            {
                return nullptr;
            }
            return _device->Texture(h).texture.get();
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
                throw std::runtime_error("[LightRHI] Debug resource slot " + std::to_string(slot) +
                                         " overlaps or exceeds the push-constant slot (" +
                                         std::to_string(kPushConstantSlot) + ")");
            }
        }

        void _pushDebugGroup(NS::String *name)
        {
            if (_renderEncoder)
            {
                _renderEncoder->pushDebugGroup(name);
            }
            else if (_computeEncoder)
            {
                _computeEncoder->pushDebugGroup(name);
            }
            else
            {
                _cmd->pushDebugGroup(name);
            }
        }

        void _popDebugGroup()
        {
            if (_renderEncoder)
            {
                _renderEncoder->popDebugGroup();
            }
            else if (_computeEncoder)
            {
                _computeEncoder->popDebugGroup();
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

    std::unique_ptr<ICommandList> MetalDevice::CreateCommandList(QueueType q, std::string_view name)
    {
        auto resources{AcquireCommandResources()};
        return std::make_unique<MetalCommandList>(this, std::move(resources), q, name);
    }

    FenceHandle MetalDevice::Submit(ICommandList &cmdList, const SubmitDesc & /*desc*/)
    {
        auto                &mcl = static_cast<MetalCommandList &>(cmdList);
        FenceHandle          fence{NextFence()};
        MTL4::CommandBuffer *buffers[1]{mcl.commandBuffer()};
        auto                *captureScope = SubmissionCaptureScope(mcl.debugName());
        if (captureScope)
        {
            captureScope->beginScope();
        }
        Mtl4Queue()->commit(buffers, 1);
        Mtl4Queue()->signalEvent(TimelineEvent(), fence.Id);
        mcl.setCompletionFence(fence);
        if (captureScope)
        {
            captureScope->endScope();
        }
        return fence;
    }

} // namespace rhi::metal
