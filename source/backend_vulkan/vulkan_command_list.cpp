// vulkan_command_list.cpp — VulkanCommandList + VulkanDevice::CreateCommandList/Submit

module;
#include <cassert>
#include <cstring>

module lightRHI;
import rhi;

#include "vulkan_internal.h"

namespace rhi::vulkan
{

    // ============================================================================
    // VulkanCommandList
    // ============================================================================

    class VulkanCommandList final : public ICommandList
    {
      public:
        VulkanCommandList(VulkanDevice *dev, VkCommandPool pool, std::mutex *poolMtx, VkCommandBuffer cmd,
                          QueueType qtype)
            : _dev{dev}, _pool{pool}, _poolMtx{poolMtx}, _cmd{cmd}, _qtype{qtype}
        {
        }

        ~VulkanCommandList() override
        {
            // TLAS instance buffers created in BuildAccelerationStructure()
            // must outlive the build command's execution; freed here once
            // this command list (and the fence guaranteeing GPU completion,
            // per this API's documented contract) is done with it.
            for (BufferHandle buf : _asInstanceBuffers)
            {
                _dev->DestroyBuffer(buf);
            }

            if (_cmd != VK_NULL_HANDLE)
            {
                std::scoped_lock lk{*_poolMtx};
                vkFreeCommandBuffers(_dev->NativeDevice(), _pool, 1, &_cmd);
            }
        }

        // ---- Lifecycle ----

        void Begin() override
        {
            VkCommandBufferBeginInfo bi{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(_cmd, &bi);

            // Bind the global bindless descriptor buffer once per command buffer.
            // All draws/dispatches in this buffer see the same bindless heap.
            VkDeviceAddress heapAddr{_dev->Heap().DescriptorBufferAddress()};
            if (heapAddr != 0)
            {
                VkDescriptorBufferBindingInfoEXT bnd{
                    .sType   = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
                    .address = heapAddr,
                    .usage   = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                               VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                };
                _dev->pfn_CmdBindDescriptorBuffers(_cmd, 1, &bnd);
                _bindHeapOffsets(VK_PIPELINE_BIND_POINT_GRAPHICS);
                _bindHeapOffsets(VK_PIPELINE_BIND_POINT_COMPUTE);
            }
        }

        void End() override
        {
            FlushBarriers();
            vkEndCommandBuffer(_cmd);
        }

        // ---- Barriers ----

        void Transition(const TextureBarrier &b) override
        {
            _pendingImg.push_back(_makeImageBarrier(b));
            // Tracks the layout the image will be in once the queued barriers are
            // recorded. Nothing may read rec.layout without flushing first —
            // _ensureLayout() is the only reader and it flushes, which is what
            // keeps this value honest.
            if (b.Texture.Valid())
            {
                _dev->Texture(b.Texture).layout = _toLayout(b.After);
            }
        }
        void Transition(const BufferBarrier &b) override
        {
            _pendingBuf.push_back(_makeBufBarrier(b));
        }
        void Transition(const MemoryBarrier &b) override
        {
            _pendingMem.push_back(_makeMemBarrier(b));
        }
        void Transition(const AccelerationStructureBarrier &b) override
        {
            // No AS-specific VkXxxMemoryBarrier2 variant exists — a global
            // memory barrier scoped to the AS build/read stage+access flags
            // (see _toStage/_toAccess) is the documented way to order an AS
            // build against its later consumption by a ray-query shader.
            _pendingMem.push_back(_makeMemBarrier(MemoryBarrier{b.Before, b.After}));
        }

        void FlushBarriers() override
        {
            if (_pendingImg.empty() && _pendingBuf.empty() && _pendingMem.empty())
            {
                return;
            }
            VkDependencyInfo dep{
                .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount       = static_cast<uint32_t>(_pendingMem.size()),
                .pMemoryBarriers          = _pendingMem.data(),
                .bufferMemoryBarrierCount = static_cast<uint32_t>(_pendingBuf.size()),
                .pBufferMemoryBarriers    = _pendingBuf.data(),
                .imageMemoryBarrierCount  = static_cast<uint32_t>(_pendingImg.size()),
                .pImageMemoryBarriers     = _pendingImg.data(),
            };
            vkCmdPipelineBarrier2(_cmd, &dep);
            _pendingImg.clear();
            _pendingBuf.clear();
            _pendingMem.clear();
        }

        // ---- Dynamic rendering ----

        void BeginRendering(const RenderingDesc &desc) override
        {
            FlushBarriers();

            // Move every attachment into its rendering layout automatically.
            for (const auto &ca : desc.Color)
            {
                _ensureLayout(ca.Texture, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                _ensureLayout(ca.ResolveTexture, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
            if (desc.Depth.Texture.Valid())
            {
                _ensureLayout(desc.Depth.Texture, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            }

            std::vector<VkRenderingAttachmentInfo> color;
            color.reserve(desc.Color.size());
            for (const auto &ca : desc.Color)
            {
                color.push_back({
                    .sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .imageView        = _lookupView(ca.Texture),
                    .imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .resolveMode      = ca.ResolveTexture.Valid() ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
                    .resolveImageView = _lookupView(ca.ResolveTexture),
                    .resolveImageLayout = ca.ResolveTexture.Valid() ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                                                    : VK_IMAGE_LAYOUT_UNDEFINED,
                    .loadOp             = _toLoadOp(ca.LoadOp),
                    .storeOp            = _toStoreOp(ca.StoreOp),
                    .clearValue         = {.color = {.float32 = {ca.ClearValue.R, ca.ClearValue.G, ca.ClearValue.B,
                                                                 ca.ClearValue.A}}},
                });
            }

            VkRenderingAttachmentInfo depth{};
            const bool                hasDepth{desc.Depth.Texture.Valid()};
            if (hasDepth)
            {
                depth = {
                    .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .imageView   = _lookupView(desc.Depth.Texture),
                    .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .loadOp      = _toLoadOp(desc.Depth.LoadOp),
                    .storeOp     = _toStoreOp(desc.Depth.StoreOp),
                    .clearValue  = {.depthStencil = {desc.Depth.ClearValue.Depth, desc.Depth.ClearValue.Stencil}},
                };
            }

            VkRenderingInfo ri{
                .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea           = {.offset = {0, 0}, .extent = {desc.RenderArea.Width, desc.RenderArea.Height}},
                .layerCount           = desc.LayerCount,
                .colorAttachmentCount = static_cast<uint32_t>(color.size()),
                .pColorAttachments    = color.data(),
                .pDepthAttachment     = hasDepth ? &depth : nullptr,
            };
            vkCmdBeginRendering(_cmd, &ri);
        }

        void EndRendering() override
        {
            vkCmdEndRendering(_cmd);
        }

        void SetViewport(const Viewport &vp) override
        {
            VkViewport v{vp.X, vp.Y, vp.Width, vp.Height, vp.MinDepth, vp.MaxDepth};
            vkCmdSetViewport(_cmd, 0, 1, &v);
        }

        void SetScissor(const Scissor &s) override
        {
            VkRect2D r{.offset = {s.X, s.Y}, .extent = {s.Width, s.Height}};
            vkCmdSetScissor(_cmd, 0, 1, &r);
        }

        // ---- Pipeline ----

        void SetPipeline(PipelineHandle h) override
        {
            if (!h.Valid())
            {
                return;
            }
            const auto &p  = _dev->Pipeline(h);
            auto        bp = p.isCompute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;
            vkCmdBindPipeline(_cmd, bp, p.pipeline);
            _currentLayout            = _dev->GlobalPipelineLayout();
            _currentPushConstantBytes = p.pushConstantBytes;
            // Re-set descriptor buffer offsets for the new bind point
            _bindHeapOffsets(bp);
        }

        // ---- Texture/sampler binding ----
        //
        // No-op on Vulkan: textures/samplers here stay fully bindless via
        // this backend's own descriptor-buffer heap and DescriptorHandle<
        // Texture2D>/<SamplerState> embedded directly in push constants —
        // see ICommandList::BindTexture's doc comment (commandList.cppm).

        void BindTexture(TextureHandle /*texture*/, uint32_t /*index*/) override {}
        void BindSampler(SamplerHandle /*sampler*/, uint32_t /*index*/) override {}

        // ---- Push constants ----

        void SetPushConstants(const void *data, uint32_t size, uint32_t offset) override
        {
            if (!_currentLayout || !data || size > _currentPushConstantBytes ||
                offset > _currentPushConstantBytes - size)
            {
                return;
            }
            vkCmdPushConstants(_cmd, _currentLayout, VK_SHADER_STAGE_ALL, offset, size, data);
        }

        // ---- Index buffer ----

        void BindIndexBuffer(BufferHandle buf, uint64_t offset, IndexType idxType) override
        {
            vkCmdBindIndexBuffer(_cmd, _lookupBuf(buf), offset,
                                 idxType == IndexType::Uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        }

        // ---- Draw ----

        void Draw(uint32_t vcount, uint32_t icount, uint32_t fv, uint32_t fi) override
        {
            vkCmdDraw(_cmd, vcount, icount, fv, fi);
        }
        void DrawIndexed(uint32_t icount, uint32_t instanceCount, uint32_t firstIdx, int32_t vOff, uint32_t fi) override
        {
            vkCmdDrawIndexed(_cmd, icount, instanceCount, firstIdx, vOff, fi);
        }
        void DrawIndirect(BufferHandle args, uint64_t off, uint32_t count, uint32_t stride) override
        {
            vkCmdDrawIndirect(_cmd, _lookupBuf(args), off, count, stride);
        }
        void DrawIndexedIndirect(BufferHandle args, uint64_t off, uint32_t count, uint32_t stride) override
        {
            vkCmdDrawIndexedIndirect(_cmd, _lookupBuf(args), off, count, stride);
        }
        void DrawIndirectCount(BufferHandle args, uint64_t argsOff, BufferHandle countBuf, uint64_t countOff,
                               uint32_t maxDraw, uint32_t stride) override
        {
            vkCmdDrawIndirectCount(_cmd, _lookupBuf(args), argsOff, _lookupBuf(countBuf), countOff, maxDraw, stride);
        }

        // ---- Compute ----

        void Dispatch(uint32_t x, uint32_t y, uint32_t z) override
        {
            // Emit any barriers queued via Transition() before the work that
            // depends on them — deferred barriers are otherwise only flushed at
            // end()/BeginRendering(), which would place a barrier between two
            // dispatches after both have already been recorded.
            FlushBarriers();
            vkCmdDispatch(_cmd, x, y, z);
        }
        void DispatchIndirect(BufferHandle args, uint64_t off) override
        {
            FlushBarriers();
            vkCmdDispatchIndirect(_cmd, _lookupBuf(args), off);
        }

        // ---- Ray tracing ----
        //
        // There is no bind step: shaders receive the AS bindlessly, as its
        // device address (IDevice::AccelerationStructureAddress) carried in
        // push constants and converted back via
        // OpConvertUToAccelerationStructureKHR — see device.cppm.

        void BuildAccelerationStructure(AccelerationStructureHandle handle, const AccelerationStructureDesc &desc,
                                        BufferHandle scratchBuffer, uint64_t scratchOffset) override
        {
            VkDeviceAddress instanceAddr{0};
            if (desc.Type == AccelerationStructureType::TopLevel)
            {
                std::vector<VkAccelerationStructureInstanceKHR> instances;
                instances.reserve(desc.Instances.size());
                for (const auto &inst : desc.Instances)
                {
                    if (!inst.Blas.Valid())
                    {
                        throw std::runtime_error(
                            "[LightRHI::Vulkan] TlasFromInstances: AccelerationStructureInstance.Blas is not a "
                            "valid handle");
                    }

                    VkDeviceAddress blasAddr{_dev->AccelStruct(inst.Blas).deviceAddress};

                    VkAccelerationStructureInstanceKHR vkInst{};
                    for (int r = 0; r < 3; ++r)
                    {
                        for (int c = 0; c < 4; ++c)
                        {
                            vkInst.transform.matrix[r][c] = inst.Transform[r][c];
                        }
                    }
                    vkInst.instanceCustomIndex                    = inst.InstanceCustomIndex;
                    vkInst.mask                                   = inst.InstanceMask;
                    vkInst.instanceShaderBindingTableRecordOffset = 0;
                    vkInst.flags                                  = static_cast<VkGeometryInstanceFlagsKHR>(inst.Flags);
                    vkInst.accelerationStructureReference         = blasAddr;
                    instances.push_back(vkInst);
                }

                // Written directly into a host-visible mapped buffer instead
                // of going through the device's UploadBuffer() staging path:
                // UploadBuffer submits a separate command buffer and blocks
                // on a GPU fence wait, which would stall this command list's
                // *recording* on every TLAS build. Instance data is small
                // (sizeof(VkAccelerationStructureInstanceKHR) per instance),
                // so host-visible (CpuToGpu) memory is the right tradeoff.
                // Freed in ~VulkanCommandList() — see _asInstanceBuffers.
                uint64_t     instanceBytes{sizeof(VkAccelerationStructureInstanceKHR) * instances.size()};
                BufferHandle instanceBuf{_dev->CreateBuffer({
                    .Size       = instanceBytes,
                    .Usage      = BufferUsage::Storage | BufferUsage::DeviceAddress,
                    .MemoryType = MemoryType::CpuToGpu,
                    .DebugName  = "tlas_instance_buffer",
                })};
                MappedBuffer mapped{_dev->MapBuffer(instanceBuf)};
                std::memcpy(mapped.Data, instances.data(), instanceBytes);
                _dev->UnmapBuffer(instanceBuf);
                _asInstanceBuffers.push_back(instanceBuf);
                instanceAddr = _dev->BufferAddress(instanceBuf).Address;
            }

            VkAccelerationStructureGeometryKHR          geometry{};
            VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
            uint32_t                                    primCount{0};
            _dev->BuildAccelGeometryInfo(desc, instanceAddr, geometry, buildInfo, primCount);

            buildInfo.dstAccelerationStructure  = _dev->AccelStruct(handle).as;
            buildInfo.scratchData.deviceAddress = _dev->BufferAddress(scratchBuffer).Address + scratchOffset;

            VkAccelerationStructureBuildRangeInfoKHR        rangeInfo{.primitiveCount = primCount};
            const VkAccelerationStructureBuildRangeInfoKHR *pRange{&rangeInfo};
            // Device-side build only — vkCmdBuildAccelerationStructuresKHR,
            // never the host-side vkBuildAccelerationStructuresKHR (deprecated
            // by Khronos; see
            // https://www.khronos.org/blog/vulkan-ray-tracing-deprecating-host-side-acceleration-structure-builds).
            _dev->pfn_CmdBuildAccelerationStructures(_cmd, 1, &buildInfo, &pRange);
        }

        // ---- Copy ----

        void CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &r) override
        {
            FlushBarriers();
            VkBufferCopy2     cp{.sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                 .srcOffset = r.SrcOffset,
                                 .dstOffset = r.DstOffset,
                                 .size      = r.Size};
            VkCopyBufferInfo2 ci{.sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                 .srcBuffer   = _lookupBuf(src),
                                 .dstBuffer   = _lookupBuf(dst),
                                 .regionCount = 1,
                                 .pRegions    = &cp};
            vkCmdCopyBuffer2(_cmd, &ci);
        }

        void CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &r) override
        {
            _ensureLayout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            _ensureLayout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageCopy2 cp{
                .sType          = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
                .srcSubresource = {_aspectOf(src), r.MipLevel, r.ArrayLayer, 1},
                .srcOffset      = {r.DstOffset.X, r.DstOffset.Y, r.DstOffset.Z},
                .dstSubresource = {_aspectOf(dst), r.MipLevel, r.ArrayLayer, 1},
                .dstOffset      = {0, 0, 0},
                .extent         = {r.Extent.Width, r.Extent.Height, r.Extent.Depth ? r.Extent.Depth : 1u},
            };
            VkCopyImageInfo2 ci{
                .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
                .srcImage       = _lookupImg(src),
                .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .dstImage       = _lookupImg(dst),
                .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .regionCount    = 1,
                .pRegions       = &cp,
            };
            vkCmdCopyImage2(_cmd, &ci);
        }

        void CopyBufferToTexture(BufferHandle src, uint64_t srcOff, TextureHandle dst,
                                 const TextureCopyRegion &r) override
        {
            _ensureLayout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy2 cp{
                .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                .bufferOffset      = srcOff,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {_aspectOf(dst), r.MipLevel, r.ArrayLayer, 1},
                .imageOffset       = {r.DstOffset.X, r.DstOffset.Y, r.DstOffset.Z},
                .imageExtent       = {r.Extent.Width, r.Extent.Height, r.Extent.Depth ? r.Extent.Depth : 1u},
            };
            VkCopyBufferToImageInfo2 ci{
                .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                .srcBuffer      = _lookupBuf(src),
                .dstImage       = _lookupImg(dst),
                .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .regionCount    = 1,
                .pRegions       = &cp,
            };
            vkCmdCopyBufferToImage2(_cmd, &ci);
        }

        void CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &r, BufferHandle dst,
                                 uint64_t dstOff) override
        {
            _ensureLayout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy2 cp{
                .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                .bufferOffset      = dstOff,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {_aspectOf(src), r.MipLevel, r.ArrayLayer, 1},
                .imageOffset       = {r.DstOffset.X, r.DstOffset.Y, r.DstOffset.Z},
                .imageExtent       = {r.Extent.Width, r.Extent.Height, r.Extent.Depth ? r.Extent.Depth : 1u},
            };
            VkCopyImageToBufferInfo2 ci{
                .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                .srcImage       = _lookupImg(src),
                .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .dstBuffer      = _lookupBuf(dst),
                .regionCount    = 1,
                .pRegions       = &cp,
            };
            vkCmdCopyImageToBuffer2(_cmd, &ci);
        }

        void ClearTexture(TextureHandle tex, const ClearColor &c, const SubresourceRange &range) override
        {
            _ensureLayout(tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue       cv{.float32 = {c.R, c.G, c.B, c.A}};
            VkImageSubresourceRange sr{
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = range.BaseMip,
                .levelCount     = range.MipCount,
                .baseArrayLayer = range.BaseLayer,
                .layerCount     = range.LayerCount,
            };
            vkCmdClearColorImage(_cmd, _lookupImg(tex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &sr);
        }

        void ClearDepthTexture(TextureHandle tex, const ClearDepthStencil &c, const SubresourceRange &range) override
        {
            _ensureLayout(tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearDepthStencilValue cv{c.Depth, c.Stencil};
            VkImageSubresourceRange  sr{
                // Depth-only formats have no stencil aspect — naming one is invalid.
                .aspectMask = _aspectOf(tex),      .baseMipLevel = range.BaseMip,  .levelCount = range.MipCount,
                .baseArrayLayer = range.BaseLayer, .layerCount = range.LayerCount,
            };
            vkCmdClearDepthStencilImage(_cmd, _lookupImg(tex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &sr);
        }

        void FillBuffer(BufferHandle buf, uint64_t off, uint64_t sz, uint32_t val) override
        {
            FlushBarriers();
            vkCmdFillBuffer(_cmd, _lookupBuf(buf), off, sz, val);
        }

        // ---- Debug ----

        void BeginDebugGroup(std::string_view name, float r, float g, float b) override
        {
            VkDebugUtilsLabelEXT l{
                .sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                .pLabelName = name.data(),
                .color      = {r, g, b, 1.f},
            };
            if (_dev->pfn_CmdBeginDebugUtilsLabel)
            {
                _dev->pfn_CmdBeginDebugUtilsLabel(_cmd, &l);
            }
        }
        void EndDebugGroup() override
        {
            if (_dev->pfn_CmdEndDebugUtilsLabel)
            {
                _dev->pfn_CmdEndDebugUtilsLabel(_cmd);
            }
        }
        void InsertDebugLabel(std::string_view lbl) override
        {
            VkDebugUtilsLabelEXT l{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, .pLabelName = lbl.data()};
            if (_dev->pfn_CmdInsertDebugUtilsLabel)
            {
                _dev->pfn_CmdInsertDebugUtilsLabel(_cmd, &l);
            }
        }

        void DebugExposeBuffer(BufferHandle, uint32_t) override {}

        void DebugExposeAccelerationStructure(AccelerationStructureHandle, uint32_t) override {}

        void DebugExposeTexture(TextureHandle, uint32_t) override {}

        // ---- Accessors for VulkanDevice::Submit ----
        [[nodiscard]] VkCommandBuffer commandBuffer() const noexcept
        {
            return _cmd;
        }
        [[nodiscard]] QueueType queueType() const noexcept
        {
            return _qtype;
        }

      private:
        VulkanDevice    *_dev{nullptr};
        VkCommandPool    _pool{VK_NULL_HANDLE};
        std::mutex      *_poolMtx{nullptr};
        VkCommandBuffer  _cmd{VK_NULL_HANDLE};
        QueueType        _qtype{QueueType::Graphics};
        VkPipelineLayout _currentLayout{VK_NULL_HANDLE};
        uint32_t         _currentPushConstantBytes{0};

        // TLAS instance buffers created by BuildAccelerationStructure(),
        // freed in ~VulkanCommandList() — see the destructor comment.
        std::vector<BufferHandle> _asInstanceBuffers;

        // Deferred barriers — flushed together for a single vkCmdPipelineBarrier2 call
        std::vector<VkImageMemoryBarrier2>  _pendingImg;
        std::vector<VkBufferMemoryBarrier2> _pendingBuf;
        std::vector<VkMemoryBarrier2>       _pendingMem;

        // ---- Handle lookups ----

        [[nodiscard]] VkBuffer _lookupBuf(BufferHandle h) const noexcept
        {
            return h.Valid() ? _dev->Buffer(h).buffer : VK_NULL_HANDLE;
        }
        [[nodiscard]] VkImage _lookupImg(TextureHandle h) const noexcept
        {
            return h.Valid() ? _dev->Texture(h).image : VK_NULL_HANDLE;
        }
        [[nodiscard]] VkImageView _lookupView(TextureHandle h) const noexcept
        {
            return h.Valid() ? _dev->Texture(h).view : VK_NULL_HANDLE;
        }

        // ---- Automatic image layout tracking ----
        // Vulkan requires images to be in the right layout for each use, but the
        // RHI's examples/tests don't issue explicit texture barriers — they
        // expect the backend to manage layouts. We track each texture's current
        // layout on its record and emit a transition when a use needs a
        // different one. Coarse ALL_COMMANDS masks: correctness over precision.
        // Aspect mask implied by a texture's format. Single source of truth for
        // barriers, copy subresources and clears — a depth image must never be
        // referenced through the colour aspect.
        [[nodiscard]] VkImageAspectFlags _aspectOf(TextureHandle h) const noexcept
        {
            if (!h.Valid())
            {
                return VK_IMAGE_ASPECT_COLOR_BIT;
            }
            switch (_dev->Texture(h).desc.Format)
            {
                case Format::D16Unorm:
                case Format::D32Float:
                    return VK_IMAGE_ASPECT_DEPTH_BIT;
                case Format::D24UnormS8Uint:
                case Format::D32FloatS8Uint:
                    return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
                default:
                    return VK_IMAGE_ASPECT_COLOR_BIT;
            }
        }

        void _ensureLayout(TextureHandle h, VkImageLayout newLayout)
        {
            if (!h.Valid())
            {
                return;
            }

            // Record any queued barriers first: they may themselves move this
            // image, and until they land the tracked layout is not yet the
            // image's real layout.
            FlushBarriers();

            auto &rec = _dev->Texture(h);
            if (rec.layout == newLayout)
            {
                return;
            }

            const VkImageAspectFlags aspect{_aspectOf(h)};

            VkImageMemoryBarrier2 b{
                .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout           = rec.layout,
                .newLayout           = newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = rec.image,
                .subresourceRange    = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS},
            };
            VkDependencyInfo dep{
                .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers    = &b,
            };
            vkCmdPipelineBarrier2(_cmd, &dep);
            rec.layout = newLayout;
        }

        // ---- Descriptor buffer binding ----

        void _bindHeapOffsets(VkPipelineBindPoint bp)
        {
            if (!_currentLayout && bp == VK_PIPELINE_BIND_POINT_GRAPHICS)
            {
                return;
            }
            // Use the global layout — it's always valid by the time we begin
            VkPipelineLayout layout{_dev->GlobalPipelineLayout()};
            uint32_t         descriptorBufferIndex{0};
            VkDeviceSize     off{0};
            _dev->pfn_CmdSetDescriptorBufferOffsets(_cmd, bp, layout, 1, 1, &descriptorBufferIndex, &off);
        }

        // ---- Conversion helpers ----

        // Compute/transfer queues reject graphics-only pipeline stages. Collapse
        // any such bits to ALL_COMMANDS (valid on every queue) so the coarse
        // shader/UAV barriers the RHI emits still synchronize correctly there.
        [[nodiscard]] VkPipelineStageFlags2 _sanitizeStages(VkPipelineStageFlags2 f) const noexcept
        {
            if (_qtype == QueueType::Graphics)
            {
                return f;
            }
            constexpr VkPipelineStageFlags2 kGraphicsOnly{
                VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT};
            if (f & kGraphicsOnly)
            {
                return (f & ~kGraphicsOnly) | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
            return f;
        }

        // Access flags are validated against the queue family too, so the stage
        // clamp above is only half the job: drop the access bits the queue can't
        // express, or vkCmdPipelineBarrier2 rejects the barrier and the
        // synchronization silently never happens.
        [[nodiscard]] VkAccessFlags2 _sanitizeAccess(VkAccessFlags2 f) const noexcept
        {
            constexpr VkAccessFlags2 kGraphicsOnly{
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT};
            constexpr VkAccessFlags2 kShaderOnly{VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                                                 VK_ACCESS_2_UNIFORM_READ_BIT};
            switch (_qtype)
            {
                case QueueType::Graphics:
                    return f;
                case QueueType::Compute:
                    return f & ~kGraphicsOnly;
                case QueueType::Transfer:
                    // A transfer queue can only express transfer access.
                    return f & ~(kGraphicsOnly | kShaderOnly | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
            }
            return f;
        }

        [[nodiscard]] VkPipelineStageFlags2 _toStage(ResourceState s) const noexcept
        {
            VkPipelineStageFlags2 f{0};
            if (HasState(s, ResourceState::VertexBuffer))
            {
                f |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            }
            if (HasState(s, ResourceState::IndexBuffer))
            {
                f |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
            }
            if (HasState(s, ResourceState::ConstantBuffer))
            {
                f |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
            }
            if (HasState(s, ResourceState::ShaderRead))
            {
                f |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
            if (HasState(s, ResourceState::UnorderedAccess))
            {
                f |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
            if (HasState(s, ResourceState::RenderTarget))
            {
                f |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            }
            if (HasState(s, ResourceState::DepthRead) || HasState(s, ResourceState::DepthWrite))
            {
                f |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            }
            if (HasState(s, ResourceState::TransferSrc) || HasState(s, ResourceState::TransferDst))
            {
                f |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            }
            if (HasState(s, ResourceState::IndirectArgument))
            {
                f |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            }
            if (HasState(s, ResourceState::AccelerationStructureWrite))
            {
                f |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            }
            if (HasState(s, ResourceState::AccelerationStructureRead))
            {
                // Inline ray query (RayQuery in a plain compute shader) reads
                // the AS from the compute stage — LightRHI has no dedicated
                // ray tracing pipeline/shader stage.
                f |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            }
            if (f == 0)
            {
                f = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            }
            return f;
        }

        [[nodiscard]] VkAccessFlags2 _toAccess(ResourceState s) const noexcept
        {
            VkAccessFlags2 f{0};
            if (HasState(s, ResourceState::VertexBuffer))
            {
                f |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            }
            if (HasState(s, ResourceState::IndexBuffer))
            {
                f |= VK_ACCESS_2_INDEX_READ_BIT;
            }
            if (HasState(s, ResourceState::ConstantBuffer))
            {
                f |= VK_ACCESS_2_UNIFORM_READ_BIT;
            }
            if (HasState(s, ResourceState::ShaderRead))
            {
                f |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (HasState(s, ResourceState::UnorderedAccess))
            {
                f |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            }
            if (HasState(s, ResourceState::RenderTarget))
            {
                f |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            }
            if (HasState(s, ResourceState::DepthRead))
            {
                f |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            if (HasState(s, ResourceState::DepthWrite))
            {
                f |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            if (HasState(s, ResourceState::TransferSrc))
            {
                f |= VK_ACCESS_2_TRANSFER_READ_BIT;
            }
            if (HasState(s, ResourceState::TransferDst))
            {
                f |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }
            if (HasState(s, ResourceState::IndirectArgument))
            {
                f |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            }
            if (HasState(s, ResourceState::AccelerationStructureWrite))
            {
                f |= VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            }
            if (HasState(s, ResourceState::AccelerationStructureRead))
            {
                f |= VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            }
            return f;
        }

        [[nodiscard]] VkImageLayout _toLayout(ResourceState s) const noexcept
        {
            // Intentionally specific — VK_IMAGE_LAYOUT_GENERAL only for UAV.
            if (HasState(s, ResourceState::RenderTarget))
            {
                return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            if (HasState(s, ResourceState::DepthWrite))
            {
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
            if (HasState(s, ResourceState::DepthRead))
            {
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            }
            if (HasState(s, ResourceState::ShaderRead))
            {
                return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            if (HasState(s, ResourceState::UnorderedAccess))
            {
                return VK_IMAGE_LAYOUT_GENERAL;
            }
            if (HasState(s, ResourceState::TransferSrc))
            {
                return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            }
            if (HasState(s, ResourceState::TransferDst))
            {
                return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            }
            if (HasState(s, ResourceState::Present))
            {
                return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            }
            return VK_IMAGE_LAYOUT_UNDEFINED;
        }

        [[nodiscard]] VkImageMemoryBarrier2 _makeImageBarrier(const TextureBarrier &b) const
        {
            const VkImageAspectFlags aspect{_aspectOf(b.Texture)};
            return VkImageMemoryBarrier2{
                .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask        = _sanitizeStages(_toStage(b.Before)),
                .srcAccessMask       = _sanitizeAccess(_toAccess(b.Before)),
                .dstStageMask        = _sanitizeStages(_toStage(b.After)),
                .dstAccessMask       = _sanitizeAccess(_toAccess(b.After)),
                .oldLayout           = _toLayout(b.Before),
                .newLayout           = _toLayout(b.After),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = _lookupImg(b.Texture),
                .subresourceRange =
                    {
                        .aspectMask     = aspect,
                        .baseMipLevel   = b.Range.BaseMip,
                        .levelCount     = b.Range.MipCount,
                        .baseArrayLayer = b.Range.BaseLayer,
                        .layerCount     = b.Range.LayerCount,
                    },
            };
        }

        [[nodiscard]] VkBufferMemoryBarrier2 _makeBufBarrier(const BufferBarrier &b) const
        {
            return VkBufferMemoryBarrier2{
                .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask        = _sanitizeStages(_toStage(b.Before)),
                .srcAccessMask       = _sanitizeAccess(_toAccess(b.Before)),
                .dstStageMask        = _sanitizeStages(_toStage(b.After)),
                .dstAccessMask       = _sanitizeAccess(_toAccess(b.After)),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer              = _lookupBuf(b.Buffer),
                .offset              = b.Offset,
                .size                = b.Size,
            };
        }

        [[nodiscard]] VkMemoryBarrier2 _makeMemBarrier(const MemoryBarrier &b) const
        {
            return VkMemoryBarrier2{
                .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask  = _sanitizeStages(_toStage(b.Before)),
                .srcAccessMask = _sanitizeAccess(_toAccess(b.Before)),
                .dstStageMask  = _sanitizeStages(_toStage(b.After)),
                .dstAccessMask = _sanitizeAccess(_toAccess(b.After)),
            };
        }

        [[nodiscard]] static VkAttachmentLoadOp _toLoadOp(LoadOp o) noexcept
        {
            switch (o)
            {
                case LoadOp::Load:
                    return VK_ATTACHMENT_LOAD_OP_LOAD;
                case LoadOp::Clear:
                    return VK_ATTACHMENT_LOAD_OP_CLEAR;
                case LoadOp::DontCare:
                    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            }
            return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }
        [[nodiscard]] static VkAttachmentStoreOp _toStoreOp(StoreOp o) noexcept
        {
            switch (o)
            {
                case StoreOp::Store:
                    return VK_ATTACHMENT_STORE_OP_STORE;
                case StoreOp::DontCare:
                    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
            }
            return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }
    };

    // ============================================================================
    // VulkanDevice methods that reference VulkanCommandList
    // ============================================================================

    std::unique_ptr<ICommandList> VulkanDevice::CreateCommandList(QueueType qtype, std::string_view /*name*/)
    {
        VkCommandPool pool;
        std::mutex   *mtx;
        switch (qtype)
        {
            case QueueType::Compute:
                pool = computePool;
                mtx  = &computePoolMtx;
                break;
            case QueueType::Transfer:
                pool = xferPool;
                mtx  = &xferPoolMtx;
                break;
            default:
                pool = gfxPool;
                mtx  = &gfxPoolMtx;
                break;
        }

        VkCommandBufferAllocateInfo ai{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer cmd;
        {
            std::scoped_lock lk{*mtx};
            VK_CHECK(vkAllocateCommandBuffers(_device, &ai, &cmd));
        }
        return std::make_unique<VulkanCommandList>(this, pool, mtx, cmd, qtype);
    }

    FenceHandle VulkanDevice::Submit(ICommandList &cmdList, const SubmitDesc &desc)
    {
        auto &vcl = static_cast<VulkanCommandList &>(cmdList);

        uint64_t signalVal;
        {
            std::scoped_lock lk{_timelineMtx};
            signalVal = ++_timelineValue;
        }

        std::vector<VkSemaphoreSubmitInfo> waits;
        if (desc.WaitFence.Valid())
        {
            waits.push_back({
                .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .semaphore = _timeline,
                .value     = desc.WaitValue > 0 ? desc.WaitValue : desc.WaitFence.Id,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            });
        }

        VkCommandBufferSubmitInfo csi{
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = vcl.commandBuffer(),
        };
        VkSemaphoreSubmitInfo ssi{
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = _timeline,
            .value     = signalVal,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        VkSubmitInfo2 si{
            .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount   = static_cast<uint32_t>(waits.size()),
            .pWaitSemaphoreInfos      = waits.data(),
            .commandBufferInfoCount   = 1,
            .pCommandBufferInfos      = &csi,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos    = &ssi,
        };

        VkQueue     q;
        std::mutex *qmtx;
        switch (vcl.queueType())
        {
            case QueueType::Compute:
                q    = computeQ.q;
                qmtx = &computeQ.mtx;
                break;
            case QueueType::Transfer:
                q    = transferQ.q;
                qmtx = &transferQ.mtx;
                break;
            default:
                q    = gfxQ.q;
                qmtx = &gfxQ.mtx;
                break;
        }
        {
            std::scoped_lock lk{*qmtx};
            VK_CHECK(vkQueueSubmit2(q, 1, &si, VK_NULL_HANDLE));
        }

        return FenceHandle{signalVal};
    }

} // namespace rhi::vulkan
