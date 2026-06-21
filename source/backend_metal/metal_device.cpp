// metal_device.cpp — MetalDevice implementation.
// Pure C++ / metal-cpp — no Objective-C or .mm required.
//
// Targets Metal 4 exclusively (MTL4CommandQueue/MTL4Compiler/MTL4ArgumentTable/
// MTLResidencySet) — see API_GUIDELINES.md's "Target the latest platform API
// version — always" section and metal_internal.h's header comment.
//
// NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION / CA_PRIVATE_IMPLEMENTATION
// are defined in metal_impl.cpp (exactly one TU), NOT here.

module;
// Global module fragment: all system + metal-cpp headers included before
// "module rhi.metal;" to keep libc++ include-guard state consistent and
// avoid __promote_t redefinition with Clang's C++23 module support.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
// Metal.hpp only forward-declares MTL4::AccelerationStructureDescriptor
// (via MTL4ComputeCommandEncoder.hpp) — pull in the concrete MTL4 AS
// descriptor/geometry types here, in the global module fragment.
#include <Metal/MTL4AccelerationStructure.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <dispatch/dispatch.h>
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

// ============================================================================
// CreateDevice — module factory (declared in metal_backend.cppm)
// ============================================================================

namespace rhi
{
    std::unique_ptr<IDevice> CreateDevice(const DeviceDesc &desc)
    {
        return std::make_unique<metal::MetalDevice>(desc);
    }

    SharedDevice AcquireSharedDevice(const DeviceDesc &desc)
    {
        return rhi::AcquireSharedDevice(desc, &CreateDevice);
    }
} // namespace rhi

namespace rhi::metal
{

    // ============================================================================
    // MetalBindlessHeap — out-of-line definitions (kept in this one TU only —
    // see the doc comment on the declarations in metal_internal.h)
    // ============================================================================

    uint32_t MetalBindlessHeap::MaxBuffers() const noexcept
    {
        return BindlessLimits{}.MaxBuffers;
    }
    uint32_t MetalBindlessHeap::MaxTextures() const noexcept
    {
        return BindlessLimits{}.MaxTextures;
    }
    uint32_t MetalBindlessHeap::MaxSamplers() const noexcept
    {
        return BindlessLimits{}.MaxSamplers;
    }
    GpuAddress MetalBindlessHeap::HeapAddress() const noexcept
    {
        return {};
    }
    uint32_t MetalBindlessHeap::UsedBuffers() const noexcept
    {
        return _usedBuffers.load(std::memory_order_relaxed);
    }
    uint32_t MetalBindlessHeap::UsedTextures() const noexcept
    {
        return _usedTextures.load(std::memory_order_relaxed);
    }
    uint32_t MetalBindlessHeap::UsedSamplers() const noexcept
    {
        return _usedSamplers.load(std::memory_order_relaxed);
    }
    void MetalBindlessHeap::RegisterBuffer() noexcept
    {
        _usedBuffers.fetch_add(1, std::memory_order_relaxed);
    }
    void MetalBindlessHeap::RegisterTexture() noexcept
    {
        _usedTextures.fetch_add(1, std::memory_order_relaxed);
    }
    void MetalBindlessHeap::RegisterSampler() noexcept
    {
        _usedSamplers.fetch_add(1, std::memory_order_relaxed);
    }
    void MetalBindlessHeap::UnregisterBuffer() noexcept
    {
        _usedBuffers.fetch_sub(1, std::memory_order_relaxed);
    }
    void MetalBindlessHeap::UnregisterTexture() noexcept
    {
        _usedTextures.fetch_sub(1, std::memory_order_relaxed);
    }
    void MetalBindlessHeap::UnregisterSampler() noexcept
    {
        _usedSamplers.fetch_sub(1, std::memory_order_relaxed);
    }

    // ============================================================================
    // MetalDevice — out-of-line accessor definitions
    // ============================================================================

    std::string_view MetalDevice::AdapterName() const noexcept
    {
        return _adapterName;
    }
    uint64_t MetalDevice::VideoMemoryBytes() const noexcept
    {
        return _videoMemoryBytes;
    }
    IBindlessHeap &MetalDevice::BindlessHeap() noexcept
    {
        return _heap;
    }
    MTL::Device *MetalDevice::MtlDevice() const noexcept
    {
        return _device.get();
    }
    MTL4::CommandQueue *MetalDevice::Mtl4Queue() const noexcept
    {
        return _queue.get();
    }

    bool MetalDevice::DebugCaptureEnabled() const noexcept
    {
        return _debugCaptureEnabled;
    }
    MetalBuffer &MetalDevice::Buffer(BufferHandle h)
    {
        return _buffers.get(h.Index);
    }
    MetalTexture &MetalDevice::Texture(TextureHandle h)
    {
        return _textures.get(h.Index);
    }
    MetalSampler &MetalDevice::Sampler(SamplerHandle h)
    {
        return _samplers.get(h.Index);
    }
    MetalPipeline &MetalDevice::Pipeline(PipelineHandle h)
    {
        return _pipelines.get(h.Index);
    }
    MetalAccelerationStructure &MetalDevice::AccelStruct(AccelerationStructureHandle h)
    {
        return _accelStructs.get(h.Index);
    }
    FenceHandle MetalDevice::NextFence() noexcept
    {
        return FenceHandle{++_timelineValue};
    }
    MTL::SharedEvent *MetalDevice::TimelineEvent() const noexcept
    {
        return _timelineEvent.get();
    }
    MetalCommandResources MetalDevice::AcquireCommandResources()
    {
        {
            const uint64_t   completedFence{_timelineEvent->signaledValue()};
            std::scoped_lock lock{_commandResourcePoolMutex};
            for (std::size_t i{0}; i < _commandResourcePool.size(); ++i)
            {
                const auto fence{_commandResourcePool[i].CompletionFence};
                if (!fence.Valid() || fence.Id <= completedFence)
                {
                    MetalCommandResources resources{std::move(_commandResourcePool[i])};
                    if (i + 1 != _commandResourcePool.size())
                    {
                        _commandResourcePool[i] = std::move(_commandResourcePool.back());
                    }
                    _commandResourcePool.pop_back();
                    return resources;
                }
            }
        }

        MetalCommandResources resources{
            .Allocator     = NS::TransferPtr(_device->newCommandAllocator()),
            .CommandBuffer = NS::TransferPtr(_device->newCommandBuffer()),
        };
        if (!resources.Allocator)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newCommandAllocator failed");
        }
        if (!resources.CommandBuffer)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newCommandBuffer failed");
        }
        return resources;
    }
    void MetalDevice::RecycleCommandResources(MetalCommandResources &&resources)
    {
        std::scoped_lock lock{_commandResourcePoolMutex};
        _commandResourcePool.push_back(std::move(resources));
    }
    MTL::CommandQueue *MetalDevice::LegacyQueue() const noexcept
    {
        return _legacyQueue.get();
    }

    // ============================================================================
    // MetalDevice — residency helpers
    // ============================================================================

    void MetalDevice::_addResident(MTL::Allocation *res)
    {
        _residencySet->addAllocation(res);
        _residencySet->commit();
    }

    void MetalDevice::_removeResident(MTL::Allocation *res)
    {
        _residencySet->removeAllocation(res);
        _residencySet->commit();
    }

    // ============================================================================
    // MetalDevice — constructor / destructor
    // ============================================================================

    MetalDevice::MetalDevice(const DeviceDesc &desc)
    {
        _debugCaptureEnabled  = desc.EnableValidation;
        _gpuValidationEnabled = desc.EnableGpuValidation;
        _device               = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        if (!_device)
        {
            throw std::runtime_error("[LightRHI] No Metal device found");
        }

        if (_debugCaptureEnabled)
        {
            auto queueDesc{NS::TransferPtr(MTL4::CommandQueueDescriptor::alloc()->init())};
            queueDesc->setLabel(NS::String::string("HdRestir / LightRHI compute queue", NS::UTF8StringEncoding));
            NS::Error *queueError{nullptr};
            _queue = NS::TransferPtr(_device->newMTL4CommandQueue(queueDesc.get(), &queueError));
        }
        else
        {
            _queue = NS::TransferPtr(_device->newMTL4CommandQueue());
        }
        if (!_queue)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newMTL4CommandQueue failed");
        }

        // Classic queue used only for the acceleration-structure build
        // exception — see LegacyQueue()'s doc comment in metal_internal.h.
        _legacyQueue = NS::TransferPtr(_device->newCommandQueue());
        if (!_legacyQueue)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newCommandQueue (legacy AS queue) failed");
        }
        if (_debugCaptureEnabled)
        {
            _legacyQueue->setLabel(
                NS::String::string("HdRestir / acceleration structure build queue", NS::UTF8StringEncoding));
        }

        {
            auto       compilerDesc{NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init())};
            NS::Error *err{nullptr};
            _compiler = NS::TransferPtr(_device->newCompiler(compilerDesc.get(), &err));
            if (!_compiler)
            {
                const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
                throw std::runtime_error(std::string{"[LightRHI] MTL4::Device::newCompiler failed: "} + msg);
            }
        }

        auto       residencyDesc{NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init())};
        NS::Error *err{nullptr};
        _residencySet = NS::TransferPtr(_device->newResidencySet(residencyDesc.get(), &err));
        if (!_residencySet)
        {
            const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
            throw std::runtime_error(std::string{"[LightRHI] MTLDevice::newResidencySet failed: "} + msg);
        }
        _queue->addResidencySet(_residencySet.get());

        _adapterName      = std::string{_device->name()->utf8String()};
        _videoMemoryBytes = _device->recommendedMaxWorkingSetSize();

        _timelineEvent = NS::TransferPtr(_device->newSharedEvent());

        _raytracingSupported = _device->supportsRaytracing();

        // GPU/shader validation can also be forced process-wide through
        // Xcode's scheme flags / the Metal environment; EnableGpuValidation
        // additionally lets a DeviceDesc request it directly (wired into
        // every pipeline's MTL4::PipelineOptions — see
        // _configurePipelineForDebugging). EnableValidation is unrelated:
        // it only enables capture-only metadata (queue labels, argument-
        // table zero-init, shader reflection).
    }

    MetalDevice::~MetalDevice()
    {
        WaitIdle();
    }

    MTL::CaptureScope *MetalDevice::SubmissionCaptureScope(std::string_view name)
    {
        const std::string           key{name.empty() ? std::string_view{"LightRHI submission"} : name};
        std::lock_guard<std::mutex> lock{_captureScopeMutex};
        auto [it, inserted]{_submissionCaptureScopes.try_emplace(key)};
        if (inserted)
        {
            auto *manager = MTL::CaptureManager::sharedCaptureManager();
            it->second    = NS::TransferPtr(manager->newCaptureScope(_queue.get()));
            if (it->second)
            {
                it->second->setLabel(NS::String::string(key.c_str(), NS::UTF8StringEncoding));
            }
        }
        return it->second.get();
    }

    void MetalDevice::BeginCaptureScope(std::string_view name)
    {
        const std::string key{name.empty() ? std::string_view{"LightRHI queue frame"} : name};

        // A FRESH scope every call — deliberately NOT cached/reused by name
        // like SubmissionCaptureScope above. FrameCaptureScope
        // (source/gpu_functions/gpu_frame_capture_scope.cpp) constructs one
        // of these per Render() call, always with the same literal name; a
        // cached-by-name scope would mean every frame of the whole process
        // shares one MTL::CaptureScope object, so Xcode's capture UI shows
        // one scope begun/ended hundreds of times with no way to tell which
        // begin/end pair corresponds to the frame you actually captured —
        // triggering a capture mid-session lands on an arbitrary begin/end
        // cycle whose recorded command buffers don't correspond to one
        // coherent frame, which is a plausible cause of both replay crashes
        // and the capture hanging (Xcode waiting on a scope-end signal from
        // a scope that's been reused since).
        NS::SharedPtr<MTL::CaptureScope> scope;
        {
            std::lock_guard<std::mutex> lock{_captureScopeMutex};
            auto                       *manager = MTL::CaptureManager::sharedCaptureManager();
            // Capture only LightRHI's MTL4 queue. A device-wide scope also
            // records unrelated presentation work from the host process
            // (for example usdview's Metal-backed OpenGL renderer), which
            // obscures HdRestir's dependency graph and can make Xcode's
            // replay service fail while rebuilding those render pipelines.
            scope = NS::TransferPtr(manager->newCaptureScope(_queue.get()));
            if (scope)
            {
                scope->setLabel(NS::String::string(key.c_str(), NS::UTF8StringEncoding));
            }
            _frameCaptureScope  = scope;
            _activeCaptureScope = scope.get();
        }
        if (_activeCaptureScope)
        {
            MTL::CaptureManager::sharedCaptureManager()->setDefaultCaptureScope(_activeCaptureScope);
        }
        if (_activeCaptureScope)
        {
            _activeCaptureScope->beginScope();
        }
    }

    void MetalDevice::EndCaptureScope()
    {
        if (_activeCaptureScope)
        {
            _activeCaptureScope->endScope();
            _activeCaptureScope = nullptr;
        }
        std::lock_guard<std::mutex> lock{_captureScopeMutex};
        _frameCaptureScope.reset();
    }

    void MetalDevice::SuspendActiveCaptureScope() noexcept
    {
        if (_activeCaptureScope)
        {
            _activeCaptureScope->endScope();
        }
    }

    void MetalDevice::ResumeActiveCaptureScope() noexcept
    {
        if (_activeCaptureScope)
        {
            _activeCaptureScope->beginScope();
        }
    }

    // ============================================================================
    // Buffer
    // ============================================================================

    BufferHandle MetalDevice::CreateBuffer(const BufferDesc &desc)
    {
        auto *buf = _device->newBuffer(desc.Size, _toOptions(desc.MemoryType));
        if (!buf)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newBuffer failed");
        }

        if (!desc.DebugName.empty())
        {
            buf->setLabel(NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding));
        }

        uint32_t idx{_buffers.alloc()};
        _buffers.get(idx) = MetalBuffer{NS::TransferPtr(buf), desc.Size, desc.Usage};

        _addResident(buf);
        _heap.RegisterBuffer();

        // Track base GPU address -> slot so acceleration-structure geometry
        // (addressed by GpuAddress/BDA) can be resolved back to an MTL::Buffer*
        // + offset — see _bufferAndOffsetFromAddress.
        {
            std::scoped_lock lk{_bufferAddrMutex};
            _bufferAddrToIndex[buf->gpuAddress()] = idx;
        }

        return BufferHandle{idx};
    }

    void MetalDevice::DestroyBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        {
            std::scoped_lock lk{_bufferAddrMutex};
            _bufferAddrToIndex.erase(_buffers.get(h.Index).buffer->gpuAddress());
        }
        _removeResident(_buffers.get(h.Index).buffer.get());
        _heap.UnregisterBuffer();
        _buffers.free(h.Index);
    }

    GpuAddress MetalDevice::BufferAddress(BufferHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{_buffers.get(h.Index).buffer->gpuAddress()};
    }

    BufferInfo MetalDevice::GetBufferInfo(BufferHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        const auto &b = _buffers.get(h.Index);
        return BufferInfo{.Size = b.size, .Usage = b.usage, .DeviceAddress = GpuAddress{b.buffer->gpuAddress()}};
    }

    MappedBuffer MetalDevice::MapBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return {};
        }
        auto &b = _buffers.get(h.Index);
        return MappedBuffer{b.buffer->contents(), b.size};
    }

    void MetalDevice::UnmapBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &b = _buffers.get(h.Index);
        if (b.buffer->storageMode() == MTL::StorageModeManaged)
        {
            b.buffer->didModifyRange(NS::Range::Make(0, b.size));
        }
    }

    // ============================================================================
    // Texture
    // ============================================================================

    TextureHandle MetalDevice::CreateTexture(const TextureDesc &desc)
    {
        auto mtd{NS::TransferPtr(MTL::TextureDescriptor::alloc()->init())};
        mtd->setTextureType(_toTexType(desc.Dimension));
        mtd->setPixelFormat(_toPixFmt(desc.Format));
        mtd->setWidth(desc.Extent.Width);
        mtd->setHeight(desc.Extent.Height);
        mtd->setDepth(desc.Extent.Depth);
        mtd->setMipmapLevelCount(desc.MipLevels);
        mtd->setArrayLength(desc.ArrayLayers);
        mtd->setSampleCount(desc.SampleCount);
        mtd->setUsage(_toTexUsage(desc.Usage));
        mtd->setStorageMode(MTL::StorageModePrivate);

        auto tex{NS::TransferPtr(_device->newTexture(mtd.get()))};
        if (!tex)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newTexture failed");
        }

        if (!desc.DebugName.empty())
        {
            tex->setLabel(NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding));
        }

        uint32_t idx{_textures.alloc()};
        _addResident(tex.get());
        _heap.RegisterTexture();
        _textures.get(idx) = MetalTexture{std::move(tex), desc};
        return TextureHandle{idx};
    }

    void MetalDevice::DestroyTexture(TextureHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _removeResident(_textures.get(h.Index).texture.get());
        _heap.UnregisterTexture();
        _textures.free(h.Index);
    }

    GpuAddress MetalDevice::TextureAddress(TextureHandle h) const
    {
        // The bindless mechanism for a sampled texture on Metal: the
        // MTLTexture's own gpuResourceID. Store these bits in a Slang
        // DescriptorHandle<Texture2D> in root/scene data; no per-dispatch
        // texture bind is needed. The queue-attached residency set makes the
        // indirectly referenced texture accessible to the GPU.
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{_textures.get(h.Index).texture->gpuResourceID()._impl};
    }

    // ============================================================================
    // Sampler
    // ============================================================================

    SamplerHandle MetalDevice::CreateSampler(const SamplerDesc &desc)
    {
        auto sd{NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init())};
        sd->setMinFilter(_toMinMag(desc.MinFilter));
        sd->setMagFilter(_toMinMag(desc.MagFilter));
        sd->setMipFilter(_toMipFlt(desc.MipMode));
        sd->setSAddressMode(_toAddrMode(desc.AddressU));
        sd->setTAddressMode(_toAddrMode(desc.AddressV));
        sd->setRAddressMode(_toAddrMode(desc.AddressW));
        sd->setMaxAnisotropy(desc.Anisotropy ? static_cast<NS::UInteger>(desc.MaxAniso) : 1);
        sd->setCompareFunction(desc.CompareEnable ? _toCompare(desc.CompareOp) : MTL::CompareFunctionNever);
        sd->setLodMinClamp(desc.MinLod);
        sd->setLodMaxClamp(desc.MaxLod);
        sd->setLodAverage(false);

        if (!desc.DebugName.empty())
        {
            sd->setLabel(NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding));
        }

        auto smp{NS::TransferPtr(_device->newSamplerState(sd.get()))};
        if (!smp)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newSamplerState failed");
        }

        uint32_t idx{_samplers.alloc()};
        _samplers.get(idx) = MetalSampler{std::move(smp)};
        _heap.RegisterSampler();
        // MTL::SamplerState is not a MTL::Resource/Allocation — no residency
        // tracking needed (unchanged from classic Metal).
        return SamplerHandle{idx};
    }

    void MetalDevice::DestroySampler(SamplerHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _heap.UnregisterSampler();
        _samplers.free(h.Index);
    }

    GpuAddress MetalDevice::SamplerAddress(SamplerHandle h) const
    {
        // Samplers bind via ICommandList::BindSampler -> MTL4::ArgumentTable::
        // setSamplerState(handle-derived gpuResourceID, index) — see
        // metal_internal.h's MetalDevice header comment. This method is kept
        // for IDevice interface completeness; MTLSamplerState has no
        // separately-embeddable address the way a buffer/texture does, so
        // callers needing the underlying resource ID go through Sampler(h)
        // internally rather than this method.
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{h.Index};
    }

    // ============================================================================
    // Ray tracing
    // ============================================================================

    MTL::Buffer *MetalDevice::_bufferAndOffsetFromAddress(GpuAddress addr, uint64_t &outOffset) const
    {
        outOffset = 0;
        if (!addr.Valid())
        {
            return nullptr;
        }
        // Exact-key lookup: AccelerationStructureDesc always addresses
        // vertex/index data by a buffer's own base address (device.cppm's
        // BlasFromTriangleBuffer() takes IDevice::BufferAddress()'s result
        // directly, never an offset sub-range), so there is no "closest base
        // below addr" case to resolve — O(1) instead of scanning every buffer.
        std::scoped_lock lk{_bufferAddrMutex};
        auto             it{_bufferAddrToIndex.find(addr.Address)};
        if (it == _bufferAddrToIndex.end())
        {
            return nullptr;
        }
        outOffset = 0;
        return _buffers.get(it->second).buffer.get();
    }

    NS::SharedPtr<MTL::AccelerationStructureDescriptor>
    MetalDevice::MakeAccelerationStructureDescriptor(const AccelerationStructureDesc        &desc,
                                                     std::vector<NS::SharedPtr<NS::Object>> &keepAlive)
    {
        if (desc.Type == AccelerationStructureType::BottomLevel)
        {
            uint64_t     vtxOffset{0};
            MTL::Buffer *vtxBuf{_bufferAndOffsetFromAddress(desc.VertexBufferAddress, vtxOffset)};
            if (!vtxBuf)
            {
                throw std::runtime_error("[LightRHI] BlasFromTriangleBuffer: VertexBufferAddress does not resolve "
                                         "to a live LightRHI buffer");
            }

            auto tri = NS::TransferPtr(MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init());
            tri->setVertexBuffer(vtxBuf);
            tri->setVertexBufferOffset(vtxOffset);
            tri->setVertexStride(desc.VertexStride);
            tri->setVertexFormat(MTL::AttributeFormatFloat3);
            tri->setTriangleCount(desc.IndexBufferAddress.Valid() ? desc.IndexCount / 3 : desc.VertexCount / 3);
            tri->setOpaque(true);

            if (desc.IndexBufferAddress.Valid())
            {
                uint64_t     idxOffset{0};
                MTL::Buffer *idxBuf{_bufferAndOffsetFromAddress(desc.IndexBufferAddress, idxOffset)};
                if (!idxBuf)
                {
                    throw std::runtime_error(
                        "[LightRHI] BlasFromTriangleBuffer: IndexBufferAddress does not resolve to a live buffer");
                }
                tri->setIndexBuffer(idxBuf);
                tri->setIndexBufferOffset(idxOffset);
                tri->setIndexType(desc.IndexType == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32);
            }

            NS::Object *geomArr[]{tri.get()};
            auto        geoms = NS::RetainPtr(NS::Array::array(geomArr, 1));

            auto pd = NS::TransferPtr(MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init());
            pd->setGeometryDescriptors(geoms.get());
            pd->setUsage(desc.PreferFastTrace ? MTL::AccelerationStructureUsageNone
                                              : MTL::AccelerationStructureUsagePreferFastBuild);

            keepAlive.push_back(tri);
            keepAlive.push_back(geoms);
            return pd;
        }

        // TopLevel: build an instance-descriptor buffer host-side. Metal
        // permits duplicate BLAS entries, so no dedup pass is needed for
        // correctness (see this method's doc comment in metal_internal.h
        // for why this classic path exists at all, in an otherwise-MTL4
        // backend).
        //
        // UserID descriptors, not Default: Slang's RayQuery
        // CommittedInstanceID()/CandidateInstanceID() lower to Metal's
        // get_{committed,candidate}_user_instance_id(), which reads the
        // descriptor's userID field — a field the Default descriptor type
        // simply does not have, so InstanceCustomIndex would be silently
        // dropped (every instance reads back as 0). Matches the Vulkan
        // backend, which maps InstanceCustomIndex to
        // VkAccelerationStructureInstanceKHR::instanceCustomIndex.
        const uint64_t instanceBufBytes{sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor) *
                                        desc.Instances.size()};
        auto           instanceBuf = NS::TransferPtr(
            _device->newBuffer(std::max<uint64_t>(instanceBufBytes, 1), MTL::ResourceStorageModeShared));
        auto *dst = static_cast<MTL::AccelerationStructureUserIDInstanceDescriptor *>(instanceBuf->contents());
        std::vector<NS::Object *> blasObjs(desc.Instances.size());
        for (size_t i = 0; i < desc.Instances.size(); ++i)
        {
            const auto &inst = desc.Instances[i];
            if (!inst.Blas.Valid())
            {
                throw std::runtime_error("[LightRHI] TlasFromInstances: AccelerationStructureInstance.Blas is not "
                                         "a valid handle");
            }

            MTL::PackedFloat4x3 xform{
                MTL::PackedFloat3{inst.Transform[0][0], inst.Transform[1][0], inst.Transform[2][0]},
                MTL::PackedFloat3{inst.Transform[0][1], inst.Transform[1][1], inst.Transform[2][1]},
                MTL::PackedFloat3{inst.Transform[0][2], inst.Transform[1][2], inst.Transform[2][2]},
                MTL::PackedFloat3{inst.Transform[0][3], inst.Transform[1][3], inst.Transform[2][3]},
            };
            blasObjs[i] = AccelStruct(inst.Blas).as.get();
            dst[i]      = MTL::AccelerationStructureUserIDInstanceDescriptor{
                .transformationMatrix            = xform,
                .options                         = static_cast<MTL::AccelerationStructureInstanceOptions>(inst.Flags),
                .mask                            = inst.InstanceMask,
                .intersectionFunctionTableOffset = 0,
                .accelerationStructureIndex      = static_cast<uint32_t>(i),
                .userID                          = inst.InstanceCustomIndex,
            };
        }

        auto blasArr = NS::RetainPtr(NS::Array::array(blasObjs.data(), blasObjs.size()));

        auto id = NS::TransferPtr(MTL::InstanceAccelerationStructureDescriptor::alloc()->init());
        id->setInstanceCount(desc.Instances.size());
        id->setInstanceDescriptorBuffer(instanceBuf.get());
        id->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
        id->setInstancedAccelerationStructures(blasArr.get());
        id->setUsage(desc.PreferFastTrace ? MTL::AccelerationStructureUsageNone
                                          : MTL::AccelerationStructureUsagePreferFastBuild);

        keepAlive.push_back(instanceBuf);
        keepAlive.push_back(blasArr);
        return id;
    }

    bool MetalDevice::SupportsRayTracing() const noexcept
    {
        return _raytracingSupported;
    }

    AccelerationStructureBuildSizes
    MetalDevice::QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &desc) const
    {
        if (!_raytracingSupported)
        {
            throw std::runtime_error("[LightRHI] QueryAccelerationStructureBuildSizes: device does not support ray "
                                     "tracing (MTLDevice::supportsRaytracing() == false)");
        }

        std::vector<NS::SharedPtr<NS::Object>> keepAlive;
        auto descriptor{const_cast<MetalDevice *>(this)->MakeAccelerationStructureDescriptor(desc, keepAlive)};
        MTL::AccelerationStructureSizes sizes{_device->accelerationStructureSizes(descriptor.get())};
        return AccelerationStructureBuildSizes{
            .AccelerationStructureSize = sizes.accelerationStructureSize,
            .BuildScratchSize          = sizes.buildScratchBufferSize,
            .UpdateScratchSize         = sizes.refitScratchBufferSize,
        };
    }

    AccelerationStructureHandle MetalDevice::CreateAccelerationStructure(const AccelerationStructureDesc &desc)
    {
        if (!_raytracingSupported)
        {
            throw std::runtime_error("[LightRHI] CreateAccelerationStructure: device does not support ray tracing "
                                     "(MTLDevice::supportsRaytracing() == false)");
        }

        auto  sizes{QueryAccelerationStructureBuildSizes(desc)};
        auto *as = _device->newAccelerationStructure(sizes.AccelerationStructureSize);
        if (!as)
        {
            throw std::runtime_error("[LightRHI] MTLDevice::newAccelerationStructure failed");
        }
        if (!desc.DebugName.empty())
        {
            as->setLabel(NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding));
        }

        uint32_t idx{_accelStructs.alloc()};
        _accelStructs.get(idx) =
            MetalAccelerationStructure{NS::TransferPtr(as), desc.Type, sizes.AccelerationStructureSize};
        _addResident(as);
        return AccelerationStructureHandle{idx};
    }

    void MetalDevice::DestroyAccelerationStructure(AccelerationStructureHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _removeResident(_accelStructs.get(h.Index).as.get());
        _accelStructs.free(h.Index);
    }

    GpuAddress MetalDevice::AccelerationStructureAddress(AccelerationStructureHandle h) const
    {
        // The AS's bindless GPU handle: the MTLAccelerationStructure resource
        // ID bits. Shaders receive it in push constants as a
        // DescriptorHandle<RaytracingAccelerationStructure> field, which Metal
        // reads with argument-buffer semantics (an 8-byte resource ID in
        // buffer memory) — see the doc comment in device.cppm. Residency for
        // this indirect access is handled by the device's persistent
        // MTL::ResidencySet (_addResident, called from CreateAccelerationStructure).
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{_accelStructs.get(h.Index).as->gpuResourceID()._impl};
    }

    // ============================================================================
    // Shader loading helper
    // ============================================================================

    MTL::Library *MetalDevice::_loadLibrary(const ShaderDesc &sd)
    {
        // Dispatch on which bytecode alternative is active.
        // std::visit with an if-constexpr generic lambda — no RTTI required.
        NS::Error *err{nullptr};

        return std::visit(
            [&](auto &&src) -> MTL::Library *
            {
                using T = std::decay_t<decltype(src)>;

                if constexpr (std::is_same_v<T, std::monostate>)
                {
                    throw std::runtime_error("[LightRHI] _loadLibrary: ShaderDesc has no bytecode (monostate)");
                }
                else if constexpr (std::is_same_v<T, SpirvBytecode>)
                {
                    throw std::runtime_error("[LightRHI] _loadLibrary: Metal backend received SPIR-V ShaderDesc");
                }
                else if constexpr (std::is_same_v<T, MetalLibBytecode>)
                {
                    // Pre-compiled .metallib bytes — fastest path (shipping builds).
                    dispatch_data_t dd{dispatch_data_create(src.Bytes.data(), src.Bytes.size(), nullptr,
                                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT)};
                    auto           *lib = _device->newLibrary(dd, &err);
                    // dispatch_data_create returns a +1 reference; balance it.
                    dispatch_release(dd);
                    if (!lib)
                    {
                        const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
                        throw std::runtime_error(std::string{"[LightRHI] MTL metallib load: "} + msg);
                    }
                    return lib;
                }
                else // MslSource — compile at runtime (dev builds / tests)
                {
                    static_assert(std::is_same_v<T, MslSource>);
                    // Source may not be NUL-terminated; use the explicit-length
                    // NS::String initializer so we don't overrun the buffer.
                    // alloc()->init(...) here is itself a Create-Rule call
                    // (NS::String::alloc() takes the +1 ref; init(...) is the
                    // designated initializer, not a second allocation), so
                    // this is a single object to transfer ownership of.
                    auto str{NS::TransferPtr(NS::String::alloc()->init(
                        const_cast<char *>(src.Source.data()), src.Source.size(), NS::UTF8StringEncoding, false))};
                    auto opts{NS::TransferPtr(MTL::CompileOptions::alloc()->init())};
                    opts->setLanguageVersion(MTL::LanguageVersion3_0);
                    auto *lib = _device->newLibrary(str.get(), opts.get(), &err);
                    if (!lib)
                    {
                        const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
                        throw std::runtime_error(std::string{"[LightRHI] MTL shader compile: "} + msg);
                    }
                    return lib;
                }
            },
            sd.Bytecode);
    }

    // ============================================================================
    // Graphics pipeline
    // ============================================================================

    void MetalDevice::_configurePipelineForDebugging(MTL4::PipelineDescriptor *descriptor) const
    {
        if (!_debugCaptureEnabled && !_gpuValidationEnabled)
        {
            return;
        }
        auto options{NS::TransferPtr(MTL4::PipelineOptions::alloc()->init())};
        if (_debugCaptureEnabled)
        {
            options->setShaderReflection(static_cast<MTL4::ShaderReflection>(MTL4::ShaderReflectionBindingInfo |
                                                                             MTL4::ShaderReflectionBufferTypeInfo));
        }
        if (_gpuValidationEnabled)
        {
            // DeviceDesc::EnableGpuValidation — slow (per-shader-invocation
            // bounds/UB checking), off by default. Xcode's own scheme-level
            // "GPU Shader Validation" flag does the same thing process-wide;
            // this lets it be requested per-DeviceDesc instead, e.g. from a
            // debug build that always wants it without depending on the
            // launching scheme/environment.
            options->setShaderValidation(MTL::ShaderValidationEnabled);
        }
        // setOptions retains its own strong reference to options (standard
        // Cocoa setter convention — same as every other setFoo(descriptor)
        // call in this file), so options is safe to release once this scope
        // ends.
        descriptor->setOptions(options.get());
    }

    PipelineHandle MetalDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc &desc)
    {
        auto pd{NS::TransferPtr(MTL4::RenderPipelineDescriptor::alloc()->init())};
        _configurePipelineForDebugging(pd.get());

        // Vertex shader — skip if not provided (monostate = "no shader")
        if (!std::holds_alternative<std::monostate>(desc.VertexShader.Bytecode))
        {
            auto  vsLib  = NS::TransferPtr(_loadLibrary(desc.VertexShader));
            auto *name   = NS::String::string(desc.VertexShader.EntryPoint.data(), NS::UTF8StringEncoding);
            auto  vsDesc = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
            vsDesc->setLibrary(vsLib.get());
            vsDesc->setName(name);
            pd->setVertexFunctionDescriptor(vsDesc.get());
        }

        // Fragment shader — skip if not provided (monostate = "no shader")
        if (!std::holds_alternative<std::monostate>(desc.FragmentShader.Bytecode))
        {
            auto  fsLib  = NS::TransferPtr(_loadLibrary(desc.FragmentShader));
            auto *name   = NS::String::string(desc.FragmentShader.EntryPoint.data(), NS::UTF8StringEncoding);
            auto  fsDesc = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
            fsDesc->setLibrary(fsLib.get());
            fsDesc->setName(name);
            pd->setFragmentFunctionDescriptor(fsDesc.get());
        }

        // Topology class (Metal PSO needs it for tessellation; rasterized topology set at draw)
        pd->setInputPrimitiveTopology(_toTopology(desc.Topology));

        // Color attachments
        for (size_t i = 0; i < desc.ColorFormats.size(); ++i)
        {
            auto *ca = pd->colorAttachments()->object(static_cast<NS::UInteger>(i));
            ca->setPixelFormat(_toPixFmt(desc.ColorFormats[i]));
            if (i < desc.ColorBlend.size() && desc.ColorBlend[i].Enable)
            {
                const auto &bs = desc.ColorBlend[i];
                ca->setBlendingState(MTL4::BlendStateEnabled);
                ca->setSourceRGBBlendFactor(_toBlendF(bs.SrcColor));
                ca->setDestinationRGBBlendFactor(_toBlendF(bs.DstColor));
                ca->setRgbBlendOperation(_toBlendOp(bs.ColorOp));
                ca->setSourceAlphaBlendFactor(_toBlendF(bs.SrcAlpha));
                ca->setDestinationAlphaBlendFactor(_toBlendF(bs.DstAlpha));
                ca->setAlphaBlendOperation(_toBlendOp(bs.AlphaOp));
            }
        }

        // Depth/stencil pixel format is no longer part of the pipeline
        // descriptor under MTL4 (MTL4::RenderPipelineDescriptor has no
        // setDepthAttachmentPixelFormat) — it moves to MTL4::RenderPassDescriptor's
        // depthAttachment(), supplied at BeginRendering time (MetalCommandList).

        pd->setRasterSampleCount(desc.SampleCount);

        if (!desc.DebugName.empty())
        {
            pd->setLabel(NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding));
        }

        NS::Error *err{nullptr};
        auto pso{NS::TransferPtr(_compiler->newRenderPipelineState(pd.get(), /*compilerTaskOptions*/ nullptr, &err))};

        if (!pso)
        {
            const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
            throw std::runtime_error(std::string{"[LightRHI] RenderPipelineState: "} + msg);
        }

        // Depth-stencil state
        NS::SharedPtr<MTL::DepthStencilState> dss;
        {
            auto dsd{NS::TransferPtr(_makeDepthStencilDesc(desc.DepthStencil, _device.get()))};
            dss = NS::TransferPtr(_device->newDepthStencilState(dsd.get()));
        }

        uint32_t idx{_pipelines.alloc()};
        _pipelines.get(idx) = MetalPipeline{
            .renderPso         = std::move(pso),
            .computePso        = {},
            .depthStencilState = std::move(dss),
            .winding           = _toWinding(desc.Rasterizer.FrontFace),
            .cullMode          = _toCull(desc.Rasterizer.CullMode),
            .fillMode          = (desc.Rasterizer.FillMode == FillMode::Wireframe) ? MTL::TriangleFillModeLines
                                                                                   : MTL::TriangleFillModeFill,
            .depthBiasConstant = desc.Rasterizer.DepthBiasConstant,
            .depthBiasSlope    = desc.Rasterizer.DepthBiasSlope,
            .isCompute         = false,
        };
        return PipelineHandle{idx};
    }

    // ============================================================================
    // Compute pipeline
    // ============================================================================

    PipelineHandle MetalDevice::CreateComputePipeline(const ComputePipelineDesc &desc)
    {
        auto  lib  = NS::TransferPtr(_loadLibrary(desc.Shader));
        auto *name = NS::String::string(desc.Shader.EntryPoint.data(), NS::UTF8StringEncoding);

        auto fnDesc{NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init())};
        fnDesc->setLibrary(lib.get());
        fnDesc->setName(name);

        auto pipelineDesc{NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init())};
        pipelineDesc->setComputeFunctionDescriptor(fnDesc.get());
        _configurePipelineForDebugging(pipelineDesc.get());
        if (!desc.DebugName.empty())
        {
            pipelineDesc->setLabel(NS::String::string(desc.DebugName.data(), NS::UTF8StringEncoding));
        }

        // Resource binding needs no pipeline reflection. Bindless resources
        // travel as native addresses/resource IDs in root data; explicitly
        // slotted resources use the command list's MTL4 argument table.
        NS::Error *err{nullptr};
        auto       pso = NS::TransferPtr(
            _compiler->newComputePipelineState(pipelineDesc.get(), /*compilerTaskOptions*/ nullptr, &err));

        if (!pso)
        {
            const char *msg{err ? err->localizedDescription()->utf8String() : "unknown"};
            throw std::runtime_error(std::string{"[LightRHI] ComputePipelineState: "} + msg);
        }

        // Each axis defaults independently: an explicit non-zero component
        // from the caller wins, a zero component (including the ComputePipelineDesc
        // default of {0,0,0}) falls back to 1 — except X, which falls back to
        // the shader's own [[max_total_threads_per_threadgroup]] annotation.
        uint32_t tgX{desc.ThreadGroupSize.Width != 0 ? desc.ThreadGroupSize.Width
                                                     : static_cast<uint32_t>(pso->maxTotalThreadsPerThreadgroup())};
        uint32_t tgY{desc.ThreadGroupSize.Height != 0 ? desc.ThreadGroupSize.Height : 1};
        uint32_t tgZ{desc.ThreadGroupSize.Depth != 0 ? desc.ThreadGroupSize.Depth : 1};

        uint32_t idx{_pipelines.alloc()};
        _pipelines.get(idx) = MetalPipeline{
            .renderPso         = {},
            .computePso        = std::move(pso),
            .depthStencilState = {},
            .isCompute         = true,
            .threadGroupSizeX  = tgX,
            .threadGroupSizeY  = tgY,
            .threadGroupSizeZ  = tgZ,
        };
        return PipelineHandle{idx};
    }

    void MetalDevice::DestroyPipeline(PipelineHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _pipelines.free(h.Index);
    }

    // ============================================================================
    // Sync
    // ============================================================================

    void MetalDevice::WaitForFence(FenceHandle fence)
    {
        if (!fence.Valid())
        {
            return;
        }
        while (_timelineEvent->signaledValue() < fence.Id)
        {
            std::this_thread::yield();
        }
    }

    bool MetalDevice::IsFenceComplete(FenceHandle fence)
    {
        if (!fence.Valid())
        {
            return true;
        }
        return _timelineEvent->signaledValue() >= fence.Id;
    }

    void MetalDevice::WaitIdle()
    {
        // MTL4 has no waitUntilCompleted() at all: commit an (empty) command
        // buffer, signal the timeline event from the queue after it, and
        // block on the existing polling wait — same signaled-value counter
        // WaitForFence already uses.
        auto allocator{NS::TransferPtr(_device->newCommandAllocator())};
        auto cmdBuffer{NS::TransferPtr(_device->newCommandBuffer())};

        allocator->reset();
        cmdBuffer->beginCommandBuffer(allocator.get());
        cmdBuffer->endCommandBuffer();

        MTL4::CommandBuffer *buffers[1]{cmdBuffer.get()};
        _queue->commit(buffers, 1);

        const uint64_t value{++_timelineValue};
        _queue->signalEvent(_timelineEvent.get(), value);
        _timelineEvent->waitUntilSignaledValue(value, UINT64_MAX);
    }

    // ============================================================================
    // Upload helpers
    // ============================================================================

    void MetalDevice::UploadBuffer(BufferHandle dst, const void *data, uint64_t size, uint64_t dstOffset)
    {
        if (!dst.Valid() || !data || !size)
        {
            return;
        }
        auto &b = _buffers.get(dst.Index);
        if (b.buffer->storageMode() == MTL::StorageModeShared)
        {
            std::memcpy(static_cast<uint8_t *>(b.buffer->contents()) + dstOffset, data, size);
            return;
        }
        // Private storage — staging-buffer copy. MTL4 has no separate blit
        // encoder: copyFromBuffer/copyFromTexture live directly on
        // MTL4::ComputeCommandEncoder now.
        //
        // Labeled (when capture-debugging is on) so Xcode's capture UI can
        // attribute this command buffer/encoder to something other than
        // "unnamed" — these commits bypass MetalCommandList/Submit() (and
        // its SubmissionCaptureScope labeling) entirely, since they're a
        // synchronous CPU-side upload helper, not a caller-visible command
        // list.
        auto staging{NS::TransferPtr(_device->newBuffer(size, MTL::ResourceStorageModeShared))};
        if (!staging)
        {
            throw std::runtime_error("[LightRHI] UploadBuffer staging allocation failed");
        }
        std::memcpy(staging->contents(), data, size);
        // MTL4 resources must be resident explicitly. Unlike buffers created
        // through CreateBuffer(), this short-lived staging allocation is not
        // otherwise part of the device's persistent residency set.
        _addResident(staging.get());
        auto allocator{NS::TransferPtr(_device->newCommandAllocator())};
        auto cmdBuffer{NS::TransferPtr(_device->newCommandBuffer())};

        allocator->reset();
        cmdBuffer->beginCommandBuffer(allocator.get());
        if (_debugCaptureEnabled)
        {
            cmdBuffer->setLabel(NS::String::string("HdRestir / LightRHI UploadBuffer", NS::UTF8StringEncoding));
        }
        auto *enc = cmdBuffer->computeCommandEncoder();
        enc->copyFromBuffer(staging.get(), 0, b.buffer.get(), dstOffset, size);
        enc->endEncoding();
        cmdBuffer->endCommandBuffer();

        MTL4::CommandBuffer *buffers[1]{cmdBuffer.get()};
        _queue->commit(buffers, 1);
        const uint64_t value{++_timelineValue};
        _queue->signalEvent(_timelineEvent.get(), value);
        _timelineEvent->waitUntilSignaledValue(value, UINT64_MAX);
        _removeResident(staging.get());
    }

    void MetalDevice::UploadTexture(TextureHandle dst, const void *data, uint64_t rowPitch, uint64_t slicePitch,
                                    const TextureCopyRegion &region)
    {
        if (!dst.Valid() || !data)
        {
            return;
        }
        uint64_t totalSize{slicePitch > 0 ? slicePitch : rowPitch * region.Extent.Height};
        auto     staging{NS::TransferPtr(_device->newBuffer(totalSize, MTL::ResourceStorageModeShared))};
        if (!staging)
        {
            throw std::runtime_error("[LightRHI] UploadTexture staging allocation failed");
        }
        std::memcpy(staging->contents(), data, totalSize);
        _addResident(staging.get());
        auto &t{_textures.get(dst.Index)};

        auto allocator{NS::TransferPtr(_device->newCommandAllocator())};
        auto cmdBuffer{NS::TransferPtr(_device->newCommandBuffer())};

        allocator->reset();
        cmdBuffer->beginCommandBuffer(allocator.get());
        if (_debugCaptureEnabled)
        {
            cmdBuffer->setLabel(NS::String::string("HdRestir / LightRHI UploadTexture", NS::UTF8StringEncoding));
        }
        auto *enc = cmdBuffer->computeCommandEncoder();
        enc->copyFromBuffer(staging.get(), 0, rowPitch, slicePitch,
                            MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth),
                            t.texture.get(), region.ArrayLayer, region.MipLevel,
                            MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                              static_cast<NS::UInteger>(region.DstOffset.Y),
                                              static_cast<NS::UInteger>(region.DstOffset.Z)));
        enc->endEncoding();
        cmdBuffer->endCommandBuffer();

        MTL4::CommandBuffer *buffers[1]{cmdBuffer.get()};
        _queue->commit(buffers, 1);
        const uint64_t value{++_timelineValue};
        _queue->signalEvent(_timelineEvent.get(), value);
        _timelineEvent->waitUntilSignaledValue(value, UINT64_MAX);
        _removeResident(staging.get());
    }

    // ============================================================================
    // Static conversion helpers
    // ============================================================================

    MTL::ResourceOptions MetalDevice::_toOptions(MemoryType m) noexcept
    {
        switch (m)
        {
            case MemoryType::GpuOnly:
                return MTL::ResourceStorageModePrivate;
            case MemoryType::CpuToGpu:
                return MTL::ResourceStorageModeShared;
            case MemoryType::GpuToCpu:
                return MTL::ResourceStorageModeShared;
        }
        return MTL::ResourceStorageModePrivate;
    }

    MTL::TextureType MetalDevice::_toTexType(TextureDimension d) noexcept
    {
        switch (d)
        {
            case TextureDimension::Tex1D:
                return MTL::TextureType1D;
            case TextureDimension::Tex2D:
                return MTL::TextureType2D;
            case TextureDimension::Tex3D:
                return MTL::TextureType3D;
            case TextureDimension::TexCube:
                return MTL::TextureTypeCube;
            case TextureDimension::Tex1DArray:
                return MTL::TextureType1DArray;
            case TextureDimension::Tex2DArray:
                return MTL::TextureType2DArray;
            case TextureDimension::TexCubeArray:
                return MTL::TextureTypeCubeArray;
        }
        return MTL::TextureType2D;
    }

    MTL::PixelFormat MetalDevice::_toPixFmt(Format f) noexcept
    {
        switch (f)
        {
            case Format::RGBA8Unorm:
                return MTL::PixelFormatRGBA8Unorm;
            case Format::RGBA8Srgb:
                return MTL::PixelFormatRGBA8Unorm_sRGB;
            case Format::BGRA8Unorm:
                return MTL::PixelFormatBGRA8Unorm;
            case Format::BGRA8Srgb:
                return MTL::PixelFormatBGRA8Unorm_sRGB;
            case Format::RGBA16Float:
                return MTL::PixelFormatRGBA16Float;
            case Format::R32Float:
                return MTL::PixelFormatR32Float;
            case Format::RG32Float:
                return MTL::PixelFormatRG32Float;
            case Format::RGBA32Float:
                return MTL::PixelFormatRGBA32Float;
            case Format::R16Float:
                return MTL::PixelFormatR16Float;
            case Format::R8Unorm:
                return MTL::PixelFormatR8Unorm;
            case Format::RG8Unorm:
                return MTL::PixelFormatRG8Unorm;
            case Format::R32Uint:
                return MTL::PixelFormatR32Uint;
            case Format::R16Uint:
                return MTL::PixelFormatR16Uint;
            case Format::D32Float:
                return MTL::PixelFormatDepth32Float;
            case Format::D16Unorm:
                return MTL::PixelFormatDepth16Unorm;
            case Format::D32FloatS8Uint:
                return MTL::PixelFormatDepth32Float_Stencil8;
            case Format::BC1Unorm:
                return MTL::PixelFormatBC1_RGBA;
            case Format::BC1Srgb:
                return MTL::PixelFormatBC1_RGBA_sRGB;
            case Format::BC3Unorm:
                return MTL::PixelFormatBC3_RGBA;
            case Format::BC3Srgb:
                return MTL::PixelFormatBC3_RGBA_sRGB;
            case Format::BC4Unorm:
                return MTL::PixelFormatBC4_RUnorm;
            case Format::BC5Unorm:
                return MTL::PixelFormatBC5_RGUnorm;
            case Format::BC6HUfloat:
                return MTL::PixelFormatBC6H_RGBUfloat;
            case Format::BC6HSfloat:
                return MTL::PixelFormatBC6H_RGBFloat;
            case Format::BC7Unorm:
                return MTL::PixelFormatBC7_RGBAUnorm;
            case Format::BC7Srgb:
                return MTL::PixelFormatBC7_RGBAUnorm_sRGB;
            default:
                return MTL::PixelFormatInvalid;
        }
    }

    MTL::TextureUsage MetalDevice::_toTexUsage(TextureUsage u) noexcept
    {
        MTL::TextureUsage flags{MTL::TextureUsageUnknown};
        if (HasUsage(u, TextureUsage::Sampled))
        {
            flags |= MTL::TextureUsageShaderRead;
        }
        if (HasUsage(u, TextureUsage::Storage))
        {
            flags |= MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
        }
        if (HasUsage(u, TextureUsage::RenderTarget))
        {
            flags |= MTL::TextureUsageRenderTarget;
        }
        if (HasUsage(u, TextureUsage::DepthStencil))
        {
            flags |= MTL::TextureUsageRenderTarget;
        }
        // Metal uses render passes to clear textures, so TransferDst implies RenderTarget.
        if (HasUsage(u, TextureUsage::TransferDst))
        {
            flags |= MTL::TextureUsageRenderTarget;
        }
        return flags;
    }

    MTL::SamplerMinMagFilter MetalDevice::_toMinMag(SamplerFilter f) noexcept
    {
        return f == SamplerFilter::Linear ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest;
    }

    MTL::SamplerMipFilter MetalDevice::_toMipFlt(SamplerMipMode m) noexcept
    {
        return m == SamplerMipMode::Linear ? MTL::SamplerMipFilterLinear : MTL::SamplerMipFilterNearest;
    }

    MTL::SamplerAddressMode MetalDevice::_toAddrMode(SamplerAddressMode m) noexcept
    {
        switch (m)
        {
            case SamplerAddressMode::Repeat:
                return MTL::SamplerAddressModeRepeat;
            case SamplerAddressMode::MirroredRepeat:
                return MTL::SamplerAddressModeMirrorRepeat;
            case SamplerAddressMode::ClampToEdge:
                return MTL::SamplerAddressModeClampToEdge;
            case SamplerAddressMode::ClampToBorder:
                return MTL::SamplerAddressModeClampToBorderColor;
        }
        return MTL::SamplerAddressModeRepeat;
    }

    MTL::CompareFunction MetalDevice::_toCompare(CompareOp op) noexcept
    {
        switch (op)
        {
            case CompareOp::Never:
                return MTL::CompareFunctionNever;
            case CompareOp::Less:
                return MTL::CompareFunctionLess;
            case CompareOp::Equal:
                return MTL::CompareFunctionEqual;
            case CompareOp::LessEqual:
                return MTL::CompareFunctionLessEqual;
            case CompareOp::Greater:
                return MTL::CompareFunctionGreater;
            case CompareOp::NotEqual:
                return MTL::CompareFunctionNotEqual;
            case CompareOp::GreaterEqual:
                return MTL::CompareFunctionGreaterEqual;
            case CompareOp::Always:
                return MTL::CompareFunctionAlways;
        }
        return MTL::CompareFunctionAlways;
    }

    MTL::BlendFactor MetalDevice::_toBlendF(BlendFactor f) noexcept
    {
        switch (f)
        {
            case BlendFactor::Zero:
                return MTL::BlendFactorZero;
            case BlendFactor::One:
                return MTL::BlendFactorOne;
            case BlendFactor::SrcColor:
                return MTL::BlendFactorSourceColor;
            case BlendFactor::OneMinusSrcColor:
                return MTL::BlendFactorOneMinusSourceColor;
            case BlendFactor::DstColor:
                return MTL::BlendFactorDestinationColor;
            case BlendFactor::OneMinusDstColor:
                return MTL::BlendFactorOneMinusDestinationColor;
            case BlendFactor::SrcAlpha:
                return MTL::BlendFactorSourceAlpha;
            case BlendFactor::OneMinusSrcAlpha:
                return MTL::BlendFactorOneMinusSourceAlpha;
            case BlendFactor::DstAlpha:
                return MTL::BlendFactorDestinationAlpha;
            case BlendFactor::OneMinusDstAlpha:
                return MTL::BlendFactorOneMinusDestinationAlpha;
            case BlendFactor::SrcAlphaSaturate:
                return MTL::BlendFactorSourceAlphaSaturated;
            default:
                return MTL::BlendFactorOne;
        }
    }

    MTL::BlendOperation MetalDevice::_toBlendOp(BlendOp op) noexcept
    {
        switch (op)
        {
            case BlendOp::Add:
                return MTL::BlendOperationAdd;
            case BlendOp::Subtract:
                return MTL::BlendOperationSubtract;
            case BlendOp::ReverseSubtract:
                return MTL::BlendOperationReverseSubtract;
            case BlendOp::Min:
                return MTL::BlendOperationMin;
            case BlendOp::Max:
                return MTL::BlendOperationMax;
        }
        return MTL::BlendOperationAdd;
    }

    MTL::PrimitiveTopologyClass MetalDevice::_toTopology(PrimitiveTopology t) noexcept
    {
        switch (t)
        {
            case PrimitiveTopology::PointList:
                return MTL::PrimitiveTopologyClassPoint;
            case PrimitiveTopology::LineList:
            case PrimitiveTopology::LineStrip:
                return MTL::PrimitiveTopologyClassLine;
            case PrimitiveTopology::TriangleList:
            case PrimitiveTopology::TriangleStrip:
                return MTL::PrimitiveTopologyClassTriangle;
        }
        return MTL::PrimitiveTopologyClassTriangle;
    }

    MTL::Winding MetalDevice::_toWinding(FrontFace f) noexcept
    {
        return f == FrontFace::CounterClockwise ? MTL::WindingCounterClockwise : MTL::WindingClockwise;
    }

    MTL::CullMode MetalDevice::_toCull(CullMode m) noexcept
    {
        switch (m)
        {
            case CullMode::None:
                return MTL::CullModeNone;
            case CullMode::Front:
                return MTL::CullModeFront;
            case CullMode::Back:
                return MTL::CullModeBack;
        }
        return MTL::CullModeNone;
    }

    MTL::DepthStencilDescriptor *MetalDevice::_makeDepthStencilDesc(const DepthStencilState &ds, MTL::Device *)
    {
        auto *dsd = MTL::DepthStencilDescriptor::alloc()->init();
        dsd->setDepthWriteEnabled(ds.DepthWrite);
        dsd->setDepthCompareFunction(ds.DepthTest ? _toCompare(ds.DepthOp) : MTL::CompareFunctionAlways);
        return dsd;
    }

} // namespace rhi::metal
