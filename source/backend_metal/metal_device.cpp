/**
 * {file} metal_device.cpp
 * {brief} Implements Metal device and resource lifecycles. Command recording is implemented elsewhere.
 */
// Pure C++ / metal-cpp — no Objective-C or .mm required.
//
// Targets Metal 4 exclusively (MTL4CommandQueue/MTL4Compiler/MTL4ArgumentTable/
// MTLResidencySet) — see docs/API_GUIDELINES.md's "Target the latest platform API
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
#include "LightRHIMetalDisplayLink-Swift.h"

#include <Metal/MTL4AccelerationStructure.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dispatch/dispatch.h>
#include <expected>
#include <functional>
#include <mach/mach_time.h>
#include <map>
#include <memory>
#include <mutex>
#include <span>
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
    std::expected<std::unique_ptr<IDevice>, DeviceError> CreateDevice(const DeviceDesc &desc)
    {
        return metal::MetalDevice::Create(desc).transform([](std::unique_ptr<metal::MetalDevice> device)
                                                          { return std::unique_ptr<IDevice>{std::move(device)}; });
    }

    ExternalTextureOwnerContext ExternalTextureOwnerContextOf(IDevice & /*device*/)
    {
        // Nothing has to be handed over first: a CAMetalLayer belongs to the window already.
        return {};
    }

    std::expected<std::unique_ptr<IExternalTextureProvider>, DeviceError>
    CreateExternalTextureProvider(IDevice &device, ExternalTextureOwner owner, Format requestedFormat,
                                  PresentTiming timing)
    {
        // The one cast, written where the type is known: this build is the Metal one, so the
        // platform object a caller can have obtained is a CAMetalLayer and nothing else.
        auto *nativeLayer{static_cast<CA::MetalLayer *>(owner.Value)};
        if (nativeLayer == nullptr)
        {
            ReportRefusal("a Metal external texture provider needs a CAMetalLayer");
            return std::unexpected{DeviceError::InvalidArgument};
        }

        // A Metal provider only ever receives the Metal device, and the build has no RTTI to check it.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto        &metalDevice{static_cast<metal::MetalDevice &>(device)};
        auto         layer{NS::RetainPtr(nativeLayer)};
        const Format format{requestedFormat == Format::Undefined ? Format::BGRA8Unorm : requestedFormat};

        // The layer stays the application's: this sets only what it must in order to draw into it,
        // never its size.
        layer->setDevice(&metalDevice.MtlDevice());
        layer->setPixelFormat(metal::MetalDevice::ExternalTexturePixelFormat(format));
        // Without display sync the window server takes the newest finished frame at its next
        // composite, instead of showing every frame on its own refresh in order.
        layer->setDisplaySyncEnabled(timing == PresentTiming::OnRefresh);
        // A surface texture is a texture like any other, so it has to accept every use a texture
        // accepts. A drawable is a colour attachment and nothing else until this is cleared, which
        // would refuse a copy into it - the cost is that Core Animation keeps the pixels readable
        // rather than discarding them after a frame.
        layer->setFramebufferOnly(false);
        // Metal 4 makes drawable textures reachable through the layer's own residency set.
        if (auto *layerResidency{layer->residencySet()})
        {
            metalDevice.Mtl4Queue().addResidencySet(layerResidency);
        }

        // Showing the newest state at each refresh is what a display link is for.
        return std::make_unique<metal::MetalPresentingTextureProvider>(metalDevice, std::move(layer), format,
                                                                       timing == PresentTiming::AsSoonAsReady);
    }

    std::expected<SharedDevice, DeviceError> AcquireSharedDevice(const DeviceDesc &desc)
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
    MTL::Device &MetalDevice::MtlDevice() const noexcept
    {
        return *_device.get();
    }
    MTL4::CommandQueue &MetalDevice::Mtl4Queue() const noexcept
    {
        return *_queue.get();
    }

    bool MetalDevice::DebugCaptureEnabled() const noexcept
    {
        return _debugCaptureEnabled;
    }
    MetalBuffer &MetalDevice::Buffer(BufferHandle h)
    {
        assert(h.Valid());
        MetalBuffer &buffer{_buffers.Get(h.Index)};
        assert(buffer.Buffer);
        return buffer;
    }
    MetalTexture &MetalDevice::Texture(TextureHandle h)
    {
        assert(h.Valid());
        MetalTexture &texture{_textures.Get(h.Index)};
        assert(texture.Texture);
        return texture;
    }
    MetalSampler &MetalDevice::Sampler(SamplerHandle h)
    {
        assert(h.Valid());
        MetalSampler &sampler{_samplers.Get(h.Index)};
        assert(sampler.State);
        return sampler;
    }
    MetalTimestampQueryPool &MetalDevice::TimestampQueryPool(TimestampQueryPoolHandle h)
    {
        assert(h.Valid());
        MetalTimestampQueryPool &pool{_timestampQueryPools.Get(h.Index)};
        assert(pool.Heap);
        return pool;
    }
    MetalPipeline &MetalDevice::Pipeline(PipelineHandle h)
    {
        assert(h.Valid());
        MetalPipeline &pipeline{_pipelines.Get(h.Index)};
        assert(pipeline.RenderPso || pipeline.ComputePso);
        return pipeline;
    }
    MetalAccelerationStructure &MetalDevice::AccelStruct(AccelerationStructureHandle h)
    {
        assert(h.Valid());
        MetalAccelerationStructure &accelerationStructure{_accelStructs.Get(h.Index)};
        assert(accelerationStructure.As);
        return accelerationStructure;
    }

    // Defined here rather than in metal_internal.h: that header sits in the module purview, where a
    // member defined inside its class is not implicitly inline.
    IFramePresenter::IFramePresenter()                  = default;
    IFramePresenter::~IFramePresenter()                 = default;
    MetalDevice::ConstructionToken::ConstructionToken() = default;

    const MetalAccelerationStructure &MetalDevice::AccelStruct(AccelerationStructureHandle h) const
    {
        assert(h.Valid());
        const MetalAccelerationStructure &accelerationStructure{_accelStructs.Get(h.Index)};
        assert(accelerationStructure.As);
        return accelerationStructure;
    }
    FenceHandle MetalDevice::NextFence() noexcept
    {
        return FenceHandle{.Id = ++_timelineValue};
    }
    MTL::SharedEvent &MetalDevice::TimelineEvent() const noexcept
    {
        return *_timelineEvent.get();
    }
    bool MetalDevice::DeviceLost() const noexcept
    {
        return _deviceLost;
    }

    MTL::PixelFormat MetalDevice::ExternalTexturePixelFormat(Format f) noexcept
    {
        return _toPixFmt(f);
    }

    std::expected<TextureHandle, DeviceError> MetalDevice::AdoptExternalTexture(NS::SharedPtr<MTL::Texture> tex,
                                                                                const TextureDesc          &desc)
    {
        return _adoptTexture(std::move(tex), desc, MemoryHeapHandle{}, 0, /*borrowed=*/true);
    }

    std::expected<MetalCommandResources, DeviceError> MetalDevice::AcquireCommandResources()
    {
        {
            const uint64_t         completedFence{_timelineEvent->signaledValue()};
            const std::scoped_lock lock{_commandResourcePoolMutex};
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

        auto allocator{
            AdoptCreated(_device->newCommandAllocator(), "MTLDevice::newCommandAllocator refused the allocator")};
        if (!allocator)
        {
            return std::unexpected{allocator.error()};
        }
        auto commandBuffer{AdoptCreated(_device->newCommandBuffer(), "MTLDevice::newCommandBuffer refused the buffer")};
        if (!commandBuffer)
        {
            return std::unexpected{commandBuffer.error()};
        }
        return MetalCommandResources{
            .Allocator     = std::move(*allocator),
            .CommandBuffer = std::move(*commandBuffer),
        };
    }
    void MetalDevice::RecycleCommandResources(MetalCommandResources &&resources)
    {
        const std::scoped_lock lock{_commandResourcePoolMutex};
        _commandResourcePool.push_back(std::move(resources));
    }
    MTL::CommandQueue &MetalDevice::LegacyQueue() const noexcept
    {
        return *_legacyQueue.get();
    }

    // ============================================================================
    // MetalDevice — residency helpers
    // ============================================================================

    void MetalDevice::_addResident(MTL::Allocation &res)
    {
        _residencySet->addAllocation(&res);
        _residencySet->commit();
    }

    void MetalDevice::_removeResident(MTL::Allocation &res)
    {
        _residencySet->removeAllocation(&res);
        _residencySet->commit();
    }

    // ============================================================================
    // MetalDevice — constructor / destructor
    // ============================================================================

    std::expected<MetalDevice::RequiredObjects, DeviceError> MetalDevice::_createRequiredObjects(const DeviceDesc &desc)
    {
        // Every refusal here means this system cannot provide a device, whatever Metal's reason.
        constexpr DeviceError kNoDevice{DeviceError::NoDevice};

        auto device{
            AdoptCreated(MTL::CreateSystemDefaultDevice(), "no Metal device on this system", nullptr, kNoDevice)};
        if (!device)
        {
            return std::unexpected{device.error()};
        }

        NS::Error *queueError{nullptr};
        auto      *queue{
            [&]
            {
                if (!desc.EnableValidation)
                {
                    return (*device)->newMTL4CommandQueue();
                }
                auto queueDesc{AdoptRequired(MTL4::CommandQueueDescriptor::alloc()->init(),
                                             "MTL4::CommandQueueDescriptor could not be created")};
                queueDesc->setLabel(NS::String::string("HdRestir / LightRHI compute queue", NS::UTF8StringEncoding));
                return (*device)->newMTL4CommandQueue(queueDesc.get(), &queueError);
            }()};
        auto adoptedQueue{
            AdoptCreated(queue, "MTLDevice::newMTL4CommandQueue refused the queue", queueError, kNoDevice)};
        if (!adoptedQueue)
        {
            return std::unexpected{adoptedQueue.error()};
        }

        // Classic queue used only for the acceleration-structure build
        // exception — see LegacyQueue()'s doc comment in metal_internal.h.
        auto legacyQueue{AdoptCreated((*device)->newCommandQueue(),
                                      "MTLDevice::newCommandQueue refused the acceleration-structure queue", nullptr,
                                      kNoDevice)};
        if (!legacyQueue)
        {
            return std::unexpected{legacyQueue.error()};
        }
        if (desc.EnableValidation)
        {
            (*legacyQueue)
                ->setLabel(NS::String::string("HdRestir / acceleration structure build queue", NS::UTF8StringEncoding));
        }

        auto compilerDescCreated{
            AdoptCreated(MTL4::CompilerDescriptor::alloc()->init(), "MTL4::CompilerDescriptor could not be created")};
        if (!compilerDescCreated)
        {
            return std::unexpected{compilerDescCreated.error()};
        }
        auto       compilerDesc{std::move(*compilerDescCreated)};
        NS::Error *compilerError{nullptr};
        auto       compiler{AdoptCreated((*device)->newCompiler(compilerDesc.get(), &compilerError),
                                         "MTL4::Device::newCompiler refused the compiler", compilerError, kNoDevice)};
        if (!compiler)
        {
            return std::unexpected{compiler.error()};
        }

        auto residencyDescCreated{AdoptCreated(MTL::ResidencySetDescriptor::alloc()->init(),
                                               "MTL::ResidencySetDescriptor could not be created")};
        if (!residencyDescCreated)
        {
            return std::unexpected{residencyDescCreated.error()};
        }
        auto       residencyDesc{std::move(*residencyDescCreated)};
        NS::Error *residencyError{nullptr};
        auto       residencySet{AdoptCreated((*device)->newResidencySet(residencyDesc.get(), &residencyError),
                                             "MTLDevice::newResidencySet refused the set", residencyError, kNoDevice)};
        if (!residencySet)
        {
            return std::unexpected{residencySet.error()};
        }

        auto timelineEvent{AdoptCreated((*device)->newSharedEvent(),
                                        "MTLDevice::newSharedEvent refused the timeline event", nullptr, kNoDevice)};
        if (!timelineEvent)
        {
            return std::unexpected{timelineEvent.error()};
        }

        return RequiredObjects{
            .Device        = std::move(*device),
            .Queue         = std::move(*adoptedQueue),
            .LegacyQueue   = std::move(*legacyQueue),
            .Compiler      = std::move(*compiler),
            .ResidencySet  = std::move(*residencySet),
            .TimelineEvent = std::move(*timelineEvent),
        };
    }

    std::expected<std::unique_ptr<MetalDevice>, DeviceError> MetalDevice::Create(const DeviceDesc &desc)
    {
        return _createRequiredObjects(desc).transform(
            [&](RequiredObjects objects)
            { return std::make_unique<MetalDevice>(ConstructionToken{}, std::move(objects), desc); });
    }

    MetalDevice::MetalDevice(ConstructionToken /*unused*/, RequiredObjects objects, const DeviceDesc &desc)
        : _device{std::move(objects.Device)}, _queue{std::move(objects.Queue)},
          _legacyQueue{std::move(objects.LegacyQueue)}, _compiler{std::move(objects.Compiler)},
          _residencySet{std::move(objects.ResidencySet)}, _timelineEvent{std::move(objects.TimelineEvent)},
          _raytracingSupported(_device->supportsRaytracing()), _debugCaptureEnabled{desc.EnableValidation},
          _gpuValidationEnabled{desc.EnableGpuValidation}, _videoMemoryBytes(_device->recommendedMaxWorkingSetSize())
    {
        _queue->addResidencySet(_residencySet.get());

        _adapterName = std::string{_device->name()->utf8String()};

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
        if (!_debugCaptureEnabled)
        {
            // Capture scopes exist for Xcode's capture UI. With capture metadata off, every
            // submission would otherwise still take the mutex and hash its label for nothing.
            return nullptr;
        }
        const std::string      key{name.empty() ? std::string_view{"LightRHI submission"} : name};
        const std::scoped_lock lock{_captureScopeMutex};
        auto [it, inserted]{_submissionCaptureScopes.try_emplace(key)};
        if (inserted)
        {
            auto *manager{MTL::CaptureManager::sharedCaptureManager()};
            it->second = NS::TransferPtr(manager->newCaptureScope(_queue.get()));
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
        NS::SharedPtr<MTL::CaptureScope> scope{};
        {
            const std::scoped_lock lock{_captureScopeMutex};
            auto                  *manager{MTL::CaptureManager::sharedCaptureManager()};
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
        if (_activeCaptureScope != nullptr)
        {
            MTL::CaptureManager::sharedCaptureManager()->setDefaultCaptureScope(_activeCaptureScope);
        }
        if (_activeCaptureScope != nullptr)
        {
            _activeCaptureScope->beginScope();
        }
    }

    void MetalDevice::EndCaptureScope()
    {
        if (_activeCaptureScope != nullptr)
        {
            _activeCaptureScope->endScope();
            _activeCaptureScope = nullptr;
        }
        const std::scoped_lock lock{_captureScopeMutex};
        _frameCaptureScope.reset();
    }

    void MetalDevice::SuspendActiveCaptureScope() noexcept
    {
        if (_activeCaptureScope != nullptr)
        {
            _activeCaptureScope->endScope();
        }
    }

    void MetalDevice::ResumeActiveCaptureScope() noexcept
    {
        if (_activeCaptureScope != nullptr)
        {
            _activeCaptureScope->beginScope();
        }
    }

    // ============================================================================
    // MetalPresentingTextureProvider
    //
    // The layer is the application's and it sizes it; this only draws into it. A drawable and the
    // texture naming it live between NextTexture() and the present that shows them, so a null
    // drawable means no frame is being held.
    // ============================================================================

    MetalPresentingTextureProvider::MetalPresentingTextureProvider(MetalDevice                  &device,
                                                                   NS::SharedPtr<CA::MetalLayer> layer, Format format,
                                                                   const bool paced) noexcept
        : _device{device}, _layer{std::move(layer)}, _format{format},
          _handle{device.RegisterExternalTextureProvider(*this)}
    {
        if (paced)
        {
            _pacer = LightRHIMetalDisplayLink::createMetalDisplayPacer(_layer.get());
        }
    }

    MetalPresentingTextureProvider::~MetalPresentingTextureProvider()
    {
        // Unregistered first: a present recorded against this handle must not resolve to a target
        // that is already giving its frame back.
        _device.UnregisterExternalTextureProvider(_handle);
        if (_pacer != 0U)
        {
            LightRHIMetalDisplayLink::destroyMetalDisplayPacer(_pacer);
        }
        // A frame taken and never shown still owns a texture slot and a drawable.
        if (_texture.Valid())
        {
            _device.DestroyTexture(_texture);
        }
    }

    Format MetalPresentingTextureProvider::TextureFormat() const noexcept
    {
        return _format;
    }

    ExternalTextureProviderHandle MetalPresentingTextureProvider::Handle() const noexcept
    {
        return _handle;
    }

#if METRICS_ENABLED
    namespace
    {
        /**
         * {brief} Keep one presented frame's fate, reported on whatever thread Core Animation uses.
         *
         * `presentedTime` counts seconds on the host clock Core Animation uses, which is not the
         * steady clock callers measure with; both are read now, so the time is moved across exactly.
         * Zero means the frame was replaced before the screen took it.
         */
        void RecordShown(MetalPresentingTextureProvider::ShownFrames &shown, const uint64_t present,
                         const double presentedSeconds) noexcept
        {
            uint64_t shownNanoseconds{};
            if (presentedSeconds > 0.0)
            {
                mach_timebase_info_data_t timebase{};
                mach_timebase_info(&timebase);
                const double hostNow{static_cast<double>(mach_absolute_time()) * timebase.numer / timebase.denom *
                                     1e-9};
                const auto   steadyNow{std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count()};
                const double ago{std::max(0.0, hostNow - presentedSeconds)};
                shownNanoseconds = static_cast<uint64_t>(steadyNow) - static_cast<uint64_t>(ago * 1e9);
            }
            const std::scoped_lock lock{shown.Lock};
            if (shown.Count == shown.Fates.size())
            {
                shown.First = (shown.First + 1U) % shown.Fates.size(); // the oldest goes first
                --shown.Count;
            }
            shown.Fates.at((shown.First + shown.Count) % shown.Fates.size()) =
                ShownFrame{.Present = present, .ShownNanoseconds = shownNanoseconds};
            ++shown.Count;
        }
    } // namespace

    std::size_t MetalPresentingTextureProvider::TakeShownFrames(const std::span<ShownFrame> frames) noexcept
    {
        const std::scoped_lock lock{_shown->Lock};
        const std::size_t      taken{std::min(frames.size(), _shown->Count)};
        for (std::size_t index{}; index < taken; ++index)
        {
            frames[index] = _shown->Fates.at((_shown->First + index) % _shown->Fates.size());
        }
        _shown->First = (_shown->First + taken) % _shown->Fates.size();
        _shown->Count -= taken;
        return taken;
    }
#endif // METRICS_ENABLED

    ExternalTextureProviderHandle MetalDevice::RegisterExternalTextureProvider(IFramePresenter &target)
    {
        const uint32_t index{_externalTextureProviders.Alloc()};
        if (index == kInvalidIndex)
        {
            ReportRefusal("this device already holds as many external texture providers as it can");
            return {};
        }
        _externalTextureProviders.Get(index) = &target;
        return ExternalTextureProviderHandle{.Index = index};
    }

    void MetalDevice::UnregisterExternalTextureProvider(ExternalTextureProviderHandle handle) noexcept
    {
        if (handle.Valid())
        {
            _externalTextureProviders.Free(handle.Index);
        }
    }

    IFramePresenter *MetalDevice::ResolveExternalTextureProvider(ExternalTextureProviderHandle handle) const noexcept
    {
        return handle.Valid() ? _externalTextureProviders.Get(handle.Index) : nullptr;
    }

    Extent2D MetalPresentingTextureProvider::_layerExtent() const noexcept
    {
        const CGSize size{_layer->drawableSize()};
        return Extent2D{.Width = static_cast<uint32_t>(size.width), .Height = static_cast<uint32_t>(size.height)};
    }

    std::expected<TextureHandle, ExternalTextureError> MetalPresentingTextureProvider::NextTexture()
    {
        if (_drawable)
        {
            // Taking a second frame before the first is shown strands the first one's drawable and
            // its texture slot, and there is no use for it: a frame is taken, drawn and shown.
            FailContract("taking a frame from a surface that is already holding one");
        }
        if (_device.DeviceLost())
        {
            return std::unexpected{ExternalTextureError::Lost};
        }

        const Extent2D extent{_layerExtent()};
        if (extent.Width == 0 || extent.Height == 0)
        {
            return std::unexpected{ExternalTextureError::Unavailable}; // nothing to draw into without area
        }

        // With a display link the system hands over the drawable for the coming refresh, at the
        // moment a frame drawn into it now still reaches it, and never more than one per refresh.
        if (_pacer != 0U)
        {
            const std::uintptr_t handedOver{LightRHIMetalDisplayLink::takeMetalDisplayPacerDrawable(_pacer)};
            if (handedOver == 0U)
            {
                return std::unexpected{ExternalTextureError::Unavailable}; // not yet: the refresh is not near
            }
            // The pointer crossed from Swift as a number, so only a cast turns it back.
            // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
            _drawable = NS::TransferPtr(reinterpret_cast<CA::MetalDrawable *>(handedOver)); // retained for us
            return _adoptHeldDrawable(extent);
        }

        // `nextDrawable` blocks until the display frees an image, which can be a whole refresh. One
        // image is on screen and the rest may wait to be shown, so a frame is taken only while at
        // least one is still free; then it is there without waiting. The caller retries later.
        // Waiting for every frame to be shown first would halve the rate: the window server
        // shows a frame about two refreshes after it is presented.
        if (_framesWaiting->load(std::memory_order_acquire) + 1U >= _layer->maximumDrawableCount())
        {
            return std::unexpected{ExternalTextureError::Unavailable};
        }
        auto *drawable{_layer->nextDrawable()};
        if (drawable == nullptr)
        {
            // Every image is still being shown, or the window is occluded. The frame is skipped.
            return std::unexpected{ExternalTextureError::Unavailable};
        }
        _drawable = NS::RetainPtr(drawable);
        return _adoptHeldDrawable(extent);
    }

    std::expected<TextureHandle, ExternalTextureError>
    MetalPresentingTextureProvider::_adoptHeldDrawable(const Extent2D extent)
    {
        auto *drawable{_drawable.get()};
        _device.Mtl4Queue().wait(drawable);

        const TextureDesc desc{
            .Format = _format,
            .Extent = {.Width = extent.Width, .Height = extent.Height, .Depth = 1},
            .Usage  = TextureUsage::RenderTarget,
        };
        auto adopted{_device.AdoptExternalTexture(NS::RetainPtr(drawable->texture()), desc)};
        if (!adopted)
        {
            // The slot pool is full, so the frame cannot be named; the drawable is released rather
            // than presented half-used.
            _drawable.reset();
            return std::unexpected{ExternalTextureError::Unavailable};
        }
        _texture = *adopted;
        return _texture;
    }

    void MetalPresentingTextureProvider::PresentHeldFrame()
    {
        if (!_drawable)
        {
            FailContract("presenting a surface that is holding no frame");
        }
        // Metal 4 orders presentation on the queue: the drawable was waited for when it was taken
        // and is signalled after the work that drew it, so present() shows a finished frame.
        _device.Mtl4Queue().signalDrawable(_drawable.get());
        // Counted until it reaches the screen, or is dropped, which also runs the handler.
        _framesWaiting->fetch_add(1U, std::memory_order_acq_rel);
#if METRICS_ENABLED
        _drawable->addPresentedHandler(
            [waiting = _framesWaiting, shown = _shown, present = ++_presents](MTL::Drawable *drawable)
            {
                RecordShown(*shown, present, drawable->presentedTime());
                waiting->fetch_sub(1U, std::memory_order_acq_rel);
            });
#else
        _drawable->addPresentedHandler([waiting = _framesWaiting](MTL::Drawable * /*drawable*/)
                                       { waiting->fetch_sub(1U, std::memory_order_acq_rel); });
#endif
        _drawable->present();

        // The texture named the drawable for this frame only. Releasing the slot here is what makes
        // the handle's documented lifetime true.
        _device.DestroyTexture(_texture);
        _texture = TextureHandle{};
        _drawable.reset();
    }

    // ============================================================================
    // Memory
    //
    // A memory heap is one MTLHeap of type placement. Metal leaves a placement heap untracked for
    // hazards, which matches LightRHI: callers already record explicit transitions and barriers.
    //
    // The heap joins the queue residency set once, here, and its placed textures do not join it
    // individually — that is the point of the heap. Membership at heap granularity is what keeps
    // allocating many small textures from committing the residency set once per texture.
    // ============================================================================

    std::expected<MemoryHeapHandle, DeviceError> MetalDevice::CreateMemoryHeap(const MemoryHeapDesc &desc)
    {
        auto hdCreated{AdoptCreated(MTL::HeapDescriptor::alloc()->init(), "MTL::HeapDescriptor could not be created")};
        if (!hdCreated)
        {
            return std::unexpected{hdCreated.error()};
        }
        auto hd{std::move(*hdCreated)};
        hd->setType(MTL::HeapTypePlacement);
        hd->setStorageMode(_toStorageMode(desc.MemoryType));
        hd->setSize(desc.Size);

        auto heap{AdoptCreated(_device->newHeap(hd.get()), "MTLDevice::newHeap refused the memory heap")};
        if (!heap)
        {
            return std::unexpected{heap.error()};
        }

        if (!desc.DebugName.empty())
        {
            (*heap)->setLabel(MakeLabel(desc.DebugName));
        }

        const uint32_t idx{_memoryHeaps.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many memory heaps as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        // Resident only once it has a slot, so a refused heap never stays in the residency set.
        _addResident(*heap->get());
        _memoryHeaps.Get(idx) =
            MetalMemoryHeap{.Heap = std::move(*heap), .Size = desc.Size, .Memory = desc.MemoryType, .LiveRanges = {}};
        return MemoryHeapHandle{.Index = idx};
    }

    void MetalDevice::DestroyMemoryHeap(MemoryHeapHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _removeResident(*_memoryHeaps.Get(h.Index).Heap.get());
        _memoryHeaps.Free(h.Index);
    }

    std::expected<PlacementRequirements, PlacementError>
    MetalDevice::TexturePlacementRequirements(const TextureDesc &desc, MemoryType memory) const
    {
        // Apple silicon gives every storage mode the same unified memory, so the fastest texture
        // layout is available in a CPU-visible heap too and there is no second-best path to take.
        auto                    mtd{_makeTextureDescriptor(desc, _toStorageMode(memory))};
        const MTL::SizeAndAlign sizeAlign{_device->heapTextureSizeAndAlign(mtd.get())};
        if (sizeAlign.size == 0 || sizeAlign.align == 0)
        {
            return std::unexpected{PlacementError::CreationFailed};
        }
        return PlacementRequirements{.Size = sizeAlign.size, .Alignment = sizeAlign.align};
    }

    // ============================================================================
    // Buffer
    // ============================================================================

    std::expected<BufferHandle, DeviceError> MetalDevice::_adoptBuffer(NS::SharedPtr<MTL::Buffer> buf,
                                                                       const BufferDesc &desc, MemoryHeapHandle heap,
                                                                       uint64_t offset)
    {
        if (!desc.DebugName.empty())
        {
            buf->setLabel(MakeLabel(desc.DebugName));
        }

        const uint64_t address{buf->gpuAddress()};
        const uint32_t idx{_buffers.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many buffers as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        // A placed buffer is reachable through its heap's residency, already committed once.
        if (!heap.Valid())
        {
            _addResident(*buf.get());
        }
        _heap.RegisterBuffer();
        _buffers.Get(idx) = MetalBuffer{
            .Buffer = std::move(buf), .Size = desc.Size, .Usage = desc.Usage, .Heap = heap, .Offset = offset};

        // Track base GPU address -> slot so acceleration-structure geometry
        // (addressed by GpuAddress/BDA) can be resolved back to an MTL::Buffer*
        // + offset — see _bufferAndOffsetFromAddress.
        {
            const std::scoped_lock lk{_bufferAddrMutex};
            _bufferAddrToIndex[address] = idx;
        }

        return BufferHandle{.Index = idx};
    }

    std::expected<BufferHandle, DeviceError> MetalDevice::CreateBuffer(const BufferDesc &desc)
    {
        return AdoptCreated(_device->newBuffer(desc.Size, _toOptions(desc.MemoryType)),
                            "MTLDevice::newBuffer refused the allocation")
            .and_then([&](NS::SharedPtr<MTL::Buffer> buf)
                      { return _adoptBuffer(std::move(buf), desc, MemoryHeapHandle{}, 0); });
    }

    std::expected<PlacementRequirements, PlacementError>
    MetalDevice::BufferPlacementRequirements(const BufferDesc &desc) const
    {
        const MTL::SizeAndAlign sizeAlign{_device->heapBufferSizeAndAlign(desc.Size, _toOptions(desc.MemoryType))};
        if (sizeAlign.size == 0 || sizeAlign.align == 0)
        {
            return std::unexpected{PlacementError::CreationFailed};
        }
        return PlacementRequirements{.Size = sizeAlign.size, .Alignment = sizeAlign.align};
    }

    std::expected<BufferHandle, PlacementError> MetalDevice::CreateBuffer(const BufferDesc    &desc,
                                                                          const HeapPlacement &placement)
    {
        const MTL::SizeAndAlign sizeAlign{_device->heapBufferSizeAndAlign(desc.Size, _toOptions(desc.MemoryType))};

        auto reserved{_reservePlacement(placement, desc.MemoryType, sizeAlign.size, sizeAlign.align)};
        if (!reserved)
        {
            return std::unexpected(reserved.error());
        }

        auto buf{
            NS::TransferPtr(reserved->get().Heap->newBuffer(desc.Size, _toOptions(desc.MemoryType), placement.Offset))};
        if (!buf)
        {
            reserved->get().LiveRanges.erase(placement.Offset);
            return std::unexpected(PlacementError::CreationFailed);
        }
        auto adopted{_adoptBuffer(std::move(buf), desc, placement.Heap, placement.Offset)};
        if (!adopted)
        {
            reserved->get().LiveRanges.erase(placement.Offset);
            return std::unexpected{PlacementError::CreationFailed};
        }
        return *adopted;
    }

    void MetalDevice::DestroyBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &record{_buffers.Get(h.Index)};
        {
            const std::scoped_lock lk{_bufferAddrMutex};
            _bufferAddrToIndex.erase(record.Buffer->gpuAddress());
        }
        if (record.Heap.Valid())
        {
            _memoryHeaps.Get(record.Heap.Index).LiveRanges.erase(record.Offset);
        }
        else
        {
            _removeResident(*record.Buffer.get());
        }
        _heap.UnregisterBuffer();
        _buffers.Free(h.Index);
    }

    GpuAddress MetalDevice::BufferAddress(BufferHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{.Address = _buffers.Get(h.Index).Buffer->gpuAddress()};
    }

    BufferInfo MetalDevice::GetBufferInfo(BufferHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        const auto &b{_buffers.Get(h.Index)};
        return BufferInfo{
            .Size = b.Size, .Usage = b.Usage, .DeviceAddress = GpuAddress{.Address = b.Buffer->gpuAddress()}};
    }

    TextureInfo MetalDevice::GetTextureInfo(TextureHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        const TextureDesc &desc{_textures.Get(h.Index).Desc};
        return TextureInfo{
            .Extent      = desc.Extent,
            .Format      = desc.Format,
            .Usage       = desc.Usage,
            .Dimension   = desc.Dimension,
            .MipLevels   = desc.MipLevels,
            .ArrayLayers = desc.ArrayLayers,
            .SampleCount = desc.SampleCount,
        };
    }

    MappedBuffer MetalDevice::MapBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return {};
        }
        auto &b{_buffers.Get(h.Index)};
        return MappedBuffer{.Data = b.Buffer->contents(), .Size = b.Size};
    }

    void MetalDevice::UnmapBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &b{_buffers.Get(h.Index)};
        if (b.Buffer->storageMode() == MTL::StorageModeManaged)
        {
            b.Buffer->didModifyRange(NS::Range::Make(0, b.Size));
        }
    }

    // ============================================================================
    // Texture
    // ============================================================================

    NS::SharedPtr<MTL::TextureDescriptor> MetalDevice::_makeTextureDescriptor(const TextureDesc &desc,
                                                                              MTL::StorageMode   storage)
    {
        auto mtd{AdoptRequired(MTL::TextureDescriptor::alloc()->init(), "MTL::TextureDescriptor could not be created")};
        mtd->setTextureType(_toTexType(desc.Dimension));
        mtd->setPixelFormat(_toPixFmt(desc.Format));
        mtd->setWidth(desc.Extent.Width);
        mtd->setHeight(desc.Extent.Height);
        mtd->setDepth(desc.Extent.Depth);
        mtd->setMipmapLevelCount(desc.MipLevels);
        mtd->setArrayLength(desc.ArrayLayers);
        mtd->setSampleCount(desc.SampleCount);
        mtd->setUsage(_toTexUsage(desc.Usage));
        mtd->setStorageMode(storage);
        return mtd;
    }

    std::expected<TextureHandle, DeviceError> MetalDevice::_adoptTexture(NS::SharedPtr<MTL::Texture> tex,
                                                                         const TextureDesc &desc, MemoryHeapHandle heap,
                                                                         uint64_t offset, bool borrowed)
    {
        if (!desc.DebugName.empty())
        {
            tex->setLabel(MakeLabel(desc.DebugName));
        }

        const uint32_t idx{_textures.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many textures as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        // A placed texture is reachable through its heap's residency, already committed once, and a
        // borrowed drawable texture through the layer's own residency set.
        if (!heap.Valid() && !borrowed)
        {
            _addResident(*tex.get());
        }
        _heap.RegisterTexture();
        _textures.Get(idx) =
            MetalTexture{.Texture = std::move(tex), .Desc = desc, .Heap = heap, .Offset = offset, .Borrowed = borrowed};
        return TextureHandle{.Index = idx};
    }

    std::expected<TextureHandle, DeviceError> MetalDevice::CreateTexture(const TextureDesc &desc)
    {
        return AdoptCreated(_device->newTexture(_makeTextureDescriptor(desc, MTL::StorageModePrivate).get()),
                            "MTLDevice::newTexture refused the allocation")
            .and_then([&](NS::SharedPtr<MTL::Texture> tex)
                      { return _adoptTexture(std::move(tex), desc, MemoryHeapHandle{}, 0); });
    }

    std::expected<std::reference_wrapper<MetalMemoryHeap>, PlacementError>
    MetalDevice::_reservePlacement(const HeapPlacement &placement, MemoryType memory, uint64_t size, uint64_t alignment)
    {
        if (!placement.Heap.Valid())
        {
            return std::unexpected(PlacementError::InvalidHeap);
        }
        auto &memoryHeap{_memoryHeaps.Get(placement.Heap.Index)};
        if (!memoryHeap.Heap)
        {
            return std::unexpected(PlacementError::InvalidHeap);
        }
        if (memoryHeap.Memory != memory)
        {
            return std::unexpected(PlacementError::IncompatibleMemory);
        }
        if (alignment == 0 || placement.Offset % alignment != 0)
        {
            return std::unexpected(PlacementError::MisalignedOffset);
        }
        const uint64_t end{placement.Offset + size};
        if (end > memoryHeap.Size || end < placement.Offset)
        {
            return std::unexpected(PlacementError::OutOfRange);
        }

        // Refuse a range that a live placement already covers. `upper_bound` names the first
        // placement starting after this one, so only its predecessor and itself can overlap.
        const auto next{memoryHeap.LiveRanges.upper_bound(placement.Offset)};
        if (next != memoryHeap.LiveRanges.end() && next->first < end)
        {
            return std::unexpected(PlacementError::Overlapping);
        }
        if (next != memoryHeap.LiveRanges.begin() && std::prev(next)->second > placement.Offset)
        {
            return std::unexpected(PlacementError::Overlapping);
        }

        memoryHeap.LiveRanges.emplace(placement.Offset, end);
        return std::ref(memoryHeap);
    }

    std::expected<TextureHandle, PlacementError> MetalDevice::CreateTexture(const TextureDesc   &desc,
                                                                            const HeapPlacement &placement)
    {
        // Storage follows the heap: a CPU-visible heap holds CPU-visible textures, which Apple
        // silicon supports directly, so there is nothing to emulate. The requirement therefore
        // depends on the heap the texture is going into.
        const bool       knownHeap{placement.Heap.Valid() && _memoryHeaps.Get(placement.Heap.Index).Heap};
        const MemoryType memory{knownHeap ? _memoryHeaps.Get(placement.Heap.Index).Memory : MemoryType::GpuOnly};

        auto                    mtd{_makeTextureDescriptor(desc, _toStorageMode(memory))};
        const MTL::SizeAndAlign sizeAlign{_device->heapTextureSizeAndAlign(mtd.get())};

        auto reserved{_reservePlacement(placement, memory, sizeAlign.size, sizeAlign.align)};
        if (!reserved)
        {
            return std::unexpected(reserved.error());
        }

        auto tex{NS::TransferPtr(reserved->get().Heap->newTexture(mtd.get(), placement.Offset))};
        if (!tex)
        {
            reserved->get().LiveRanges.erase(placement.Offset);
            return std::unexpected(PlacementError::CreationFailed);
        }
        auto adopted{_adoptTexture(std::move(tex), desc, placement.Heap, placement.Offset)};
        if (!adopted)
        {
            reserved->get().LiveRanges.erase(placement.Offset);
            return std::unexpected{PlacementError::CreationFailed};
        }
        return *adopted;
    }

    void MetalDevice::DestroyTexture(TextureHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &tex{_textures.Get(h.Index)};
        if (tex.Heap.Valid())
        {
            _memoryHeaps.Get(tex.Heap.Index).LiveRanges.erase(tex.Offset);
        }
        else if (!tex.Borrowed)
        {
            _removeResident(*tex.Texture.get());
        }
        _heap.UnregisterTexture();
        _textures.Free(h.Index);
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
        return GpuAddress{.Address = _textures.Get(h.Index).Texture->gpuResourceID()._impl};
    }

    // ============================================================================
    // Sampler
    // ============================================================================

    std::expected<SamplerHandle, DeviceError> MetalDevice::CreateSampler(const SamplerDesc &desc)
    {
        auto sdCreated{
            AdoptCreated(MTL::SamplerDescriptor::alloc()->init(), "MTL::SamplerDescriptor could not be created")};
        if (!sdCreated)
        {
            return std::unexpected{sdCreated.error()};
        }
        auto sd{std::move(*sdCreated)};
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
            sd->setLabel(MakeLabel(desc.DebugName));
        }

        auto smp{AdoptCreated(_device->newSamplerState(sd.get()), "MTLDevice::newSamplerState refused the sampler")};
        if (!smp)
        {
            return std::unexpected{smp.error()};
        }

        const uint32_t idx{_samplers.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many samplers as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        _samplers.Get(idx) = MetalSampler{.State = std::move(*smp)};
        _heap.RegisterSampler();
        // MTL::SamplerState is not a MTL::Resource/Allocation — no residency
        // tracking needed (unchanged from classic Metal).
        return SamplerHandle{.Index = idx};
    }

    void MetalDevice::DestroySampler(SamplerHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _heap.UnregisterSampler();
        _samplers.Free(h.Index);
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
        return GpuAddress{.Address = h.Index};
    }

    // ============================================================================
    // Timestamp queries
    // ============================================================================

    bool MetalDevice::SupportsComputeTimestamps() const noexcept
    {
        return _device->queryTimestampFrequency() > 0;
    }

    double MetalDevice::TimestampPeriodNanoseconds() const noexcept
    {
        const uint64_t frequency{_device->queryTimestampFrequency()};
        return frequency > 0 ? 1.0e9 / static_cast<double>(frequency) : 0.0;
    }

    std::expected<TimestampQueryPoolHandle, DeviceError> MetalDevice::CreateTimestampQueryPool(uint32_t count)
    {
        if (count == 0)
        {
            return {};
        }

        auto descriptorCreated{AdoptCreated(MTL4::CounterHeapDescriptor::alloc()->init(),
                                            "MTL4::CounterHeapDescriptor could not be created")};
        if (!descriptorCreated)
        {
            return std::unexpected{descriptorCreated.error()};
        }
        auto descriptor{std::move(*descriptorCreated)};
        descriptor->setType(MTL4::CounterHeapTypeTimestamp);
        descriptor->setCount(count);
        NS::Error *error{nullptr};
        auto       heap{NS::TransferPtr(_device->newCounterHeap(descriptor.get(), &error))};
        if (!heap)
        {
            FailContract("MTLDevice::newCounterHeap failed: " + ErrorText(error));
        }

        const uint32_t index{_timestampQueryPools.Alloc()};
        if (index == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many timestamp query pools as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        _timestampQueryPools.Get(index) = MetalTimestampQueryPool{.Heap = std::move(heap), .Count = count};
        return TimestampQueryPoolHandle{.Index = index};
    }

    void MetalDevice::DestroyTimestampQueryPool(TimestampQueryPoolHandle pool)
    {
        if (pool.Valid())
        {
            _timestampQueryPools.Free(pool.Index);
        }
    }

    void MetalDevice::ResetTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, uint32_t count)
    {
        if (!pool.Valid() || count == 0)
        {
            return;
        }
        auto &record{_timestampQueryPools.Get(pool.Index)};
        assert(first <= record.Count && count <= record.Count - first);
        record.Heap->invalidateCounterRange(NS::Range{first, count});
    }

    void MetalDevice::ReadTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, std::span<uint64_t> results)
    {
        if (!pool.Valid() || results.empty())
        {
            return;
        }
        auto &record{_timestampQueryPools.Get(pool.Index)};
        assert(first <= record.Count && results.size() <= record.Count - first);
        NS::Data const *data{record.Heap->resolveCounterRange(NS::Range{first, results.size()})};
        if ((data == nullptr) || data->length() < results.size_bytes())
        {
            FailContract("MTL4 timestamp counter resolve failed");
        }
        std::memcpy(results.data(), data->bytes(), results.size_bytes());
    }

    // ============================================================================
    // Ray tracing
    // ============================================================================

    MTL::Buffer &MetalDevice::_bufferAtAddress(GpuAddress addr, std::string_view role) const
    {
        // Exact-key lookup: AccelerationStructureDesc always addresses
        // vertex/index data by a buffer's own base address (device.cppm's
        // BlasFromTriangleBuffer() takes IDevice::BufferAddress()'s result
        // directly, never an offset sub-range), so there is no "closest base
        // below addr" case to resolve — O(1) instead of scanning every buffer.
        // The geometry therefore always starts at offset zero in that buffer.
        const std::scoped_lock lk{_bufferAddrMutex};
        const auto             it{addr.Valid() ? _bufferAddrToIndex.find(addr.Address) : _bufferAddrToIndex.end()};
        if (it == _bufferAddrToIndex.end())
        {
            FailContract(std::string{role} + " does not resolve to a live LightRHI buffer");
        }
        return *_buffers.Get(it->second).Buffer.get();
    }

    NS::SharedPtr<MTL::AccelerationStructureDescriptor>
    MetalDevice::MakeAccelerationStructureDescriptor(const AccelerationStructureDesc        &desc,
                                                     std::vector<NS::SharedPtr<NS::Object>> &keepAlive) const
    {
        if (desc.Type == AccelerationStructureType::BottomLevel)
        {
            auto tri{AdoptRequired(MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init(),
                                   "MTL::AccelerationStructureTriangleGeometryDescriptor could not be created")};
            tri->setVertexBuffer(
                &_bufferAtAddress(desc.VertexBufferAddress, "BlasFromTriangleBuffer: VertexBufferAddress"));
            tri->setVertexBufferOffset(0);
            tri->setVertexStride(desc.VertexStride);
            tri->setVertexFormat(MTL::AttributeFormatFloat3);
            tri->setTriangleCount(desc.IndexBufferAddress.Valid() ? desc.IndexCount / 3 : desc.VertexCount / 3);
            tri->setOpaque(true);

            if (desc.IndexBufferAddress.Valid())
            {
                tri->setIndexBuffer(
                    &_bufferAtAddress(desc.IndexBufferAddress, "BlasFromTriangleBuffer: IndexBufferAddress"));
                tri->setIndexBufferOffset(0);
                tri->setIndexType(desc.IndexType == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32);
            }

            const std::array<const NS::Object *, 1> geometries{tri.get()};
            auto geoms{NS::RetainPtr(NS::Array::array(geometries.data(), geometries.size()))};

            auto pd{AdoptRequired(MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init(),
                                  "MTL::PrimitiveAccelerationStructureDescriptor could not be created")};
            pd->setGeometryDescriptors(geoms.get());
            pd->setUsage(desc.PreferFastTrace ? MTL::AccelerationStructureUsageNone
                                              : MTL::AccelerationStructureUsagePreferFastBuild);

            keepAlive.emplace_back(tri);
            keepAlive.emplace_back(geoms);
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
        auto           instanceBuf{NS::TransferPtr(
            _device->newBuffer(std::max<uint64_t>(instanceBufBytes, 1), MTL::ResourceStorageModeShared))};
        if (!instanceBuf)
        {
            // Neither a size query nor a build has a way to report this, and both need the instances.
            FailContract("MTLDevice::newBuffer refused the TLAS instance descriptors");
        }
        const std::span<MTL::AccelerationStructureUserIDInstanceDescriptor> dst{
            static_cast<MTL::AccelerationStructureUserIDInstanceDescriptor *>(instanceBuf->contents()),
            desc.Instances.size()};
        std::vector<NS::Object *> blasObjs(desc.Instances.size());
        for (size_t i{0}; i < desc.Instances.size(); ++i)
        {
            const auto &inst{desc.Instances[i]};
            if (!inst.Blas.Valid())
            {
                FailContract(std::string{"TlasFromInstances: AccelerationStructureInstance.Blas is not "} +
                             "a valid handle");
            }

            const MTL::PackedFloat4x3 xform{
                MTL::PackedFloat3{inst.Transform[0][0], inst.Transform[1][0], inst.Transform[2][0]},
                MTL::PackedFloat3{inst.Transform[0][1], inst.Transform[1][1], inst.Transform[2][1]},
                MTL::PackedFloat3{inst.Transform[0][2], inst.Transform[1][2], inst.Transform[2][2]},
                MTL::PackedFloat3{inst.Transform[0][3], inst.Transform[1][3], inst.Transform[2][3]},
            };
            blasObjs[i] = AccelStruct(inst.Blas).As.get();
            dst[i]      = MTL::AccelerationStructureUserIDInstanceDescriptor{
                .transformationMatrix            = xform,
                .options                         = static_cast<MTL::AccelerationStructureInstanceOptions>(inst.Flags),
                .mask                            = inst.InstanceMask,
                .intersectionFunctionTableOffset = 0,
                .accelerationStructureIndex      = static_cast<uint32_t>(i),
                .userID                          = inst.InstanceCustomIndex,
            };
        }

        auto blasArr{NS::RetainPtr(NS::Array::array(blasObjs.data(), blasObjs.size()))};

        auto id{AdoptRequired(MTL::InstanceAccelerationStructureDescriptor::alloc()->init(),
                              "MTL::InstanceAccelerationStructureDescriptor could not be created")};
        id->setInstanceCount(desc.Instances.size());
        id->setInstanceDescriptorBuffer(instanceBuf.get());
        id->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
        id->setInstancedAccelerationStructures(blasArr.get());
        id->setUsage(desc.PreferFastTrace ? MTL::AccelerationStructureUsageNone
                                          : MTL::AccelerationStructureUsagePreferFastBuild);

        keepAlive.emplace_back(instanceBuf);
        keepAlive.emplace_back(blasArr);
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
            FailContract(std::string{"QueryAccelerationStructureBuildSizes: device does not support ray "} +
                         "tracing (MTLDevice::supportsRaytracing() == false)");
        }

        std::vector<NS::SharedPtr<NS::Object>> keepAlive{};
        auto                                   descriptor{MakeAccelerationStructureDescriptor(desc, keepAlive)};
        const MTL::AccelerationStructureSizes  sizes{_device->accelerationStructureSizes(descriptor.get())};
        return AccelerationStructureBuildSizes{
            .AccelerationStructureSize = sizes.accelerationStructureSize,
            .BuildScratchSize          = sizes.buildScratchBufferSize,
            .UpdateScratchSize         = sizes.refitScratchBufferSize,
        };
    }

    std::expected<AccelerationStructureHandle, DeviceError>
    MetalDevice::CreateAccelerationStructure(const AccelerationStructureDesc &desc)
    {
        if (!_raytracingSupported)
        {
            FailContract(std::string{"CreateAccelerationStructure: device does not support ray tracing "} +
                         "(MTLDevice::supportsRaytracing() == false)");
        }

        auto sizes{QueryAccelerationStructureBuildSizes(desc)};
        auto as{AdoptCreated(_device->newAccelerationStructure(sizes.AccelerationStructureSize),
                             "MTLDevice::newAccelerationStructure refused the allocation")};
        if (!as)
        {
            return std::unexpected{as.error()};
        }
        if (!desc.DebugName.empty())
        {
            (*as)->setLabel(MakeLabel(desc.DebugName));
        }

        const uint32_t idx{_accelStructs.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many acceleration structures as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        _addResident(*as->get());
        _accelStructs.Get(idx) = MetalAccelerationStructure{
            .As = std::move(*as), .Type = desc.Type, .Size = sizes.AccelerationStructureSize};
        return AccelerationStructureHandle{.Index = idx};
    }

    void MetalDevice::DestroyAccelerationStructure(AccelerationStructureHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _removeResident(*_accelStructs.Get(h.Index).As.get());
        _accelStructs.Free(h.Index);
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
        return GpuAddress{.Address = _accelStructs.Get(h.Index).As->gpuResourceID()._impl};
    }

    // ============================================================================
    // Shader loading helper
    // ============================================================================

    NS::SharedPtr<MTL::Library> MetalDevice::_loadLibrary(const ShaderDesc &sd)
    {
        // Dispatch on which bytecode alternative is active.
        // std::visit with an if-constexpr generic lambda — no RTTI required.
        NS::Error *err{nullptr};

        return std::visit(
            [&](auto &&src) -> NS::SharedPtr<MTL::Library>
            {
                using T = std::decay_t<decltype(src)>;

                if constexpr (std::is_same_v<T, std::monostate>)
                {
                    FailContract("_loadLibrary: ShaderDesc has no bytecode (monostate)");
                }
                else if constexpr (std::is_same_v<T, SpirvBytecode>)
                {
                    FailContract("_loadLibrary: Metal backend received SPIR-V ShaderDesc");
                }
                else if constexpr (std::is_same_v<T, MetalLibBytecode>)
                {
                    // Pre-compiled .metallib bytes — fastest path (shipping builds).
                    dispatch_data_t dd{dispatch_data_create(src.Bytes.data(), src.Bytes.size(), nullptr,
                                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT)};
                    auto            lib{NS::TransferPtr(_device->newLibrary(dd, &err))};
                    // dispatch_data_create returns a +1 reference; balance it.
                    dispatch_release(dd);
                    if (!lib)
                    {
                        FailContract("MTL metallib load: " + ErrorText(err));
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
                    auto opts{AdoptRequired(MTL::CompileOptions::alloc()->init(),
                                            "MTL::CompileOptions could not be created")};
                    opts->setLanguageVersion(MTL::LanguageVersion3_0);
                    auto lib{NS::TransferPtr(_device->newLibrary(str.get(), opts.get(), &err))};
                    if (!lib)
                    {
                        FailContract("MTL shader compile: " + ErrorText(err));
                    }
                    return lib;
                }
            },
            sd.Bytecode);
    }

    // ============================================================================
    // Graphics pipeline
    // ============================================================================

    void MetalDevice::_configurePipelineForDebugging(MTL4::PipelineDescriptor &descriptor) const
    {
        if (!_debugCaptureEnabled && !_gpuValidationEnabled)
        {
            return;
        }
        auto options{
            AdoptRequired(MTL4::PipelineOptions::alloc()->init(), "MTL4::PipelineOptions could not be created")};
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
        descriptor.setOptions(options.get());
    }

    std::expected<PipelineHandle, DeviceError> MetalDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc &desc)
    {
        auto created{AdoptCreated(MTL4::RenderPipelineDescriptor::alloc()->init(),
                                  "MTL4::RenderPipelineDescriptor could not be created")};
        if (!created)
        {
            return std::unexpected{created.error()};
        }
        auto pd{std::move(*created)};
        _configurePipelineForDebugging(*pd.get());

        // Vertex shader — skip if not provided (monostate = "no shader")
        if (!std::holds_alternative<std::monostate>(desc.VertexShader.Bytecode))
        {
            auto  vsLib{_loadLibrary(desc.VertexShader)};
            auto *name{MakeLabel(desc.VertexShader.EntryPoint)};
            auto  vsDescCreated{AdoptCreated(MTL4::LibraryFunctionDescriptor::alloc()->init(),
                                             "MTL4::LibraryFunctionDescriptor could not be created")};
            if (!vsDescCreated)
            {
                return std::unexpected{vsDescCreated.error()};
            }
            auto vsDesc{std::move(*vsDescCreated)};
            vsDesc->setLibrary(vsLib.get());
            vsDesc->setName(name);
            pd->setVertexFunctionDescriptor(vsDesc.get());
        }

        // Fragment shader — skip if not provided (monostate = "no shader")
        if (!std::holds_alternative<std::monostate>(desc.FragmentShader.Bytecode))
        {
            auto  fsLib{_loadLibrary(desc.FragmentShader)};
            auto *name{MakeLabel(desc.FragmentShader.EntryPoint)};
            auto  fsDescCreated{AdoptCreated(MTL4::LibraryFunctionDescriptor::alloc()->init(),
                                             "MTL4::LibraryFunctionDescriptor could not be created")};
            if (!fsDescCreated)
            {
                return std::unexpected{fsDescCreated.error()};
            }
            auto fsDesc{std::move(*fsDescCreated)};
            fsDesc->setLibrary(fsLib.get());
            fsDesc->setName(name);
            pd->setFragmentFunctionDescriptor(fsDesc.get());
        }

        // Topology class (Metal PSO needs it for tessellation; rasterized topology set at draw)
        pd->setInputPrimitiveTopology(_toTopology(desc.Topology));

        // Color attachments
        for (size_t i{0}; i < desc.ColorFormats.size(); ++i)
        {
            auto *ca{pd->colorAttachments()->object(static_cast<NS::UInteger>(i))};
            ca->setPixelFormat(_toPixFmt(desc.ColorFormats[i]));
            if (i < desc.ColorBlend.size() && desc.ColorBlend[i].Enable)
            {
                const auto &bs{desc.ColorBlend[i]};
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
            pd->setLabel(MakeLabel(desc.DebugName));
        }

        NS::Error *err{nullptr};
        auto pso{NS::TransferPtr(_compiler->newRenderPipelineState(pd.get(), /*compilerTaskOptions*/ nullptr, &err))};

        if (!pso)
        {
            FailContract("RenderPipelineState: " + ErrorText(err));
        }

        // Depth-stencil state. SetPipeline() skips a missing one, so a refusal here leaves the
        // encoder's default depth state rather than failing the pipeline.
        NS::SharedPtr<MTL::DepthStencilState> dss{
            NS::TransferPtr(_device->newDepthStencilState(_makeDepthStencilDesc(desc.DepthStencil).get()))};

        const uint32_t idx{_pipelines.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many pipelines as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        _pipelines.Get(idx) = MetalPipeline{
            .RenderPso         = std::move(pso),
            .ComputePso        = {},
            .DepthStencilState = std::move(dss),
            .Winding           = _toWinding(desc.Rasterizer.FrontFace),
            .CullMode          = _toCull(desc.Rasterizer.CullMode),
            .FillMode          = (desc.Rasterizer.FillMode == FillMode::Wireframe) ? MTL::TriangleFillModeLines
                                                                                   : MTL::TriangleFillModeFill,
            .DepthBiasConstant = desc.Rasterizer.DepthBiasConstant,
            .DepthBiasSlope    = desc.Rasterizer.DepthBiasSlope,
            .IsCompute         = false,
        };
        return PipelineHandle{.Index = idx};
    }

    // ============================================================================
    // Compute pipeline
    // ============================================================================

    std::expected<PipelineHandle, DeviceError> MetalDevice::CreateComputePipeline(const ComputePipelineDesc &desc)
    {
        auto  lib{_loadLibrary(desc.Shader)};
        auto *name{MakeLabel(desc.Shader.EntryPoint)};
        auto  fnDescCreated{AdoptCreated(MTL4::LibraryFunctionDescriptor::alloc()->init(),
                                         "MTL4::LibraryFunctionDescriptor could not be created")};
        if (!fnDescCreated)
        {
            return std::unexpected{fnDescCreated.error()};
        }
        auto fnDesc{std::move(*fnDescCreated)};
        fnDesc->setLibrary(lib.get());
        fnDesc->setName(name);

        auto pipelineDescCreated{AdoptCreated(MTL4::ComputePipelineDescriptor::alloc()->init(),
                                              "MTL4::ComputePipelineDescriptor could not be created")};
        if (!pipelineDescCreated)
        {
            return std::unexpected{pipelineDescCreated.error()};
        }
        auto pipelineDesc{std::move(*pipelineDescCreated)};
        pipelineDesc->setComputeFunctionDescriptor(fnDesc.get());
        _configurePipelineForDebugging(*pipelineDesc.get());
        if (!desc.DebugName.empty())
        {
            pipelineDesc->setLabel(MakeLabel(desc.DebugName));
        }

        // Resource binding needs no pipeline reflection. Bindless resources
        // travel as native addresses/resource IDs in root data; explicitly
        // slotted resources use the command list's MTL4 argument table.
        NS::Error *err{nullptr};
        auto       pso{NS::TransferPtr(
            _compiler->newComputePipelineState(pipelineDesc.get(), /*compilerTaskOptions*/ nullptr, &err))};

        if (!pso)
        {
            FailContract("ComputePipelineState: " + ErrorText(err));
        }

        // Each axis defaults independently: an explicit non-zero component
        // from the caller wins, a zero component (including the ComputePipelineDesc
        // default of {0,0,0}) falls back to 1 — except X, which falls back to
        // the shader's own [[max_total_threads_per_threadgroup]] annotation.
        const uint32_t tgX{desc.ThreadGroupSize.Width != 0
                               ? desc.ThreadGroupSize.Width
                               : static_cast<uint32_t>(pso->maxTotalThreadsPerThreadgroup())};
        const uint32_t tgY{desc.ThreadGroupSize.Height != 0 ? desc.ThreadGroupSize.Height : 1};
        const uint32_t tgZ{desc.ThreadGroupSize.Depth != 0 ? desc.ThreadGroupSize.Depth : 1};

        const uint32_t idx{_pipelines.Alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many pipelines as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        _pipelines.Get(idx) = MetalPipeline{
            .RenderPso         = {},
            .ComputePso        = std::move(pso),
            .DepthStencilState = {},
            .IsCompute         = true,
            .ThreadGroupSizeX  = tgX,
            .ThreadGroupSizeY  = tgY,
            .ThreadGroupSizeZ  = tgZ,
        };
        return PipelineHandle{.Index = idx};
    }

    void MetalDevice::DestroyPipeline(PipelineHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _pipelines.Free(h.Index);
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

        const std::array<const MTL4::CommandBuffer *, 1> buffers{cmdBuffer.get()};
        _queue->commit(buffers.data(), buffers.size());

        const uint64_t value{++_timelineValue};
        _queue->signalEvent(_timelineEvent.get(), value);
        _timelineEvent->waitUntilSignaledValue(value, UINT64_MAX);
    }

    // ============================================================================
    // Upload helpers
    // ============================================================================

    void MetalDevice::UploadBuffer(BufferHandle dst, std::span<const std::byte> data, uint64_t dstOffset)
    {
        if (!dst.Valid() || data.empty())
        {
            return;
        }
        const uint64_t size{data.size_bytes()};
        auto           autoreleasePool{
            AdoptRequired(NS::AutoreleasePool::alloc()->init(), "NS::AutoreleasePool could not be created")};
        auto &b{_buffers.Get(dst.Index)};
        if (b.Buffer->storageMode() == MTL::StorageModeShared)
        {
            const std::span<std::byte> contents{static_cast<std::byte *>(b.Buffer->contents()), b.Buffer->length()};
            std::memcpy(contents.subspan(dstOffset).data(), data.data(), size);
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
            FailContract("UploadBuffer staging allocation failed");
        }
        std::memcpy(staging->contents(), data.data(), size);
        // MTL4 resources must be resident explicitly. Unlike buffers created
        // through CreateBuffer(), this short-lived staging allocation is not
        // otherwise part of the device's persistent residency set.
        _addResident(*staging.get());
        auto allocator{NS::TransferPtr(_device->newCommandAllocator())};
        auto cmdBuffer{NS::TransferPtr(_device->newCommandBuffer())};

        allocator->reset();
        cmdBuffer->beginCommandBuffer(allocator.get());
        if (_debugCaptureEnabled)
        {
            cmdBuffer->setLabel(NS::String::string("HdRestir / LightRHI UploadBuffer", NS::UTF8StringEncoding));
        }
        auto *enc{cmdBuffer->computeCommandEncoder()};
        enc->copyFromBuffer(staging.get(), 0, b.Buffer.get(), dstOffset, size);
        enc->endEncoding();
        cmdBuffer->endCommandBuffer();

        const std::array<const MTL4::CommandBuffer *, 1> buffers{cmdBuffer.get()};
        _queue->commit(buffers.data(), buffers.size());
        const uint64_t value{++_timelineValue};
        _queue->signalEvent(_timelineEvent.get(), value);
        _timelineEvent->waitUntilSignaledValue(value, UINT64_MAX);
        _removeResident(*staging.get());
    }

    void MetalDevice::UploadTexture(TextureHandle dst, std::span<const std::byte> data, uint64_t rowPitch,
                                    uint64_t slicePitch, const TextureCopyRegion &region)
    {
        if (!dst.Valid() || data.empty())
        {
            return;
        }
        const uint64_t totalSize{slicePitch > 0 ? slicePitch : rowPitch * region.Extent.Height};
        if (data.size_bytes() < totalSize)
        {
            FailContract("UploadTexture: the source holds " + std::to_string(data.size_bytes()) +
                         " bytes but the region reads " + std::to_string(totalSize));
        }
        auto autoreleasePool{
            AdoptRequired(NS::AutoreleasePool::alloc()->init(), "NS::AutoreleasePool could not be created")};
        auto staging{NS::TransferPtr(_device->newBuffer(totalSize, MTL::ResourceStorageModeShared))};
        if (!staging)
        {
            FailContract("UploadTexture staging allocation failed");
        }
        std::memcpy(staging->contents(), data.data(), totalSize);
        _addResident(*staging.get());
        auto &t{_textures.Get(dst.Index)};

        auto allocator{NS::TransferPtr(_device->newCommandAllocator())};
        auto cmdBuffer{NS::TransferPtr(_device->newCommandBuffer())};

        allocator->reset();
        cmdBuffer->beginCommandBuffer(allocator.get());
        if (_debugCaptureEnabled)
        {
            cmdBuffer->setLabel(NS::String::string("HdRestir / LightRHI UploadTexture", NS::UTF8StringEncoding));
        }
        auto *enc{cmdBuffer->computeCommandEncoder()};
        enc->copyFromBuffer(staging.get(), 0, rowPitch, slicePitch,
                            MTL::Size::Make(region.Extent.Width, region.Extent.Height, region.Extent.Depth),
                            t.Texture.get(), region.ArrayLayer, region.MipLevel,
                            MTL::Origin::Make(static_cast<NS::UInteger>(region.DstOffset.X),
                                              static_cast<NS::UInteger>(region.DstOffset.Y),
                                              static_cast<NS::UInteger>(region.DstOffset.Z)));
        enc->endEncoding();
        cmdBuffer->endCommandBuffer();

        const std::array<const MTL4::CommandBuffer *, 1> buffers{cmdBuffer.get()};
        _queue->commit(buffers.data(), buffers.size());
        const uint64_t value{++_timelineValue};
        _queue->signalEvent(_timelineEvent.get(), value);
        _timelineEvent->waitUntilSignaledValue(value, UINT64_MAX);
        _removeResident(*staging.get());
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
            case MemoryType::GpuToCpu:
                return MTL::ResourceStorageModeShared;
        }
        return MTL::ResourceStorageModePrivate;
    }

    MTL::StorageMode MetalDevice::_toStorageMode(MemoryType m) noexcept
    {
        switch (m)
        {
            case MemoryType::GpuOnly:
                return MTL::StorageModePrivate;
            case MemoryType::CpuToGpu:
            case MemoryType::GpuToCpu:
                return MTL::StorageModeShared;
        }
        return MTL::StorageModePrivate;
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

    NS::SharedPtr<MTL::DepthStencilDescriptor> MetalDevice::_makeDepthStencilDesc(const DepthStencilState &ds)
    {
        auto dsd{AdoptRequired(MTL::DepthStencilDescriptor::alloc()->init(),
                               "MTL::DepthStencilDescriptor could not be created")};
        dsd->setDepthWriteEnabled(ds.DepthWrite);
        dsd->setDepthCompareFunction(ds.DepthTest ? _toCompare(ds.DepthOp) : MTL::CompareFunctionAlways);
        return dsd;
    }

} // namespace rhi::metal
