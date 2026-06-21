module;
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

export module rhi:device;
import :types;
import :handles;
import :descriptors;
import :pipeline;
import :sync;
import :bindless;
import :commandList;
import :raytracing;

export namespace rhi
{

    // ---------------------------------------------------------------------------
    // IDevice
    //
    // The central object. One per GPU.
    // All resource creation, submission, and synchronisation goes through here.
    //
    // Typical frame loop:
    //
    //   auto cmd = device->CreateCommandList(QueueType::Graphics);
    //   cmd->Begin();
    //
    //   cmd->Transition(backbuffer, ResourceState::Undefined, ResourceState::RenderTarget);
    //   cmd->BeginRendering({ .Color = {{ .Texture = backbuffer }} });
    //   cmd->SetPipeline(myPipeline);
    //   cmd->SetPushConstants(MyConstants{...});
    //   cmd->Draw(3);
    //   cmd->EndRendering();
    //   cmd->Transition(backbuffer, ResourceState::RenderTarget, ResourceState::Present);
    //
    //   cmd->End();
    //   auto fence = device->Submit(*cmd);
    //   device->Present(...);
    //   device->WaitForFence(fence);
    //
    // ---------------------------------------------------------------------------

    struct SubmitDesc
    {
        QueueType Queue{QueueType::Graphics};
        // Optional timeline wait/signal (GPU-GPU sync across queues)
        FenceHandle WaitFence{};
        uint64_t    WaitValue{0};
        uint64_t    SignalValue{0};
    };

    class IDevice
    {
      public:
        virtual ~IDevice() = default;

        // ---- Device info ----

        [[nodiscard]] virtual std::string_view AdapterName() const noexcept      = 0;
        [[nodiscard]] virtual uint64_t         VideoMemoryBytes() const noexcept = 0;

        // ---- Bindless heap ----

        [[nodiscard]] virtual IBindlessHeap &BindlessHeap() noexcept = 0;

        // ---- Buffer ----

        [[nodiscard]] virtual BufferHandle CreateBuffer(const BufferDesc &desc) = 0;
        virtual void                       DestroyBuffer(BufferHandle handle)   = 0;

        [[nodiscard]] virtual GpuAddress BufferAddress(BufferHandle handle) const = 0;
        [[nodiscard]] virtual BufferInfo GetBufferInfo(BufferHandle handle) const = 0;

        // CPU-accessible buffers (MemoryType::CpuToGpu / GpuToCpu only)
        [[nodiscard]] virtual MappedBuffer MapBuffer(BufferHandle handle)   = 0;
        virtual void                       UnmapBuffer(BufferHandle handle) = 0;

        // ---- Texture ----

        [[nodiscard]] virtual TextureHandle CreateTexture(const TextureDesc &desc) = 0;
        virtual void                        DestroyTexture(TextureHandle handle)   = 0;

        // The texture's bindless GPU handle, used in shaders exactly like
        // AccelerationStructureAddress: put it in push constants as a
        // Value for a Slang `DescriptorHandle<Texture2D>` field in root/scene
        // data. Vulkan returns the global sampled-image descriptor-array index.
        // Metal returns MTLTexture::gpuResourceID; the queue residency set keeps
        // every indirectly reachable texture accessible to the GPU.
        [[nodiscard]] virtual GpuAddress TextureAddress(TextureHandle handle) const = 0;

        // ---- Sampler ----

        [[nodiscard]] virtual SamplerHandle CreateSampler(const SamplerDesc &desc) = 0;
        virtual void                        DestroySampler(SamplerHandle handle)   = 0;

        // NOT the same mechanism as TextureAddress: MTLSamplerState has no
        // directly embeddable resource ID (confirmed empirically — writing
        // raw MTLArgumentEncoder-produced bytes into an ordinary push-
        // constant blob, the way TextureAddress's bytes work, produced
        // samples that were non-deterministically corrupted on specific
        // SIMD lanes). Just the handle's slot index on both backends.
        //
        // A PC struct field of type `DescriptorHandle<SamplerState>`
        // (paired with a `DescriptorHandle<Texture2D>` field, matching how
        // a shader actually samples a texture) is handled transparently:
        // ICommandList::SetPushConstants detects, via pipeline reflection
        // captured at CreateComputePipeline time, when the bound pipeline's
        // push-constant struct contains a Sampler-typed member, and in that
        // case re-encodes the whole push-constant blob through the pipeline
        // function's own MTLArgumentEncoder instead of a plain byte copy —
        // reading a TextureHandle.Index/SamplerHandle.Index (not a resolved
        // address) from the caller's data at each resource field's offset.
        // Callers therefore write `.Index` (not `.Address`) for any
        // push-constant field paired with a sampler this way; this method
        // exists for completeness/parity with TextureAddress, not because
        // callers need to call it.
        [[nodiscard]] virtual GpuAddress SamplerAddress(SamplerHandle handle) const = 0;

        // ---- Pipeline ----

        [[nodiscard]] virtual PipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) = 0;
        [[nodiscard]] virtual PipelineHandle CreateComputePipeline(const ComputePipelineDesc &desc)   = 0;
        virtual void                         DestroyPipeline(PipelineHandle handle)                   = 0;

        // ---- Ray tracing ----
        //
        // false on a device/driver lacking acceleration-structure + ray-query
        // support. Callers must check this before creating acceleration
        // structures; the other ray tracing methods are undefined behaviour
        // (backends assert / throw) if called when this is false.
        [[nodiscard]] virtual bool SupportsRayTracing() const noexcept = 0;

        // Mirrors vkGetAccelerationStructureBuildSizesKHR: query the backing-
        // buffer and scratch-buffer sizes required to build `desc` *before*
        // allocating anything, so CreateAccelerationStructure and the scratch
        // buffer passed to ICommandList::BuildAccelerationStructure can both
        // be sized up front.
        [[nodiscard]] virtual AccelerationStructureBuildSizes
        QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &desc) const = 0;

        // Allocates the acceleration structure object and its backing buffer,
        // sized from QueryAccelerationStructureBuildSizes(desc). Does NOT
        // build geometry into it — the object is empty until a matching
        // ICommandList::BuildAccelerationStructure() call is recorded and
        // submitted.
        [[nodiscard]] virtual AccelerationStructureHandle
                     CreateAccelerationStructure(const AccelerationStructureDesc &desc) = 0;
        virtual void DestroyAccelerationStructure(AccelerationStructureHandle handle)   = 0;

        // The acceleration structure's 8-byte bindless GPU handle, used in
        // shaders exactly like buffer BDA: put it in push constants as a
        // `DescriptorHandle<RaytracingAccelerationStructure>` field and
        // Slang converts it back to a usable RaytracingAccelerationStructure
        // on both backends — no per-dispatch bind call needed.
        //   Vulkan: the AS device address (vkGetAccelerationStructureDeviceAddressKHR);
        //           shaders lower to OpConvertUToAccelerationStructureKHR.
        //   Metal:  the MTLAccelerationStructure gpuResourceID bits, read with
        //           argument-buffer semantics; the backend keeps live
        //           acceleration structures resident automatically.
        [[nodiscard]] virtual GpuAddress AccelerationStructureAddress(AccelerationStructureHandle handle) const = 0;

        // ---- Command lists ----

        [[nodiscard]] virtual std::unique_ptr<ICommandList> CreateCommandList(QueueType queue = QueueType::Graphics,
                                                                              std::string_view debugName = {}) = 0;

        // ---- Debug capture scopes ----
        // Marks a named region that GPU debuggers can offer as a focused
        // capture scope. Backends without capture-scope support may no-op.
        // Begin/End must be paired on the same thread.
        virtual void BeginCaptureScope(std::string_view name) = 0;
        virtual void EndCaptureScope()                        = 0;

        // ---- Submission ----
        //
        // Submit() takes ownership of the recorded commands and schedules
        // them for execution. Returns a fence for CPU-side waiting.

        [[nodiscard]] virtual FenceHandle Submit(ICommandList &cmdList, const SubmitDesc &desc = {}) = 0;

        virtual void               WaitForFence(FenceHandle fence)    = 0;
        [[nodiscard]] virtual bool IsFenceComplete(FenceHandle fence) = 0;

        // Block until all queues are idle. Only use during shutdown / resize.
        virtual void WaitIdle() = 0;

        // ---- Upload helpers ----
        //
        // Convenience for staging uploads. Internally creates a staging buffer,
        // records a copy command, submits on the transfer queue, and waits.
        // For high-frequency uploads use a streaming ring buffer instead.

        virtual void UploadBuffer(BufferHandle dst, const void *data, uint64_t size, uint64_t dstOffset = 0) = 0;

        virtual void UploadTexture(TextureHandle dst, const void *data, uint64_t rowPitch, uint64_t slicePitch,
                                   const TextureCopyRegion &region) = 0;
    };

    // ---------------------------------------------------------------------------
    // SynchronizedDevice
    //
    // A monitor over IDevice, which is not thread-safe: it owns the device and
    // the mutex guarding it, and the only way to reach the device is through a
    // StrictLockPtr that holds the lock for its (deliberately short) lifetime.
    // Same shape as std::synchronized_value (P0290) / boost::synchronized_value.
    //
    //   auto device = rhi::vulkan::AcquireSharedDevice();
    //   {
    //       auto lockedDevice = device->Synchronize();
    //       lockedDevice->WaitIdle();
    //   }
    //
    // ---------------------------------------------------------------------------

    class SynchronizedDevice
    {
      public:
        // Non-owning pointer to the device, valid only while the lock is held.
        class StrictLockPtr
        {
          public:
            StrictLockPtr(IDevice &device, std::mutex &mutex) : _device{&device}, _lock{mutex} {}

            [[nodiscard]] IDevice &operator*() const noexcept
            {
                return *_device;
            }
            [[nodiscard]] IDevice *operator->() const noexcept
            {
                return _device;
            }

          private:
            IDevice                     *_device;
            std::unique_lock<std::mutex> _lock;
        };

        SynchronizedDevice(std::unique_ptr<IDevice> device, const DeviceDesc &desc)
            : _device{std::move(device)}, _desc{desc}
        {
        }

        [[nodiscard]] StrictLockPtr Synchronize()
        {
            return StrictLockPtr{*_device, _mutex};
        }

        // Immutable after construction, so readable without holding the lock.
        [[nodiscard]] const DeviceDesc &Desc() const noexcept
        {
            return _desc;
        }

      private:
        std::unique_ptr<IDevice> _device;
        std::mutex               _mutex;
        DeviceDesc               _desc;
    };

    // Shared ownership of the process-wide device: it is created on first use and
    // destroyed once the last owner drops it. Null until acquired.
    using SharedDevice = std::shared_ptr<SynchronizedDevice>;

    using DeviceFactory = std::unique_ptr<IDevice> (*)(const DeviceDesc &);

    // Backends wrap this with their own factory; see rhi::vulkan / rhi::metal.
    [[nodiscard]] inline SharedDevice AcquireSharedDevice(const DeviceDesc &desc, DeviceFactory factory)
    {
        static std::mutex                        mutex;
        static std::weak_ptr<SynchronizedDevice> weakDevice;

        const std::scoped_lock lock{mutex};

        if (const SharedDevice device{weakDevice.lock()})
        {
            if (device->Desc().EnableValidation != desc.EnableValidation ||
                device->Desc().EnableGpuValidation != desc.EnableGpuValidation)
            {
                throw std::runtime_error("[LightRHI] shared device already exists with different validation settings");
            }
            return device;
        }

        auto device{std::make_shared<SynchronizedDevice>(factory(desc), desc)};
        weakDevice = device;
        return device;
    }

    // ---------------------------------------------------------------------------
    // Factory — implemented by each backend.
    // ---------------------------------------------------------------------------

    // Forward-declared; defined in backend modules (e.g. rhi.vulkan).
    // Callers include the relevant backend module alongside import rhi;
    //
    //   import rhi;
    //   import rhi.vulkan;
    //
    //   auto device = rhi::vulkan::CreateDevice({.EnableValidation = true});
    //
    // This keeps IDevice free of backend-specific headers.

} // namespace rhi
