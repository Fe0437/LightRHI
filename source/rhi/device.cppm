module;
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
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

    /// \brief Describes an optional dependency for one command-list submission.
    ///
    /// Leave `WaitFence` invalid for an independent submission. When it is valid,
    /// the submitted work waits for `WaitValue`, or for the fence's own value when
    /// `WaitValue` is zero.
    struct SubmitDesc
    {
        FenceHandle WaitFence{};  ///< Optional GPU-side dependency from an earlier submission.
        uint64_t    WaitValue{0}; ///< Timeline value to wait for; zero uses `WaitFence.Id`.
    };

    /// \brief Owns GPU resources and submits work to one physical GPU.
    ///
    /// Handles created by a device are valid only with that device. Destroy a
    /// resource only after every submission that uses it has completed. `IDevice`
    /// itself is not thread-safe; use SynchronizedDevice when access is shared.
    ///
    /// A typical submission creates a command list, records between Begin() and
    /// End(), submits it, and retains all referenced resources until the returned
    /// fence completes.
    ///
    /// \par Typical frame recording
    /// \code{.cpp}
    /// auto cmd = device->CreateCommandList(QueueType::Graphics, "main frame");
    /// cmd->Begin();
    /// cmd->Transition(backbuffer, ResourceState::Undefined, ResourceState::RenderTarget);
    /// cmd->FlushBarriers();
    /// cmd->BeginRendering({
    ///     .Color      = {{.Texture = backbuffer}},
    ///     .RenderArea = {.Width = width, .Height = height},
    /// });
    /// cmd->SetPipeline(pipeline);
    /// cmd->SetPushConstants(FrameConstants{});
    /// cmd->Draw(3);
    /// cmd->EndRendering();
    /// cmd->Transition(backbuffer, ResourceState::RenderTarget, ResourceState::Present);
    /// cmd->FlushBarriers();
    /// cmd->End();
    /// const FenceHandle fence = device->Submit(*cmd);
    /// device->WaitForFence(fence);
    /// \endcode
    /// A window or presentation layer may present the texture after the submitted
    /// transition to ResourceState::Present; presentation is outside IDevice.
    class IDevice
    {
      public:
        /// \brief Releases the device after all owned resources have been destroyed.
        virtual ~IDevice() = default;

        // ---- Device info ----

        /// \brief Returns the user-facing name of the selected GPU adapter.
        /// \return A view that remains valid for the lifetime of the device.
        [[nodiscard]] virtual std::string_view AdapterName() const noexcept = 0;

        /// \brief Returns the adapter's reported local-memory capacity in bytes.
        /// \note Treat this as device information, not as the currently available allocation budget.
        [[nodiscard]] virtual uint64_t VideoMemoryBytes() const noexcept = 0;

        // ---- Bindless heap ----

        /// \brief Returns the device-wide bindless heap used by resources from this device.
        /// \return A reference that remains valid for the lifetime of the device.
        [[nodiscard]] virtual IBindlessHeap &BindlessHeap() noexcept = 0;

        // ---- Buffer ----

        /// \brief Creates a buffer described by `desc`.
        /// \return A valid handle owned by this device.
        [[nodiscard]] virtual BufferHandle CreateBuffer(const BufferDesc &desc) = 0;

        /// \brief Destroys a buffer created by this device.
        /// \pre No pending or future GPU work references `handle`.
        virtual void DestroyBuffer(BufferHandle handle) = 0;

        /// \brief Returns the shader-visible address of a buffer.
        /// \pre The buffer was created with BufferUsage::DeviceAddress.
        /// \return An invalid address when `handle` is invalid or has no device address.
        [[nodiscard]] virtual GpuAddress BufferAddress(BufferHandle handle) const = 0;

        /// \brief Returns the size, usage, memory placement, and address of a buffer.
        /// \pre `handle` names a live buffer created by this device.
        [[nodiscard]] virtual BufferInfo GetBufferInfo(BufferHandle handle) const = 0;

        /// \brief Maps a CPU-visible buffer and returns its writable or readable byte range.
        /// \pre The buffer uses MemoryType::CpuToGpu or MemoryType::GpuToCpu.
        /// \pre The caller has synchronized any GPU access that could race the CPU.
        /// \note Pair every successful mapping with UnmapBuffer().
        [[nodiscard]] virtual MappedBuffer MapBuffer(BufferHandle handle) = 0;

        /// \brief Ends a mapping previously returned by MapBuffer().
        /// \pre `handle` is currently mapped by the caller.
        virtual void UnmapBuffer(BufferHandle handle) = 0;

        // ---- Texture ----

        /// \brief Creates a texture described by `desc` without uploading texel data.
        /// \return A valid handle owned by this device.
        [[nodiscard]] virtual TextureHandle CreateTexture(const TextureDesc &desc) = 0;

        /// \brief Destroys a texture created by this device.
        /// \pre No pending or future GPU work references `handle`.
        virtual void DestroyTexture(TextureHandle handle) = 0;

        /// \brief Returns the shader-visible value for a bindless texture.
        ///
        /// Store the returned value in root or scene data consumed as a Slang
        /// `DescriptorHandle<Texture2D>`. It remains valid until the texture is destroyed.
        [[nodiscard]] virtual GpuAddress TextureAddress(TextureHandle handle) const = 0;

        // ---- Sampler ----

        /// \brief Creates an immutable sampler described by `desc`.
        /// \return A valid handle owned by this device.
        [[nodiscard]] virtual SamplerHandle CreateSampler(const SamplerDesc &desc) = 0;

        /// \brief Destroys a sampler created by this device.
        /// \pre No pending or future GPU work references `handle`.
        virtual void DestroySampler(SamplerHandle handle) = 0;

        /// \brief Returns the shader-visible value for a bindless sampler.
        ///
        /// Prefer placing `SamplerHandle::Index` in a reflected Slang
        /// `DescriptorHandle<SamplerState>` field. Use this method only when a
        /// shared GPU structure represents all bindless values as GpuAddress.
        [[nodiscard]] virtual GpuAddress SamplerAddress(SamplerHandle handle) const = 0;

        // ---- Timestamp queries ----

        /// \brief Reports whether compute command lists can record timestamp queries.
        /// \return `true` when all timestamp-query methods are available.
        [[nodiscard]] virtual bool SupportsComputeTimestamps() const noexcept = 0;

        /// \brief Returns the duration of one timestamp tick in nanoseconds.
        ///
        /// Multiply the unsigned difference between two query results by this
        /// value to obtain elapsed nanoseconds.
        /// \return A positive value when SupportsComputeTimestamps() is true; otherwise zero.
        [[nodiscard]] virtual double TimestampPeriodNanoseconds() const noexcept = 0;

        /// \brief Allocates `count` timestamp slots for WriteComputeTimestamp().
        /// \pre SupportsComputeTimestamps() is true.
        /// \return An invalid handle when `count` is zero.
        [[nodiscard]] virtual TimestampQueryPoolHandle CreateTimestampQueryPool(uint32_t count) = 0;

        /// \brief Releases a timestamp query pool.
        /// \pre All submissions that reference `pool` have completed.
        virtual void DestroyTimestampQueryPool(TimestampQueryPoolHandle pool) = 0;

        /// \brief Makes a contiguous range of timestamp slots available for reuse.
        /// \param pool Pool containing the range.
        /// \param first Index of the first slot to reset.
        /// \param count Number of slots to reset; zero performs no work.
        /// \pre The range is within the pool and no in-flight submission uses it.
        virtual void ResetTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, uint32_t count) = 0;

        /// \brief Copies a contiguous range of completed timestamp results to CPU memory.
        /// \param pool Pool containing the results.
        /// \param first Index of the first result to read.
        /// \param results Destination span; its size is the number of results read.
        /// \pre The requested range is within the pool and its submission fence has completed.
        virtual void ReadTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first,
                                          std::span<uint64_t> results) = 0;

        // ---- Pipeline ----

        /// \brief Creates a graphics pipeline compatible with the formats in `desc`.
        /// \note The shader bytecode views in `desc` need only remain valid for this call.
        [[nodiscard]] virtual PipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) = 0;

        /// \brief Creates a compute pipeline from the compiled shader in `desc`.
        /// \note The shader bytecode view in `desc` need only remain valid for this call.
        [[nodiscard]] virtual PipelineHandle CreateComputePipeline(const ComputePipelineDesc &desc) = 0;

        /// \brief Destroys a graphics or compute pipeline created by this device.
        /// \pre No pending or future GPU work references `handle`.
        virtual void DestroyPipeline(PipelineHandle handle) = 0;

        // ---- Ray tracing ----

        /// \brief Reports whether the device supports acceleration structures and shader ray queries.
        /// \note Check this before using any other acceleration-structure API.
        [[nodiscard]] virtual bool SupportsRayTracing() const noexcept = 0;

        /// \brief Returns the storage and scratch sizes required to build `desc`.
        ///
        /// Query before allocating the scratch buffer passed to
        /// ICommandList::BuildAccelerationStructure().
        /// \pre SupportsRayTracing() is true and every address in `desc` is valid.
        [[nodiscard]] virtual AccelerationStructureBuildSizes
        QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &desc) const = 0;

        /// \brief Allocates an empty acceleration structure matching `desc`.
        ///
        /// Record and submit ICommandList::BuildAccelerationStructure() before
        /// using the returned handle for tracing.
        /// \pre SupportsRayTracing() is true.
        [[nodiscard]] virtual AccelerationStructureHandle
        CreateAccelerationStructure(const AccelerationStructureDesc &desc) = 0;

        /// \brief Destroys an acceleration structure created by this device.
        /// \pre No pending or future GPU work references `handle`.
        virtual void DestroyAccelerationStructure(AccelerationStructureHandle handle) = 0;

        /// \brief Returns the shader-visible value for a bindless acceleration structure.
        ///
        /// Store the returned value in root or scene data consumed as a Slang
        /// `DescriptorHandle<RaytracingAccelerationStructure>`.
        /// \pre `handle` has been built successfully and remains live.
        [[nodiscard]] virtual GpuAddress AccelerationStructureAddress(AccelerationStructureHandle handle) const = 0;

        // ---- Command lists ----

        /// \brief Creates a reusable command-recording object for `queue`.
        /// \param debugName Optional name shown by validation and GPU debugging tools.
        /// \return Exclusive ownership of a command list initially ready for Begin().
        [[nodiscard]] virtual std::unique_ptr<ICommandList> CreateCommandList(QueueType queue = QueueType::Graphics,
                                                                              std::string_view debugName = {}) = 0;

        // ---- Debug capture scopes ----

        /// \brief Begins a named region that supporting GPU debuggers may capture independently.
        /// \note Pair with EndCaptureScope() on the same thread; unsupported tools may ignore it.
        virtual void BeginCaptureScope(std::string_view name) = 0;

        /// \brief Ends the capture scope most recently begun on the calling thread.
        virtual void EndCaptureScope() = 0;

        /// \brief Submits a command list that has completed recording with End().
        ///
        /// The command list and all resources it references must remain alive
        /// until the returned fence completes.
        /// \return A fence identifying completion of this submission.
        [[nodiscard]] virtual FenceHandle Submit(ICommandList &cmdList, const SubmitDesc &desc = {}) = 0;

        /// \brief Blocks the calling thread until `fence` has completed.
        virtual void WaitForFence(FenceHandle fence) = 0;

        /// \brief Tests a submission fence without blocking.
        /// \return `true` once all work represented by `fence` has completed.
        [[nodiscard]] virtual bool IsFenceComplete(FenceHandle fence) = 0;

        /// \brief Blocks until all work submitted to this device has completed.
        /// \note Prefer per-submission fences during normal rendering; use this for teardown or global reconfiguration.
        virtual void WaitIdle() = 0;

        /// \brief Copies CPU bytes into a destination buffer and waits for completion.
        /// \param dst Destination buffer created with BufferUsage::TransferDst.
        /// \param data Source bytes that remain valid for the duration of the call.
        /// \param size Number of bytes to copy.
        /// \param dstOffset Byte offset in the destination buffer.
        /// \note Use recorded copies from a reusable staging buffer for batches or frequent updates.
        virtual void UploadBuffer(BufferHandle dst, const void *data, uint64_t size, uint64_t dstOffset = 0) = 0;

        /// \brief Copies one CPU image region into a texture and waits for completion.
        /// \param dst Destination texture created with TextureUsage::TransferDst.
        /// \param data Source bytes that remain valid for the duration of the call.
        /// \param rowPitch Byte distance between adjacent rows in `data`.
        /// \param slicePitch Byte distance between adjacent depth slices in `data`.
        /// \param region Destination mip, layer, offset, and extent.
        /// \note Use CopyBufferToTexture() with shared staging storage for multiple uploads.
        virtual void UploadTexture(TextureHandle dst, const void *data, uint64_t rowPitch, uint64_t slicePitch,
                                   const TextureCopyRegion &region) = 0;
    };

    /// \brief Serializes access to an owned IDevice for callers that share it across threads.
    ///
    /// Keep the object returned by Synchronize() in the narrowest practical scope;
    /// it holds exclusive access until destroyed.
    ///
    /// \code{.cpp}
    /// SharedDevice device = rhi::AcquireSharedDevice(desc);
    /// {
    ///     auto lockedDevice = device->Synchronize();
    ///     lockedDevice->WaitIdle();
    /// }
    /// \endcode
    class SynchronizedDevice
    {
      public:
        /// \brief Scoped exclusive access to a SynchronizedDevice's IDevice.
        class StrictLockPtr
        {
          public:
            /// \brief Locks `mutex` and exposes `device` until this object is destroyed.
            StrictLockPtr(IDevice &device, std::mutex &mutex) : _device{&device}, _lock{mutex} {}

            /// \brief Returns the locked device.
            [[nodiscard]] IDevice &operator*() const noexcept
            {
                return *_device;
            }
            /// \brief Provides member access to the locked device.
            [[nodiscard]] IDevice *operator->() const noexcept
            {
                return _device;
            }

          private:
            IDevice                     *_device;
            std::unique_lock<std::mutex> _lock;
        };

        /// \brief Takes ownership of `device` and records the descriptor used to create it.
        SynchronizedDevice(std::unique_ptr<IDevice> device, const DeviceDesc &desc)
            : _device{std::move(device)}, _desc{desc}
        {
        }

        /// \brief Acquires exclusive device access for the lifetime of the returned guard.
        [[nodiscard]] StrictLockPtr Synchronize()
        {
            return StrictLockPtr{*_device, _mutex};
        }

        /// \brief Returns the immutable descriptor used to create the device.
        /// \note This accessor does not require Synchronize().
        [[nodiscard]] const DeviceDesc &Desc() const noexcept
        {
            return _desc;
        }

      private:
        std::unique_ptr<IDevice> _device;
        std::mutex               _mutex;
        DeviceDesc               _desc;
    };

    /// \brief Shared ownership of the process-wide synchronized device.
    using SharedDevice = std::shared_ptr<SynchronizedDevice>;

    /// \brief Backend factory accepted by the backend-neutral AcquireSharedDevice() helper.
    using DeviceFactory = std::unique_ptr<IDevice> (*)(const DeviceDesc &);

    /// \brief Acquires the process-wide device, creating it through `factory` on first use.
    ///
    /// Subsequent calls while the device is alive return the same object and must
    /// request matching validation settings.
    /// \return Shared ownership of the synchronized device.
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

} // namespace rhi
