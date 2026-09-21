/**
 * {file} device.cppm
 * {brief} Defines the backend-neutral device lifecycle and synchronized access contract.
 */
module;
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

export module rhi:device;
import :types;
import :handles;
import :descriptors;
import :pipeline;
import :sync;
import :bindless;
import :diagnostics;
import :commandList;
import :raytracing;

export namespace rhi
{

    /**
     * {brief} Describes an optional dependency for one command-list submission.
     *
     * Leave `WaitFence` invalid for an independent submission. When it is valid,
     * the submitted work waits for `WaitValue`, or for the fence's own value when
     * `WaitValue` is zero.
     */
    struct SubmitDesc
    {
        FenceHandle WaitFence{};  ///< Optional GPU-side dependency from an earlier submission.
        uint64_t    WaitValue{0}; ///< Timeline value to wait for; zero uses `WaitFence.Id`.
    };

    /**
     * {brief} Owns GPU resources and submits work to one physical GPU.
     *
     * Handles created by a device are valid only with that device. Destroy a
     * resource only after every submission that uses it has completed. `IDevice`
     * itself is not thread-safe; use SynchronizedDevice when access is shared.
     *
     * A typical submission creates a command list, records between Begin() and
     * End(), submits it, and retains all referenced resources until the returned
     * fence completes.
     *
     * \par Typical frame recording
     * ```cpp
     * auto cmd{device->CreateCommandList(QueueType::Graphics, "main frame")};
     * cmd->Begin();
     * cmd->Transition(backbuffer, ResourceState::Undefined, ResourceState::RenderTarget);
     * cmd->FlushBarriers();
     * cmd->BeginRendering({
     *     .Color      = {{.Texture = backbuffer}},
     *     .RenderArea = {.Width = width, .Height = height},
     * });
     * cmd->SetPipeline(pipeline);
     * cmd->SetPushConstants(FrameConstants{});
     * cmd->Draw(3);
     * cmd->EndRendering();
     * cmd->Transition(backbuffer, ResourceState::RenderTarget, ResourceState::Present);
     * cmd->FlushBarriers();
     * cmd->End();
     * const FenceHandle fence = device->Submit(*cmd);
     * device->WaitForFence(fence);
     * ```
     * Work meant to be seen is submitted the same way: take this frame's texture from an
     * IExternalTextureProvider, record work against it, record ICommandList::Present() and submit. The
     * device stays what it is — resources, queues and submission — and what the window is, how big
     * it is and when to draw are the provider's and the caller's business.
     */
    class IDevice
    {
      public:
        /** {brief} Releases the device after all owned resources have been destroyed. */
        virtual ~IDevice() = default;

        // ---- Device info ----

        /**
         * {brief} Returns the user-facing name of the selected GPU adapter.
         * {returns} A view that remains valid for the lifetime of the device.
         */
        [[nodiscard]] virtual std::string_view AdapterName() const noexcept = 0;

        /**
         * {brief} Returns the adapter's reported local-memory capacity in bytes.
         * {note} Treat this as device information, not as the currently available allocation budget.
         */
        [[nodiscard]] virtual uint64_t VideoMemoryBytes() const noexcept = 0;

        // ---- Bindless heap ----

        /**
         * {brief} Returns the device-wide bindless heap used by resources from this device.
         * {returns} A reference that remains valid for the lifetime of the device.
         */
        [[nodiscard]] virtual IBindlessHeap &BindlessHeap() noexcept = 0;

        // ---- Memory ----

        /**
         * {brief} Creates a heap of device memory that the caller places textures into.
         *
         * Use a heap when the placement pattern is the caller's to decide: reusing one region for
         * many short-lived textures, or keeping related textures in one allocation. The device
         * neither chooses offsets nor tracks what a placement means.
         * {returns} A valid handle owned by this device.
         * {pre} `desc.Size` is non-zero.
         */
        [[nodiscard]] virtual std::expected<MemoryHeapHandle, DeviceError>
        CreateMemoryHeap(const MemoryHeapDesc &desc) = 0;

        /**
         * {brief} Destroys a memory heap created by this device.
         * {pre} Every texture placed in `handle` has been destroyed and no pending GPU work uses them.
         */
        virtual void DestroyMemoryHeap(MemoryHeapHandle handle) = 0;

        /**
         * {brief} Reports the memory one texture needs when placed in a heap of `memory`.
         *
         * Query this before creating the heap, because the requirement depends on the memory that
         * heap will hold: a device that cannot give its fastest texture layout CPU-visible memory
         * reports what its next-best layout needs there instead.
         * {returns} Non-zero size and alignment to place `desc`, or why the device cannot place it.
         */
        [[nodiscard]] virtual std::expected<PlacementRequirements, PlacementError>
        TexturePlacementRequirements(const TextureDesc &desc, MemoryType memory) const = 0;

        // ---- Buffer ----

        /**
         * {brief} Creates a buffer described by `desc`.
         * {returns} A valid handle owned by this device.
         */
        [[nodiscard]] virtual std::expected<BufferHandle, DeviceError> CreateBuffer(const BufferDesc &desc) = 0;

        /**
         * {brief} Reports the memory one buffer needs when placed in a heap.
         *
         * The requirement follows `desc`, including the memory it asks for, so query it before
         * creating the heap the buffer will live in.
         * {returns} Non-zero size and alignment to place `desc`, or why the device cannot place it.
         */
        [[nodiscard]] virtual std::expected<PlacementRequirements, PlacementError>
        BufferPlacementRequirements(const BufferDesc &desc) const = 0;

        /**
         * {brief} Creates a buffer that occupies caller-owned memory at `placement`.
         *
         * The buffer behaves like any other buffer, BufferAddress() included, so shaders reach it
         * through the same address they would use for one that owns its allocation; only its
         * storage differs. The heap's memory type must match the one `desc` asks for. The offset
         * must satisfy BufferPlacementRequirements() and must not overlap a live placement.
         * Reusing the bytes of a destroyed placement is allowed once the timeline point covering
         * its last use has completed; reusing them earlier is undefined.
         * {returns} A valid handle, or the reason the placement was refused.
         */
        [[nodiscard]] virtual std::expected<BufferHandle, PlacementError>
        CreateBuffer(const BufferDesc &desc, const HeapPlacement &placement) = 0;

        /**
         * {brief} Destroys a buffer created by this device.
         * {pre} No pending or future GPU work references `handle`.
         * {note} Destroying a placed buffer releases the buffer, never its memory heap.
         */
        virtual void DestroyBuffer(BufferHandle handle) = 0;

        /**
         * {brief} Returns the shader-visible address of a buffer.
         * {pre} The buffer was created with BufferUsage::DeviceAddress.
         * {returns} An invalid address when `handle` is invalid or has no device address.
         */
        [[nodiscard]] virtual GpuAddress BufferAddress(BufferHandle handle) const = 0;

        /**
         * {brief} Returns the size, usage, memory placement, and address of a buffer.
         * {pre} `handle` names a live buffer created by this device.
         */
        [[nodiscard]] virtual BufferInfo GetBufferInfo(BufferHandle handle) const = 0;

        /**
         * {brief} Returns the size, format and usage of a texture.
         * {pre} `handle` names a live texture created by this device, or a frame taken from a
         * provider and not yet shown.
         */
        [[nodiscard]] virtual TextureInfo GetTextureInfo(TextureHandle handle) const = 0;

        /**
         * {brief} Maps a CPU-visible buffer and returns its writable or readable byte range.
         * {pre} The buffer uses MemoryType::CpuToGpu or MemoryType::GpuToCpu.
         * {pre} The caller has synchronized any GPU access that could race the CPU.
         * {note} Pair every successful mapping with UnmapBuffer().
         */
        [[nodiscard]] virtual MappedBuffer MapBuffer(BufferHandle handle) = 0;

        /**
         * {brief} Ends a mapping previously returned by MapBuffer().
         * {pre} `handle` is currently mapped by the caller.
         */
        virtual void UnmapBuffer(BufferHandle handle) = 0;

        // ---- Texture ----

        /**
         * {brief} Creates a texture described by `desc` without uploading texel data.
         * {returns} A valid handle owned by this device.
         */
        [[nodiscard]] virtual std::expected<TextureHandle, DeviceError> CreateTexture(const TextureDesc &desc) = 0;

        /**
         * {brief} Creates a texture that occupies caller-owned memory at `placement`.
         *
         * The texture behaves like any other texture and is destroyed with DestroyTexture(); only
         * its storage differs. A heap of any memory type can hold a texture where the device allows
         * it; where the fastest layout cannot live in that memory, the device uses the best layout
         * that can, so the same call keeps working with different performance. When no layout fits,
         * the result is PlacementError::IncompatibleMemory rather than a surprise. The offset must
         * satisfy TexturePlacementRequirements() for the heap's memory type and must not overlap a
         * live placement. Reusing the bytes of a destroyed placement is allowed once the
         * timeline point covering its last use has completed; reusing them earlier is undefined.
         * {returns} A valid handle, or the reason the placement was refused.
         */
        [[nodiscard]] virtual std::expected<TextureHandle, PlacementError>
        CreateTexture(const TextureDesc &desc, const HeapPlacement &placement) = 0;

        /**
         * {brief} Destroys a texture created by this device.
         * {pre} No pending or future GPU work references `handle`.
         * {note} Destroying a placed texture releases the texture, never its memory heap.
         */
        virtual void DestroyTexture(TextureHandle handle) = 0;

        /**
         * {brief} Returns the shader-visible value for a bindless texture.
         *
         * Store the returned value in root or scene data consumed as a Slang
         * `DescriptorHandle<Texture2D>`. It remains valid until the texture is destroyed.
         */
        [[nodiscard]] virtual GpuAddress TextureAddress(TextureHandle handle) const = 0;

        // ---- Sampler ----

        /**
         * {brief} Creates an immutable sampler described by `desc`.
         * {returns} A valid handle owned by this device.
         */
        [[nodiscard]] virtual std::expected<SamplerHandle, DeviceError> CreateSampler(const SamplerDesc &desc) = 0;

        /**
         * {brief} Destroys a sampler created by this device.
         * {pre} No pending or future GPU work references `handle`.
         */
        virtual void DestroySampler(SamplerHandle handle) = 0;

        /**
         * {brief} Returns the shader-visible value for a bindless sampler.
         *
         * Prefer placing `SamplerHandle::Index` in a reflected Slang
         * `DescriptorHandle<SamplerState>` field. Use this method only when a
         * shared GPU structure represents all bindless values as GpuAddress.
         */
        [[nodiscard]] virtual GpuAddress SamplerAddress(SamplerHandle handle) const = 0;

        // ---- Timestamp queries ----

        /**
         * {brief} Reports whether compute command lists can record timestamp queries.
         * {returns} `true` when all timestamp-query methods are available.
         */
        [[nodiscard]] virtual bool SupportsComputeTimestamps() const noexcept = 0;

        /**
         * {brief} Returns the duration of one timestamp tick in nanoseconds.
         *
         * Multiply the unsigned difference between two query results by this
         * value to obtain elapsed nanoseconds.
         * {returns} A positive value when SupportsComputeTimestamps() is true; otherwise zero.
         */
        [[nodiscard]] virtual double TimestampPeriodNanoseconds() const noexcept = 0;

        /**
         * {brief} Allocates `count` timestamp slots for WriteComputeTimestamp().
         * {pre} SupportsComputeTimestamps() is true.
         * {returns} An invalid handle when `count` is zero.
         */
        [[nodiscard]] virtual std::expected<TimestampQueryPoolHandle, DeviceError>
        CreateTimestampQueryPool(uint32_t count) = 0;

        /**
         * {brief} Releases a timestamp query pool.
         * {pre} All submissions that reference `pool` have completed.
         */
        virtual void DestroyTimestampQueryPool(TimestampQueryPoolHandle pool) = 0;

        /**
         * {brief} Makes a contiguous range of timestamp slots available for reuse.
         * {param pool} Pool containing the range.
         * {param first} Index of the first slot to reset.
         * {param count} Number of slots to reset; zero performs no work.
         * {pre} The range is within the pool and no in-flight submission uses it.
         */
        virtual void ResetTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, uint32_t count) = 0;

        /**
         * {brief} Copies a contiguous range of completed timestamp results to CPU memory.
         * {param pool} Pool containing the results.
         * {param first} Index of the first result to read.
         * {param results} Destination span; its size is the number of results read.
         * {pre} The requested range is within the pool and its submission fence has completed.
         */
        virtual void ReadTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first,
                                          std::span<uint64_t> results) = 0;

        // ---- Pipeline ----

        /**
         * {brief} Creates a graphics pipeline compatible with the formats in `desc`.
         * {note} The shader bytecode views in `desc` need only remain valid for this call.
         */
        [[nodiscard]] virtual std::expected<PipelineHandle, DeviceError>
        CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) = 0;

        /**
         * {brief} Creates a compute pipeline from the compiled shader in `desc`.
         * {note} The shader bytecode view in `desc` need only remain valid for this call.
         */
        [[nodiscard]] virtual std::expected<PipelineHandle, DeviceError>
        CreateComputePipeline(const ComputePipelineDesc &desc) = 0;

        /**
         * {brief} Destroys a graphics or compute pipeline created by this device.
         * {pre} No pending or future GPU work references `handle`.
         */
        virtual void DestroyPipeline(PipelineHandle handle) = 0;

        // ---- Ray tracing ----

        /**
         * {brief} Reports whether the device supports acceleration structures and shader ray queries.
         * {note} Check this before using any other acceleration-structure API.
         */
        [[nodiscard]] virtual bool SupportsRayTracing() const noexcept = 0;

        /**
         * {brief} Returns the storage and scratch sizes required to build `desc`.
         *
         * Query before allocating the scratch buffer passed to
         * ICommandList::BuildAccelerationStructure().
         * {pre} SupportsRayTracing() is true and every address in `desc` is valid.
         */
        [[nodiscard]] virtual AccelerationStructureBuildSizes
        QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &desc) const = 0;

        /**
         * {brief} Allocates an empty acceleration structure matching `desc`.
         *
         * Record and submit ICommandList::BuildAccelerationStructure() before
         * using the returned handle for tracing.
         * {pre} SupportsRayTracing() is true.
         */
        [[nodiscard]] virtual std::expected<AccelerationStructureHandle, DeviceError>
        CreateAccelerationStructure(const AccelerationStructureDesc &desc) = 0;

        /**
         * {brief} Destroys an acceleration structure created by this device.
         * {pre} No pending or future GPU work references `handle`.
         */
        virtual void DestroyAccelerationStructure(AccelerationStructureHandle handle) = 0;

        /**
         * {brief} Returns the shader-visible value for a bindless acceleration structure.
         *
         * Store the returned value in root or scene data consumed as a Slang
         * `DescriptorHandle<RaytracingAccelerationStructure>`.
         * {pre} `handle` has been built successfully and remains live.
         */
        [[nodiscard]] virtual GpuAddress AccelerationStructureAddress(AccelerationStructureHandle handle) const = 0;

        // ---- Command lists ----

        /**
         * {brief} Creates a reusable command-recording object for `queue`.
         * {param debugName} Optional name shown by validation and GPU debugging tools.
         * {returns} Exclusive ownership of a command list initially ready for Begin().
         */
        [[nodiscard]] virtual std::expected<std::unique_ptr<ICommandList>, DeviceError>
        CreateCommandList(QueueType queue = QueueType::Graphics, std::string_view debugName = {}) = 0;

        // ---- Debug capture scopes ----

        /**
         * {brief} Begins a named region that supporting GPU debuggers may capture independently.
         * {note} Pair with EndCaptureScope() on the same thread; unsupported tools may ignore it.
         */
        virtual void BeginCaptureScope(std::string_view name) = 0;

        /** {brief} Ends the capture scope most recently begun on the calling thread. */
        virtual void EndCaptureScope() = 0;

        /**
         * {brief} Submits a command list that has completed recording with End().
         *
         * The command list and all resources it references must remain alive
         * until the returned fence completes.
         * {returns} A fence identifying completion of this submission.
         */
        [[nodiscard]] virtual FenceHandle Submit(ICommandList &cmdList, const SubmitDesc &desc = {}) = 0;

        /** {brief} Blocks the calling thread until `fence` has completed. */
        virtual void WaitForFence(FenceHandle fence) = 0;

        /**
         * {brief} Tests a submission fence without blocking.
         * {returns} `true` once all work represented by `fence` has completed.
         */
        [[nodiscard]] virtual bool IsFenceComplete(FenceHandle fence) = 0;

        /**
         * {brief} Blocks until all work submitted to this device has completed.
         * {note} Prefer per-submission fences during normal rendering; use this for teardown or global reconfiguration.
         */
        virtual void WaitIdle() = 0;

        /**
         * {brief} Copies CPU bytes into a destination buffer and waits for completion.
         * {param dst} Destination buffer created with BufferUsage::TransferDst.
         * {param data} Source bytes that remain valid for the duration of the call; all of them are copied.
         * {param dstOffset} Byte offset in the destination buffer.
         * {note} Use recorded copies from a reusable staging buffer for batches or frequent updates.
         */
        virtual void UploadBuffer(BufferHandle dst, std::span<const std::byte> data, uint64_t dstOffset = 0) = 0;

        /**
         * {brief} Copies the elements of a contiguous range of trivially copyable values into a buffer.
         * {param dst} Destination buffer created with BufferUsage::TransferDst.
         * {param data} Array, vector, span, or other contiguous range whose object representation is copied.
         * {param dstOffset} Byte offset in the destination buffer.
         *
         * ```cpp
         * const std::array<float, 9> vertices{...};
         * device->UploadBuffer(vertexBuffer, vertices);
         * ```
         */
        template <std::ranges::contiguous_range Range>
        requires std::ranges::sized_range<Range> && std::is_trivially_copyable_v<std::ranges::range_value_t<Range>> &&
                 (!std::convertible_to<Range, std::span<const std::byte>>)
        void UploadBuffer(BufferHandle dst, Range &&data, uint64_t dstOffset = 0)
        {
            UploadBuffer(dst, std::as_bytes(std::span{std::ranges::data(data), std::ranges::size(data)}), dstOffset);
        }

        /**
         * {brief} Copies one CPU image region into a texture and waits for completion.
         * {param dst} Destination texture created with TextureUsage::TransferDst.
         * {param data} Source bytes that remain valid for the duration of the call.
         * {param rowPitch} Byte distance between adjacent rows in `data`.
         * {param slicePitch} Byte distance between adjacent depth slices in `data`.
         * {param region} Destination mip, layer, offset, and extent.
         * {pre} `data` holds at least as many bytes as the region reads through `rowPitch` and `slicePitch`.
         * {note} Use CopyBufferToTexture() with shared staging storage for multiple uploads.
         */
        virtual void UploadTexture(TextureHandle dst, std::span<const std::byte> data, uint64_t rowPitch,
                                   uint64_t slicePitch, const TextureCopyRegion &region) = 0;

        /**
         * {brief} Copies one image region held in a contiguous range of trivially copyable texels.
         * {param data} Array, vector, span, or other contiguous range whose object representation is copied.
         * {param rowPitch} Byte distance between adjacent rows in `data`.
         * {param slicePitch} Byte distance between adjacent depth slices in `data`.
         * {param region} Destination mip, layer, offset, and extent.
         */
        template <std::ranges::contiguous_range Range>
        requires std::ranges::sized_range<Range> && std::is_trivially_copyable_v<std::ranges::range_value_t<Range>> &&
                 (!std::convertible_to<Range, std::span<const std::byte>>)
        void UploadTexture(TextureHandle dst, Range &&data, uint64_t rowPitch, uint64_t slicePitch,
                           const TextureCopyRegion &region)
        {
            UploadTexture(dst, std::as_bytes(std::span{std::ranges::data(data), std::ranges::size(data)}), rowPitch,
                          slicePitch, region);
        }
    };

    /**
     * {brief} Serializes access to an owned IDevice for callers that share it across threads.
     *
     * Keep the object returned by Synchronize() in the narrowest practical scope;
     * it holds exclusive access until destroyed.
     *
     * ```cpp
     * SharedDevice device = rhi::AcquireSharedDevice(desc);
     * {
     *     auto lockedDevice{device->Synchronize()};
     *     lockedDevice->WaitIdle();
     * }
     * ```
     */
    class SynchronizedDevice
    {
      public:
        /** {brief} Scoped exclusive access to a SynchronizedDevice's IDevice. */
        class StrictLockPtr
        {
          public:
            /** {brief} Locks `mutex` and exposes `device` until this object is destroyed. */
            StrictLockPtr(IDevice &device, std::mutex &mutex) : _device{device}, _lock{mutex} {}

            /** {brief} Returns the locked device. */
            [[nodiscard]] IDevice &operator*() const noexcept
            {
                return _device;
            }
            /** {brief} Provides member access to the locked device. */
            [[nodiscard]] IDevice *operator->() const noexcept
            {
                return &_device;
            }

          private:
            IDevice                     &_device;
            std::unique_lock<std::mutex> _lock{};
        };

        /** {brief} Takes ownership of `device` and records the descriptor used to create it. */
        SynchronizedDevice(std::unique_ptr<IDevice> device, const DeviceDesc &desc)
            : _device{std::move(device)}, _desc{desc}
        {
        }

        /** {brief} Acquires exclusive device access for the lifetime of the returned guard. */
        [[nodiscard]] StrictLockPtr Synchronize()
        {
            return StrictLockPtr{*_device, _mutex};
        }

        /**
         * {brief} Returns the immutable descriptor used to create the device.
         * {note} This accessor does not require Synchronize().
         */
        [[nodiscard]] const DeviceDesc &Desc() const noexcept
        {
            return _desc;
        }

      private:
        std::unique_ptr<IDevice> _device{};
        std::mutex               _mutex{};
        DeviceDesc               _desc{};
    };

    /** {brief} Shared ownership of the process-wide synchronized device. */
    using SharedDevice = std::shared_ptr<SynchronizedDevice>;

    /** {brief} Backend factory accepted by the backend-neutral AcquireSharedDevice() helper. */
    using DeviceFactory = std::expected<std::unique_ptr<IDevice>, DeviceError> (*)(const DeviceDesc &);

    /**
     * {brief} Acquires the process-wide device, creating it through `factory` on first use.
     *
     * Subsequent calls while the device is alive return the same object and must
     * request matching validation settings.
     * {returns} Shared ownership of the synchronized device.
     */
    [[nodiscard]] inline std::expected<SharedDevice, DeviceError> AcquireSharedDevice(const DeviceDesc &desc,
                                                                                      DeviceFactory     factory)
    {
        static std::mutex                        mutex{};
        static std::weak_ptr<SynchronizedDevice> weakDevice{};

        const std::scoped_lock lock{mutex};

        if (const SharedDevice device{weakDevice.lock()})
        {
            if (device->Desc().EnableValidation != desc.EnableValidation ||
                device->Desc().EnableGpuValidation != desc.EnableGpuValidation)
            {
                // Two parts of one process asking for different validation settings cannot both be
                // served by one device, and neither could act on a null: the settings have to agree.
                FailContract("a shared device already exists with different validation settings");
            }
            return device;
        }

        auto created{factory(desc)};
        if (!created)
        {
            return std::unexpected{created.error()};
        }
        auto device{std::make_shared<SynchronizedDevice>(std::move(*created), desc)};
        weakDevice = device;
        return device;
    }

} // namespace rhi
