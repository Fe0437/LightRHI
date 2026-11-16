/**
 * {file}
 * {brief} Records and executes headless CPU commands against host memory.
 * Shader code is produced by Slang; this file owns resource lifetime and dispatch only.
 */
module;
#include "kernel_abi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <latch>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <slang-cpp-prelude.h>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

module lightRHI.cpu;

import :texture;

namespace rhi::cpu
{
    namespace
    {
        using light_rhi_cpu::Grid;

        // Native module loading is the only platform interop in this backend.
        /**
         * {brief} One native compute library, opened from its bytes. The system loads native code only from a
         * file, so the bytes are written to a file of this process's own, opened, and the file removed once the
         * library is closed.
         */
        class Library
        {
          public:
            explicit Library(const std::span<const std::uint8_t> bytes) : _bytes{bytes.begin(), bytes.end()}
            {
                static std::atomic<std::uint64_t> opened{};
#ifdef _WIN32
                const auto process{static_cast<std::uint64_t>(_getpid())};
#else
                const auto process{static_cast<std::uint64_t>(getpid())};
#endif
                std::error_code error;
                _file = std::filesystem::temp_directory_path(error) /
                        ("lightrhi-cpu-" + std::to_string(process) + "-" + std::to_string(opened++) + ".lrcpu");
                if (error)
                {
                    _openError = "no temporary directory to load native code from";
                    return;
                }
                {
                    std::ofstream file{_file, std::ios::binary | std::ios::trunc};
                    // A stream writes only from char, and char may view any object's bytes.
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    file.write(reinterpret_cast<const char *>(bytes.data()),
                               static_cast<std::streamsize>(bytes.size()));
                    if (!file)
                    {
                        _openError = "could not write " + _file.string();
                        return;
                    }
                }
#ifdef _WIN32
                _handle = LoadLibraryW(_file.c_str());
                if (!_handle)
                {
                    _openError = "LoadLibrary failed with error " + std::to_string(GetLastError());
                }
#else
                _handle = dlopen(_file.c_str(), RTLD_NOW | RTLD_LOCAL);
                if (!_handle)
                {
                    const char *const reason{dlerror()};
                    _openError = reason != nullptr ? reason : "dlopen failed";
                }
#endif
            }

            Library(const Library &)            = delete;
            Library &operator=(const Library &) = delete;

            ~Library()
            {
                if (_handle)
                {
#ifdef _WIN32
                    FreeLibrary(_handle);
#else
                    dlclose(_handle);
#endif
                }
                std::error_code ignored;
                std::filesystem::remove(_file, ignored);
            }

            /** {brief} Whether this library was opened from exactly `bytes`. */
            [[nodiscard]] bool Holds(const std::span<const std::uint8_t> bytes) const noexcept
            {
                return std::ranges::equal(_bytes, bytes);
            }

            /** {brief} Why the library could not be opened; empty once it is open. */
            [[nodiscard]] const std::string &OpenError() const noexcept
            {
                return _openError;
            }

            [[nodiscard]] light_rhi_cpu::FindKernel Find() const
            {
                if (!_handle)
                {
                    return nullptr;
                }
#ifdef _WIN32
                return reinterpret_cast<light_rhi_cpu::FindKernel>(GetProcAddress(_handle, "lightRhiFindCpuKernel"));
#else
                return reinterpret_cast<light_rhi_cpu::FindKernel>(dlsym(_handle, "lightRhiFindCpuKernel"));
#endif
            }

          private:
            std::vector<std::uint8_t> _bytes; ///< What it was opened from, so a cache finds it by its contents.
            std::filesystem::path     _file;
#ifdef _WIN32
            HMODULE _handle{};
#else
            void *_handle{};
#endif
            std::string _openError;
        };

        // malloc storage has the alignment and implicit-lifetime rules needed for shader pointers.
        constexpr std::uint64_t StorageAlignment{alignof(std::max_align_t)};
        constexpr std::uint32_t MaxWorkgroupLanes{1024U};

        [[nodiscard]] std::shared_ptr<std::byte> Allocate(std::uint64_t bytes)
        {
            if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max())
            {
                return {};
            }
            auto *const memory{static_cast<std::byte *>(std::calloc(static_cast<std::size_t>(bytes), 1))};
            return memory ? std::shared_ptr<std::byte>{memory, std::free} : nullptr;
        }

        [[nodiscard]] std::span<std::byte> Slice(std::span<std::byte> storage, std::uint64_t offset, std::uint64_t size)
        {
            if (offset > storage.size() || size > storage.size() - offset)
            {
                FailContract("CPU buffer range is outside its allocation");
            }
            return storage.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(size));
        }

        struct Memory
        {
            std::shared_ptr<std::byte> Owner;
            std::span<std::byte>       Bytes;
        };

        struct Heap
        {
            Memory                                                  Storage;
            MemoryType                                              Type;
            std::shared_ptr<std::map<std::uint64_t, std::uint64_t>> Ranges;
        };

        struct Buffer
        {
            Memory     Storage;
            BufferInfo Info;
        };

        struct Image
        {
            Memory  Storage;
            Texture View;
        };

        struct Pipeline
        {
            std::shared_ptr<Library>     Native;
            const light_rhi_cpu::Kernel *Kernel;
            std::uint32_t                PushBytes;
        };

        template <typename Handle, typename T> class Slots
        {
          public:
            [[nodiscard]] Handle Add(T value)
            {
                if (!_free.empty())
                {
                    const auto index{_free.back()};
                    _values.at(index) = std::make_unique<T>(std::move(value));
                    _free.pop_back();
                    return {.Index = index};
                }
                const auto index{static_cast<std::uint32_t>(_values.size())};
                if (index == kInvalidIndex)
                {
                    FailContract("CPU resource slots exhausted");
                }
                _values.push_back(std::make_unique<T>(std::move(value)));
                return {.Index = index};
            }

            [[nodiscard]] T *Find(Handle handle) const
            {
                return handle.Index < _values.size() ? _values.at(handle.Index).get() : nullptr;
            }

            [[nodiscard]] T &Get(Handle handle) const
            {
                if (auto *value{Find(handle)})
                {
                    return *value;
                }
                FailContract("CPU command used a non-live resource");
            }

            void Remove(Handle handle)
            {
                static_cast<void>(Get(handle));
                _values.at(handle.Index).reset();
                _free.push_back(handle.Index);
            }

            [[nodiscard]] std::uint32_t Size() const noexcept
            {
                return static_cast<std::uint32_t>(_values.size() - _free.size());
            }

          private:
            std::vector<std::unique_ptr<T>> _values;
            std::vector<std::uint32_t>      _free;
        };

        [[nodiscard]] std::expected<Memory, PlacementError> Place(const Heap *heap, const HeapPlacement &placement,
                                                                  PlacementRequirements required, MemoryType memory)
        {
            if (!heap)
            {
                return std::unexpected{PlacementError::InvalidHeap};
            }
            if (heap->Type != memory)
            {
                return std::unexpected{PlacementError::IncompatibleMemory};
            }
            if (placement.Offset % required.Alignment)
            {
                return std::unexpected{PlacementError::MisalignedOffset};
            }
            if (placement.Offset > heap->Storage.Bytes.size() ||
                required.Size > heap->Storage.Bytes.size() - placement.Offset)
            {
                return std::unexpected{PlacementError::OutOfRange};
            }
            const auto  offset{placement.Offset};
            const auto  end{offset + required.Size};
            const auto &ranges{heap->Ranges};
            const auto  next{ranges->lower_bound(offset)};
            if ((next != ranges->end() && next->first < end) ||
                (next != ranges->begin() && std::prev(next)->second > offset))
            {
                return std::unexpected{PlacementError::Overlapping};
            }
            ranges->emplace_hint(next, offset, end);
            auto                       bytes{Slice(heap->Storage.Bytes, offset, required.Size)};
            std::shared_ptr<std::byte> lease{bytes.data(), [storage = heap->Storage.Owner, ranges, offset](std::byte *)
                                             { ranges->erase(offset); }};
            return Memory{std::move(lease), bytes};
        }

        [[nodiscard]] std::expected<PlacementRequirements, PlacementError> TextureRequirements(const TextureDesc &desc)
        {
            const auto  stride{TexelBytes(desc.Format)};
            const auto &size{desc.Extent};
            if (desc.Dimension != TextureDimension::Tex2D || size.Width == 0 || size.Height == 0 || size.Depth != 1 ||
                desc.MipLevels != 1 || desc.ArrayLayers != 1 || desc.SampleCount != 1 || stride == 0 ||
                size.Width > std::uint32_t(INT32_MAX / 2) || size.Height > std::uint32_t(INT32_MAX / 2))
            {
                return std::unexpected{PlacementError::CreationFailed};
            }
            if (HasUsage(desc.Usage, TextureUsage::RenderTarget) || HasUsage(desc.Usage, TextureUsage::DepthStencil) ||
                (HasUsage(desc.Usage, TextureUsage::Storage) && desc.Format != Format::R32Float &&
                 desc.Format != Format::RG32Float && desc.Format != Format::RGBA32Float))
            {
                return std::unexpected{PlacementError::CreationFailed};
            }
            const auto pixels{std::uint64_t(size.Width) * size.Height};
            return PlacementRequirements{.Size = pixels * stride, .Alignment = StorageAlignment};
        }

        [[nodiscard]] Memory MakeMemory(std::uint64_t bytes)
        {
            auto owner{Allocate(bytes)};
            if (!owner)
            {
                return {};
            }
            auto *ptr{owner.get()};
            return {std::move(owner), std::span{ptr, static_cast<std::size_t>(bytes)}};
        }

        /**
         * {brief} Threads that run the lanes of a workgroup, kept from one dispatch to the next so a dispatch
         * creates none. The calling thread runs lane 0.
         */
        class LanePool
        {
          public:
            LanePool()                            = default;
            LanePool(const LanePool &)            = delete;
            LanePool &operator=(const LanePool &) = delete;

            ~LanePool()
            {
                {
                    const std::scoped_lock lock{_mutex};
                    _stopping = true;
                    ++_generation;
                }
                _wake.notify_all();
            }

            /** {brief} Run `work` once for each lane in [0, lanes) at the same time, and return when all are done. */
            void Run(const std::uint32_t lanes, const std::function<void(std::uint32_t)> &work)
            {
                std::latch done{static_cast<std::ptrdiff_t>(lanes - 1U)};
                {
                    const std::scoped_lock lock{_mutex};
                    while (_threads.size() + 1U < lanes)
                    {
                        const auto lane{static_cast<std::uint32_t>(_threads.size() + 1U)};
                        _threads.emplace_back([this, lane, seen = _generation] { _serve(lane, seen); });
                    }
                    _work  = &work;
                    _lanes = lanes;
                    _done  = &done;
                    ++_generation;
                }
                _wake.notify_all();
                work(0U);
                done.wait();
            }

          private:
            void _serve(const std::uint32_t lane, std::uint64_t seen)
            {
                for (;;)
                {
                    std::unique_lock lock{_mutex};
                    _wake.wait(lock, [&] { return _generation != seen; });
                    if (_stopping)
                    {
                        return;
                    }
                    seen = _generation;
                    if (lane >= _lanes)
                    {
                        continue;
                    }
                    const auto *const work{_work};
                    auto *const       done{_done};
                    lock.unlock();
                    (*work)(lane);
                    done->count_down();
                }
            }

            std::mutex                                _mutex;
            std::condition_variable                   _wake;
            std::uint64_t                             _generation{};
            bool                                      _stopping{};
            const std::function<void(std::uint32_t)> *_work{};
            std::uint32_t                             _lanes{};
            std::latch                               *_done{};
            std::vector<std::jthread>                 _threads; ///< Last: joined before the state they read goes.
        };

        class Device final : public IDevice, private IBindlessHeap
        {
          public:
            std::string_view AdapterName() const noexcept override
            {
                return "LightRHI CPU";
            }
            std::uint64_t VideoMemoryBytes() const noexcept override
            {
                return 0;
            }
            IBindlessHeap &BindlessHeap() noexcept override
            {
                return *this;
            }

            std::expected<MemoryHeapHandle, DeviceError> CreateMemoryHeap(const MemoryHeapDesc &desc) override
            {
                auto storage{MakeMemory(desc.Size)};
                if (!storage.Owner)
                {
                    return std::unexpected{desc.Size == 0 ? DeviceError::InvalidArgument : DeviceError::Exhausted};
                }
                return _heaps.Add(
                    {std::move(storage), desc.MemoryType, std::make_shared<std::map<std::uint64_t, std::uint64_t>>()});
            }

            void DestroyMemoryHeap(MemoryHeapHandle handle) override
            {
                _heaps.Remove(handle);
            }

            std::expected<PlacementRequirements, PlacementError>
            BufferPlacementRequirements(const BufferDesc &desc) const override
            {
                if (desc.Size == 0)
                {
                    return std::unexpected{PlacementError::CreationFailed};
                }
                return PlacementRequirements{.Size = desc.Size, .Alignment = StorageAlignment};
            }

            std::expected<PlacementRequirements, PlacementError> TexturePlacementRequirements(const TextureDesc &desc,
                                                                                              MemoryType) const override
            {
                return TextureRequirements(desc);
            }

            std::expected<BufferHandle, DeviceError> CreateBuffer(const BufferDesc &desc) override
            {
                auto memory{MakeMemory(desc.Size)};
                if (!memory.Owner)
                {
                    return std::unexpected{desc.Size == 0 ? DeviceError::InvalidArgument : DeviceError::Exhausted};
                }
                return _addBuffer(desc, std::move(memory));
            }

            std::expected<BufferHandle, PlacementError> CreateBuffer(const BufferDesc    &desc,
                                                                     const HeapPlacement &placement) override
            {
                const auto requirements{BufferPlacementRequirements(desc)};
                if (!requirements)
                {
                    return std::unexpected{requirements.error()};
                }
                auto memory{Place(_heaps.Find(placement.Heap), placement, *requirements, desc.MemoryType)};
                return memory.transform([&](Memory &storage) { return _addBuffer(desc, std::move(storage)); });
            }

            void DestroyBuffer(BufferHandle handle) override
            {
                _buffers.Remove(handle);
            }

            GpuAddress BufferAddress(BufferHandle handle) const override
            {
                const auto *buffer{_buffers.Find(handle)};
                return buffer ? buffer->Info.DeviceAddress : GpuAddress{};
            }

            BufferInfo GetBufferInfo(BufferHandle handle) const override
            {
                const auto *buffer{_buffers.Find(handle)};
                return buffer ? buffer->Info : BufferInfo{};
            }

            MappedBuffer MapBuffer(BufferHandle handle) override
            {
                const auto *buffer{_buffers.Find(handle)};
                if (!buffer || buffer->Info.MemoryType == MemoryType::GpuOnly)
                {
                    return {};
                }
                return {.Data = buffer->Storage.Bytes.data(), .Size = buffer->Info.Size};
            }

            void UnmapBuffer(BufferHandle handle) override
            {
                static_cast<void>(_buffers.Get(handle));
            }

            std::expected<TextureHandle, DeviceError> CreateTexture(const TextureDesc &desc) override
            {
                const auto requirements{TextureRequirements(desc)};
                if (!requirements)
                {
                    return std::unexpected{DeviceError::Unsupported};
                }
                auto memory{MakeMemory(requirements->Size)};
                if (!memory.Owner)
                {
                    return std::unexpected{DeviceError::Exhausted};
                }
                return _addTexture(desc, std::move(memory));
            }

            std::expected<TextureHandle, PlacementError> CreateTexture(const TextureDesc   &desc,
                                                                       const HeapPlacement &placement) override
            {
                const auto requirements{TextureRequirements(desc)};
                if (!requirements)
                {
                    return std::unexpected{requirements.error()};
                }
                const auto *heap{_heaps.Find(placement.Heap)};
                auto        memory{Place(heap, placement, *requirements, heap ? heap->Type : MemoryType::GpuOnly)};
                return memory.transform([&](Memory &storage) { return _addTexture(desc, std::move(storage)); });
            }

            void DestroyTexture(TextureHandle handle) override
            {
                _textures.Remove(handle);
            }

            TextureInfo GetTextureInfo(TextureHandle handle) const override
            {
                const auto *image{_textures.Find(handle)};
                return image ? image->View.Info() : TextureInfo{};
            }

            GpuAddress TextureAddress(TextureHandle handle) const override
            {
                auto *image{_textures.Find(handle)};
                return {.Address = reinterpret_cast<std::uintptr_t>(image ? static_cast<IRWTexture *>(&image->View)
                                                                          : nullptr)};
            }

            std::expected<SamplerHandle, DeviceError> CreateSampler(const SamplerDesc &desc) override
            {
                if (desc.CompareEnable || desc.Anisotropy)
                {
                    return std::unexpected{DeviceError::Unsupported};
                }
                return _samplers.Add({desc});
            }

            void DestroySampler(SamplerHandle handle) override
            {
                _samplers.Remove(handle);
            }

            GpuAddress SamplerAddress(SamplerHandle handle) const override
            {
                return {.Address = reinterpret_cast<std::uintptr_t>(_samplers.Find(handle))};
            }

            std::expected<PipelineHandle, DeviceError> CreateComputePipeline(const ComputePipelineDesc &desc) override
            {
                const auto *bytecode{std::get_if<HostCallableBytecode>(&desc.Shader.Bytecode)};
                if (!bytecode)
                {
                    return std::unexpected{DeviceError::Unsupported};
                }
                if (bytecode->Bytes.empty() || desc.Shader.EntryPoint.empty() ||
                    desc.Shader.EntryPoint.find('\0') != std::string_view::npos)
                {
                    return std::unexpected{DeviceError::InvalidArgument};
                }
                // One open library for each library's contents, however many pipelines use it.
                const std::string_view contents{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): char may view any bytes
                    reinterpret_cast<const char *>(bytecode->Bytes.data()), bytecode->Bytes.size()};
                auto &cached{_libraries[std::hash<std::string_view>{}(contents)]};
                auto  native{cached.lock()};
                if (!native || !native->Holds(bytecode->Bytes))
                {
                    native = std::make_shared<Library>(bytecode->Bytes);
                    cached = native;
                }
                if (!native->OpenError().empty())
                {
                    ReportRefusal("CPU compute library could not be opened: " + native->OpenError());
                    return std::unexpected{DeviceError::InvalidArgument};
                }
                const auto  find{native->Find()};
                const auto *kernel{find ? find(std::string{desc.Shader.EntryPoint}.c_str()) : nullptr};
                if (!kernel)
                {
                    ReportRefusal("CPU compute entry point " + std::string{desc.Shader.EntryPoint} +
                                  " missing in its library");
                    return std::unexpected{DeviceError::InvalidArgument};
                }
                const auto [x, y, z]{kernel->Threads};
                const auto &[w, h, d]{desc.ThreadGroupSize};
                if (x == 0 || y == 0 || z == 0 || std::uint64_t(x) * y * z > MaxWorkgroupLanes || (w != 0 && w != x) ||
                    (h != 0 && h != y) || (d != 0 && d != z))
                {
                    return std::unexpected{DeviceError::InvalidArgument};
                }
                return _pipelines.Add({std::move(native), kernel, desc.PushConstantBytes});
            }

            void DestroyPipeline(PipelineHandle handle) override
            {
                _pipelines.Remove(handle);
            }

            std::expected<std::unique_ptr<ICommandList>, DeviceError> CreateCommandList(QueueType,
                                                                                        std::string_view) override;

            FenceHandle Submit(ICommandList &commands, const SubmitDesc &desc) override;

            /** {brief} Run `pipeline` over `groups` workgroups with `push` as its root constants. */
            void Run(PipelineHandle pipeline, Grid groups, std::span<std::byte> push);

            void WaitForFence(FenceHandle fence) override
            {
                if (fence.Valid() && !IsFenceComplete(fence))
                {
                    FailContract("CPU wait refers to a future fence");
                }
            }

            bool IsFenceComplete(FenceHandle fence) override
            {
                return fence.Valid() && fence.Id <= _timeline;
            }
            void WaitIdle() override {}

            void UploadBuffer(BufferHandle dst, std::span<const std::byte> data, std::uint64_t offset) override
            {
                std::ranges::copy(data, Slice(Bytes(dst), offset, data.size()).begin());
            }

            void UploadTexture(TextureHandle dst, std::span<const std::byte> data, std::uint64_t rowPitch,
                               std::uint64_t, const TextureCopyRegion &region) override
            {
                auto      &texture{GetTexture(dst)};
                const auto rowSize{texture.CopyRowBytes(region)};
                if (rowPitch < rowSize ||
                    (region.Extent.Height > 1 &&
                     rowPitch > std::numeric_limits<std::uint64_t>::max() / (region.Extent.Height - 1)))
                {
                    FailContract("invalid CPU texture upload pitch");
                }
                for (std::size_t row{}; row < region.Extent.Height; ++row)
                {
                    const auto offset{row * rowPitch};
                    if (offset > data.size() || rowSize > data.size() - offset)
                    {
                        FailContract("CPU texture upload exceeds source data");
                    }
                    std::ranges::copy(data.subspan(static_cast<std::size_t>(offset), rowSize),
                                      texture.Row(region, row).begin());
                }
            }

            void BeginCaptureScope(std::string_view) override {}
            void EndCaptureScope() override {}
            bool SupportsComputeTimestamps() const noexcept override
            {
                return false;
            }
            double TimestampPeriodNanoseconds() const noexcept override
            {
                return 0;
            }
            std::expected<TimestampQueryPoolHandle, DeviceError> CreateTimestampQueryPool(std::uint32_t) override
            {
                return std::unexpected{DeviceError::Unsupported};
            }
            void DestroyTimestampQueryPool(TimestampQueryPoolHandle) override {}
            void ResetTimestampQueries(TimestampQueryPoolHandle, std::uint32_t, std::uint32_t) override {}
            void ReadTimestampQueries(TimestampQueryPoolHandle, std::uint32_t, std::span<std::uint64_t>) override {}
            std::expected<PipelineHandle, DeviceError> CreateGraphicsPipeline(const GraphicsPipelineDesc &) override
            {
                return std::unexpected{DeviceError::Unsupported};
            }
            bool SupportsRayTracing() const noexcept override
            {
                return false;
            }
            AccelerationStructureBuildSizes
            QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &) const override
            {
                return {};
            }
            std::expected<AccelerationStructureHandle, DeviceError>
            CreateAccelerationStructure(const AccelerationStructureDesc &) override
            {
                return std::unexpected{DeviceError::Unsupported};
            }
            void       DestroyAccelerationStructure(AccelerationStructureHandle) override {}
            GpuAddress AccelerationStructureAddress(AccelerationStructureHandle) const override
            {
                return {};
            }

            [[nodiscard]] std::span<std::byte> Bytes(BufferHandle handle) const
            {
                return _buffers.Get(handle).Storage.Bytes;
            }
            [[nodiscard]] Texture &GetTexture(TextureHandle handle) const
            {
                return _textures.Get(handle).View;
            }
            [[nodiscard]] const Pipeline &GetPipeline(PipelineHandle handle) const
            {
                return _pipelines.Get(handle);
            }

          private:
            std::uint32_t MaxBuffers() const noexcept override
            {
                return kInvalidIndex;
            }
            std::uint32_t MaxTextures() const noexcept override
            {
                return kInvalidIndex;
            }
            std::uint32_t MaxSamplers() const noexcept override
            {
                return kInvalidIndex;
            }
            std::uint32_t UsedBuffers() const noexcept override
            {
                return _buffers.Size();
            }
            std::uint32_t UsedTextures() const noexcept override
            {
                return _textures.Size();
            }
            std::uint32_t UsedSamplers() const noexcept override
            {
                return _samplers.Size();
            }
            GpuAddress HeapAddress() const noexcept override
            {
                return {};
            }

            [[nodiscard]] BufferHandle _addBuffer(const BufferDesc &desc, Memory storage)
            {
                const GpuAddress address{.Address = HasUsage(desc.Usage, BufferUsage::DeviceAddress)
                                                        ? reinterpret_cast<std::uintptr_t>(storage.Bytes.data())
                                                        : 0};
                return _buffers.Add({std::move(storage),
                                     {.Size          = desc.Size,
                                      .Usage         = desc.Usage,
                                      .MemoryType    = desc.MemoryType,
                                      .DeviceAddress = address}});
            }

            [[nodiscard]] TextureHandle _addTexture(const TextureDesc &desc, Memory storage)
            {
                const auto bytes{storage.Bytes};
                return _textures.Add({std::move(storage), Texture{desc, bytes}});
            }

            Slots<MemoryHeapHandle, Heap>   _heaps;
            Slots<BufferHandle, Buffer>     _buffers;
            Slots<TextureHandle, Image>     _textures;
            Slots<SamplerHandle, Sampler>   _samplers;
            Slots<PipelineHandle, Pipeline> _pipelines;
            /// The open libraries, by a hash of their contents.
            std::unordered_map<std::size_t, std::weak_ptr<Library>> _libraries;
            std::uint64_t                                           _timeline{};
            LanePool                                                _lanes; ///< Kept across dispatches.
        };

        class CommandList final : public ICommandList
        {
          public:
            explicit CommandList(Device &device) : _device{device} {}

            void Begin() override;
            void End() override;
            void SetPipeline(PipelineHandle pipeline) override;
            void SetPushConstants(std::span<const std::byte> data, std::uint32_t offset) override;
            void Dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z) override;
            void DispatchIndirect(BufferHandle buffer, std::uint64_t offset) override;
            void CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &region) override;
            void CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &region) override;
            void CopyBufferToTexture(BufferHandle src, std::uint64_t offset, TextureHandle dst,
                                     const TextureCopyRegion &region) override;
            void CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &region, BufferHandle dst,
                                     std::uint64_t offset) override;
            void ClearTexture(TextureHandle texture, const ClearColor &color, const SubresourceRange &range) override;
            void FillBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size,
                            std::uint32_t value) override;

            // Commands and dispatches finish in order. No device cache needs flushing.
            void Transition(const TextureBarrier &) override
            {
                _requireRecording();
            }
            void Transition(const BufferBarrier &) override
            {
                _requireRecording();
            }
            void Transition(const MemoryBarrier &) override
            {
                _requireRecording();
            }
            void FlushBarriers() override
            {
                _requireRecording();
            }

            void BeginDebugGroup(std::string_view, float, float, float) override {}
            void EndDebugGroup() override {}
            void InsertDebugLabel(std::string_view) override {}
            void DebugExposeBuffer(BufferHandle, std::uint32_t) override {}
            void DebugExposeTexture(TextureHandle, std::uint32_t) override {}
            void DebugExposeAccelerationStructure(AccelerationStructureHandle, std::uint32_t) override {}

            void Present(ExternalTextureProviderHandle) override
            {
                _unsupported();
            }
            void Transition(const AccelerationStructureBarrier &) override
            {
                _unsupported();
            }
            void BeginRendering(const RenderingDesc &) override
            {
                _unsupported();
            }
            void EndRendering() override
            {
                _unsupported();
            }
            void SetViewport(const Viewport &) override
            {
                _unsupported();
            }
            void SetScissor(const Scissor &) override
            {
                _unsupported();
            }
            void BindTexture(TextureHandle, std::uint32_t) override
            {
                _unsupported();
            }
            void BindSampler(SamplerHandle, std::uint32_t) override
            {
                _unsupported();
            }
            void BindIndexBuffer(BufferHandle, std::uint64_t, IndexType) override
            {
                _unsupported();
            }
            void Draw(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) override
            {
                _unsupported();
            }
            void DrawIndexed(std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::uint32_t) override
            {
                _unsupported();
            }
            void DrawIndirect(BufferHandle, std::uint64_t, std::uint32_t, std::uint32_t) override
            {
                _unsupported();
            }
            void DrawIndexedIndirect(BufferHandle, std::uint64_t, std::uint32_t, std::uint32_t) override
            {
                _unsupported();
            }
            void DrawIndirectCount(BufferHandle, std::uint64_t, BufferHandle, std::uint64_t, std::uint32_t,
                                   std::uint32_t) override
            {
                _unsupported();
            }
            void WriteComputeTimestamp(TimestampQueryPoolHandle, std::uint32_t) override
            {
                _unsupported();
            }
            void BuildAccelerationStructure(AccelerationStructureHandle, const AccelerationStructureDesc &,
                                            BufferHandle, std::uint64_t) override
            {
                _unsupported();
            }
            void ClearDepthTexture(TextureHandle, const ClearDepthStencil &, const SubresourceRange &) override
            {
                _unsupported();
            }

            void Execute(Device &owner);

          private:
            [[noreturn]] static void _unsupported()
            {
                FailContract("command is not supported by the CPU backend");
            }
            void _requireRecording() const;
            void _record(std::function<void()> operation);

            Device                            &_device;
            std::vector<std::function<void()>> _commands;
            PipelineHandle                     _pipeline;
            std::vector<std::byte>             _push;
            bool                               _recording{};
            bool                               _ready{};
        };

        std::expected<std::unique_ptr<ICommandList>, DeviceError> Device::CreateCommandList(QueueType, std::string_view)
        {
            return std::make_unique<CommandList>(*this);
        }

        FenceHandle Device::Submit(ICommandList &commands, const SubmitDesc &desc)
        {
            if (desc.WaitFence.Valid() && (desc.WaitValue == 0 ? desc.WaitFence.Id : desc.WaitValue) > _timeline)
            {
                FailContract("CPU submission waits for a future fence");
            }
            // Submit takes only command lists issued by this device.
            static_cast<CommandList &>(commands).Execute(*this);
            return {.Id = ++_timeline};
        }

        void Device::Run(const PipelineHandle pipeline, const Grid groups, const std::span<std::byte> push)
        {
            if (groups.X == 0 || groups.Y == 0 || groups.Z == 0)
            {
                return;
            }
            const auto &kernel{*GetPipeline(pipeline).Kernel};
            if (!kernel.RunLane)
            {
                kernel.Run(groups, push.data());
                return;
            }

            const auto threads{kernel.Threads};
            const auto lanes{threads.X * threads.Y * threads.Z};
            // The kernel's own group barriers and the barrier that ends a group are kept apart, so a
            // phase of one is never taken for a phase of the other.
            std::barrier sync{static_cast<std::ptrdiff_t>(lanes)};
            std::barrier groupDone{static_cast<std::ptrdiff_t>(lanes)};
            alignas(std::max_align_t) std::array<std::byte, light_rhi_cpu::ScratchBytes> scratch{};
            const light_rhi_cpu::Workgroup                                               context{
                .Barrier = &sync,
                .Arrive  = [](void *data) { static_cast<decltype(sync) *>(data)->arrive_and_wait(); },
                .Scratch = scratch.data(),
            };
            _lanes.Run(lanes,
                       [&](const std::uint32_t lane)
                       {
                           const Grid local{lane % threads.X, lane / threads.X % threads.Y,
                                            lane / (threads.X * threads.Y)};
                           for (std::uint32_t z{}; z < groups.Z; ++z)
                           {
                               for (std::uint32_t y{}; y < groups.Y; ++y)
                               {
                                   for (std::uint32_t x{}; x < groups.X; ++x)
                                   {
                                       kernel.RunLane({x, y, z}, local, push.data(), &context);
                                       // The next workgroup reuses scratch only after every lane finishes.
                                       groupDone.arrive_and_wait();
                                   }
                               }
                           }
                       });
        }

        void CommandList::Begin()
        {
            if (_recording)
            {
                FailContract("CPU command list is already recording");
            }
            _commands.clear();
            _pipeline = {};
            _push.clear();
            _ready     = false;
            _recording = true;
        }

        void CommandList::End()
        {
            _requireRecording();
            _recording = false;
            _ready     = true;
        }

        void CommandList::Execute(Device &owner)
        {
            if (&owner != &_device || !_ready)
            {
                FailContract("CPU submission requires an ended list from the same device");
            }
            for (auto &command : _commands)
            {
                command();
            }
            _ready = false;
        }

        void CommandList::_requireRecording() const
        {
            if (!_recording)
            {
                FailContract("CPU command list is not recording");
            }
        }

        void CommandList::_record(std::function<void()> operation)
        {
            _requireRecording();
            _commands.push_back(std::move(operation));
        }

        void CommandList::SetPipeline(PipelineHandle pipeline)
        {
            _requireRecording();
            _pipeline = pipeline;
            const auto &program{_device.GetPipeline(pipeline)};
            // Binding a pipeline keeps root data already recorded on this list.
            const auto bytes{std::max(program.PushBytes, program.Kernel->PushConstantBytes)};
            if (_push.size() < bytes)
            {
                _push.resize(bytes);
            }
        }

        void CommandList::SetPushConstants(std::span<const std::byte> data, std::uint32_t offset)
        {
            _requireRecording();
            auto range{Slice(std::span{_push}.first(_device.GetPipeline(_pipeline).PushBytes), offset, data.size())};
            std::ranges::copy(data, range.begin());
        }

        void CommandList::Dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z)
        {
            _record([this, pipeline = _pipeline, push = _push, groups = Grid{x, y, z}]() mutable
                    { _device.Run(pipeline, groups, push); });
        }

        void CommandList::DispatchIndirect(BufferHandle buffer, std::uint64_t offset)
        {
            _record(
                [this, pipeline = _pipeline, push = _push, buffer, offset]() mutable
                {
                    DispatchIndirectArgs groups{};
                    const auto           bytes{Slice(_device.Bytes(buffer), offset, sizeof(groups))};
                    std::memcpy(&groups, bytes.data(), bytes.size());
                    _device.Run(pipeline, {groups.X, groups.Y, groups.Z}, push);
                });
        }

        void CommandList::CopyBuffer(BufferHandle src, BufferHandle dst, const BufferCopyRegion &region)
        {
            _record(
                [this, src, dst, region]
                {
                    const auto from{Slice(_device.Bytes(src), region.SrcOffset, region.Size)};
                    const auto to{Slice(_device.Bytes(dst), region.DstOffset, region.Size)};
                    if (!from.empty())
                    {
                        std::memmove(to.data(), from.data(), from.size());
                    }
                });
        }

        void CommandList::CopyTexture(TextureHandle src, TextureHandle dst, const TextureCopyRegion &region)
        {
            _record(
                [this, src, dst, region]
                {
                    auto &from{_device.GetTexture(src)};
                    auto &to{_device.GetTexture(dst)};
                    auto  source{region};
                    source.DstOffset = {};
                    if (from.Info().Format != to.Info().Format || from.CopyRowBytes(source) != to.CopyRowBytes(region))
                    {
                        FailContract("CPU texture copy requires matching formats");
                    }
                    for (std::size_t row{}; row < region.Extent.Height; ++row)
                    {
                        const auto bytes{from.Row(source, row)};
                        std::memmove(to.Row(region, row).data(), bytes.data(), bytes.size());
                    }
                });
        }

        void CommandList::CopyBufferToTexture(BufferHandle src, std::uint64_t offset, TextureHandle dst,
                                              const TextureCopyRegion &region)
        {
            _record(
                [this, src, offset, dst, region]
                {
                    const auto rowBytes{_device.GetTexture(dst).CopyRowBytes(region)};
                    const auto bytes{Slice(_device.Bytes(src), offset, rowBytes * region.Extent.Height)};
                    _device.UploadTexture(dst, bytes, rowBytes, bytes.size(), region);
                });
        }

        void CommandList::CopyTextureToBuffer(TextureHandle src, const TextureCopyRegion &region, BufferHandle dst,
                                              std::uint64_t offset)
        {
            _record(
                [this, src, region, dst, offset]
                {
                    auto      &texture{_device.GetTexture(src)};
                    const auto rowBytes{texture.CopyRowBytes(region)};
                    const auto bytes{Slice(_device.Bytes(dst), offset, rowBytes * region.Extent.Height)};
                    for (std::size_t row{}; row < region.Extent.Height; ++row)
                    {
                        std::ranges::copy(texture.Row(region, row), bytes.subspan(row * rowBytes, rowBytes).begin());
                    }
                });
        }

        void CommandList::ClearTexture(TextureHandle texture, const ClearColor &color, const SubresourceRange &range)
        {
            _record(
                [this, texture, color, range]
                {
                    if (range.BaseMip != 0 || range.BaseLayer != 0 || (range.MipCount != 1 && range.MipCount != ~0U) ||
                        (range.LayerCount != 1 && range.LayerCount != ~0U))
                    {
                        FailContract("CPU clear is outside the texture");
                    }
                    _device.GetTexture(texture).Clear(color);
                });
        }

        void CommandList::FillBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size, std::uint32_t value)
        {
            _record(
                [this, buffer, offset, size, value]
                {
                    const auto bytes{Slice(_device.Bytes(buffer), offset, size)};
                    if (offset % sizeof(value) || bytes.size() % sizeof(value))
                    {
                        FailContract("CPU buffer fill requires a whole number of aligned words");
                    }
                    for (std::size_t at{}; at < bytes.size(); at += sizeof(value))
                    {
                        std::memcpy(bytes.subspan(at, sizeof(value)).data(), &value, sizeof(value));
                    }
                });
        }
    } // namespace

    std::expected<std::unique_ptr<IDevice>, DeviceError> CreateDevice(const DeviceDesc &)
    {
        return std::make_unique<Device>();
    }
} // namespace rhi::cpu
