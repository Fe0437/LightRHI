/**
 * {file} sharedDevice.cppm
 * {brief} Shares one device across a process and serializes the threads that use it.
 */
module;
#include <expected>
#include <memory>
#include <mutex>
#include <utility>

export module rhi:sharedDevice;

import :descriptors;
import :diagnostics;
import :device;

export namespace rhi
{
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
            StrictLockPtr(IDevice &device, std::mutex &mutex) : _device{&device}, _lock{mutex} {}

            /** {brief} Returns the locked device. */
            [[nodiscard]] IDevice &operator*() const noexcept
            {
                return *_device;
            }
            /** {brief} Provides member access to the locked device. */
            [[nodiscard]] IDevice *operator->() const noexcept
            {
                return _device;
            }

          private:
            IDevice                     *_device; ///< Never null: made from a reference.
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

        if (SharedDevice device{weakDevice.lock()})
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
