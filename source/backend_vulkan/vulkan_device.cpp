/**
 * {file} vulkan_device.cpp
 * {brief} Implements Vulkan device, resource, presentation, and bindless lifecycles.
 */

module;
// vulkan_internal.h is included after the module declaration, so every standard header it
// needs must already be visible here; including one inside the module purview instead would
// re-declare what this fragment brought in.
#include "vulkan_platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

module lightRHI;
import rhi;

#include "vulkan_internal.h"

namespace rhi::vulkan
{
    namespace
    {
        [[nodiscard]] uint32_t uniqueQueueFamilies(std::array<uint32_t, 3> &families)
        {
            std::ranges::sort(families);
            return static_cast<uint32_t>(std::ranges::unique(families).begin() - families.begin());
        }
    } // namespace

    void VulkanDevice::BeginCaptureScope(std::string_view /*name*/) {}

    void VulkanDevice::EndCaptureScope() {}

    // ============================================================================
    // Debug callback
    // ============================================================================

    static VKAPI_ATTR VkBool32 VKAPI_CALL _debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                                         const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                         void * /*user*/)
    {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            std::fprintf(stderr, "[LightRHI::Vulkan] %s\n", data->pMessage);
        }
        return VK_FALSE;
    }

    // ============================================================================
    // VulkanDevice constructor / destructor
    // ============================================================================

    // volk dispatches every vk* call through process-global function pointers
    // that volkLoadInstance() binds to one instance. A second live device would
    // rebind them to its own instance, silently redirecting the first device's
    // calls (and, if the two disagree on validation, through a different layer
    // chain). Until the backend moves to a per-device volk dispatch table, fail
    // loudly rather than corrupt dispatch behind the caller's back.
    static std::atomic<int> g_liveDevices{0};

    LiveDeviceSlot::LiveDeviceSlot()
    {
        if (g_liveDevices.fetch_add(1, std::memory_order_acq_rel) != 0)
        {
            g_liveDevices.fetch_sub(1, std::memory_order_acq_rel);
            FailContract("only one Vulkan device may exist at a time "
                         "(volk uses a process-global dispatch table)");
        }
    }

    LiveDeviceSlot::~LiveDeviceSlot()
    {
        g_liveDevices.fetch_sub(1, std::memory_order_acq_rel);
    }

    bool VulkanDevice::_createRequiredDeviceObjects(const DeviceDesc &desc)
    {
        assert(_instance == VK_NULL_HANDLE);
        assert(_physDev == VK_NULL_HANDLE);
        assert(_device == VK_NULL_HANDLE);

        if (!_createInstance(desc))
        {
            return false;
        }

        if (!_pickPhysicalDevice())
        {
            return false;
        }

        // Every step from here on either succeeds or stops the process, so a device that has a
        // logical device has all of its device-level objects. The destructor relies on that.
        _createLogicalDevice();

        _loadExtensionFunctions();
        _createAllocator();
        _createCommandPools();
        _heap.Init(_device, _allocator, descBufProps, pfn_GetDescriptorEXT, pfn_GetLayoutSize, pfn_GetBindingOffset);
        _createGlobalLayout(); // needs _heap.DescriptorSetLayout()
        _createTimeline();
        return true;
    }

    std::expected<std::unique_ptr<VulkanDevice>, DeviceError> VulkanDevice::Create(const DeviceDesc &desc)
    {
        // Creation stops at the first refusal. The partly built device never leaves this function:
        // its destructor releases what was created, and the reason was reported where it was found.
        auto device{std::make_unique<VulkanDevice>(ConstructionToken{})};
        if (!device->_createRequiredDeviceObjects(desc))
        {
            return std::unexpected{DeviceError::NoDevice};
        }
        return device;
    }

    VulkanDevice::VulkanDevice(ConstructionToken) {}

    VulkanDevice::~VulkanDevice()
    {
        // Only Create() can see a device that stopped part way, and it stops before the logical
        // device or not at all, so the device-level objects exist together or not at all. Those two
        // checks stay because no vk* entry point may be called through a null device or instance.
        // The swapchain and surface are optional inside them: their entry points are loaded only
        // when presentation was requested.
        if (_device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(_device);

            _heap.Destroy(_device);
            vkDestroySemaphore(_device, _timeline, nullptr);
            vkDestroyPipelineLayout(_device, _globalLayout, nullptr);
            vkDestroyCommandPool(_device, gfxPool, nullptr);
            vkDestroyCommandPool(_device, computePool, nullptr);
            vkDestroyCommandPool(_device, xferPool, nullptr);
            vmaDestroyAllocator(_allocator);
            vkDestroyDevice(_device, nullptr);
        }

        if (_instance != VK_NULL_HANDLE)
        {
            if (_debugMessenger != VK_NULL_HANDLE)
            {
                // Loaded by name because the messenger's extension is optional; a messenger exists
                // only if the create entry point did, and the destroy one ships beside it.
                auto fn{(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(_instance,
                                                                                   "vkDestroyDebugUtilsMessengerEXT")};
                if (fn)
                {
                    fn(_instance, _debugMessenger, nullptr);
                }
            }
            vkDestroyInstance(_instance, nullptr);
        }
    }

    VulkanDevice::QueueInfo &VulkanDevice::QueueFor(QueueType type) noexcept
    {
        switch (type)
        {
            case QueueType::Compute:
                return computeQ;
            case QueueType::Transfer:
                return transferQ;
            case QueueType::Graphics:
                break;
        }
        return gfxQ;
    }

    VulkanDevice::CommandPool VulkanDevice::CommandPoolFor(QueueType type) noexcept
    {
        switch (type)
        {
            case QueueType::Compute:
                return CommandPool{.pool = computePool, .mutex = computePoolMtx};
            case QueueType::Transfer:
                return CommandPool{.pool = xferPool, .mutex = xferPoolMtx};
            case QueueType::Graphics:
                break;
        }
        return CommandPool{.pool = gfxPool, .mutex = gfxPoolMtx};
    }

} // namespace rhi::vulkan

// ---- Factory (declared in vulkan_backend.cppm / lightRHI module) -----------

namespace rhi
{
    ExternalTextureOwnerContext ExternalTextureOwnerContextOf(IDevice &device)
    {
        // The instance a caller's toolkit needs to make a surface from. It travels as a plain
        // pointer, which is what a dispatchable Vulkan handle already is.
        return {.Value = static_cast<vulkan::VulkanDevice &>(device).NativeInstance()};
    }

    std::expected<std::unique_ptr<IExternalTextureProvider>, DeviceError>
    CreateExternalTextureProvider(IDevice &device, ExternalTextureOwner owner, Format requestedFormat,
                                  PresentTiming timing)
    {
        // The one cast, written where the type is known: this build is the Vulkan one, so the
        // platform object a caller can have obtained is a VkSurfaceKHR and nothing else.
        static_assert(sizeof(VkSurfaceKHR) == sizeof(void *),
                      "a Vulkan surface handle has to be pointer-sized to travel as one");
        const auto       surface{reinterpret_cast<VkSurfaceKHR>(owner.Value)};
        auto            &vulkanDevice{static_cast<vulkan::VulkanDevice &>(device)};
        const VkInstance instance{vulkanDevice.NativeInstance()};
        if (surface == VK_NULL_HANDLE)
        {
            ReportRefusal("a Vulkan external texture provider needs a surface");
            return std::unexpected{DeviceError::InvalidArgument};
        }

        // A surface this device's queue cannot present to would fail on every frame instead of here.
        VkBool32 supported{VK_FALSE};
        vkGetPhysicalDeviceSurfaceSupportKHR(vulkanDevice.NativePhysicalDevice(), vulkanDevice.GraphicsQueueFamily(),
                                             surface, &supported);
        if (supported != VK_TRUE)
        {
            vkDestroySurfaceKHR(instance, surface, nullptr);
            ReportRefusal("this device's queue cannot present to that surface");
            return std::unexpected{DeviceError::Unsupported};
        }

        return std::make_unique<vulkan::VulkanPresentingTextureProvider>(vulkanDevice, instance, surface,
                                                                         requestedFormat, timing);
    }

    std::expected<SharedDevice, DeviceError> AcquireSharedDevice(const DeviceDesc &desc)
    {
        return rhi::AcquireSharedDevice(desc, &CreateDevice);
    }
} // namespace rhi

namespace rhi::vulkan
{

    // ============================================================================
    // Init helpers
    // ============================================================================

    // Every platform surface extension this build can use. The loader says which of them this
    // system actually has; see _createInstance.
#if defined(LIGHT_RHI_VULKAN_HAS_WIN32_SURFACE) || defined(LIGHT_RHI_VULKAN_HAS_WAYLAND_SURFACE) ||                    \
    defined(LIGHT_RHI_VULKAN_HAS_XCB_SURFACE) || defined(LIGHT_RHI_VULKAN_HAS_XLIB_SURFACE)
    constexpr std::array kPlatformSurfaceExtensions{
#if defined(LIGHT_RHI_VULKAN_HAS_WIN32_SURFACE)
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif
#if defined(LIGHT_RHI_VULKAN_HAS_WAYLAND_SURFACE)
        VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#endif
#if defined(LIGHT_RHI_VULKAN_HAS_XCB_SURFACE)
        VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif
#if defined(LIGHT_RHI_VULKAN_HAS_XLIB_SURFACE)
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
    };
#else
    // A build with no window-system integration still compiles; it simply has none to enable.
    constexpr std::array<const char *, 0> kPlatformSurfaceExtensions{};
#endif

    bool VulkanDevice::_createInstance(const DeviceDesc &desc)
    {
        // Load the driver's Vulkan loader through volk before any vk* call.
        // Idempotent, so repeated device creation is safe.
        if (volkInitialize() != VK_SUCCESS)
        {
            ReportRefusal("no Vulkan loader on this system");
            return false;
        }

        VkApplicationInfo ai{
            .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName   = desc.AppName == nullptr || *desc.AppName == '\0' ? "LightRHI" : desc.AppName,
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "LightRHI",
            .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion         = VK_API_VERSION_1_3,
        };

        std::vector<const char *> exts{};
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        // Surface extensions are enabled when the loader has them, not when a surface is at hand:
        // a device is created before anything is known about windows, and a system with no window
        // server simply reports none of these, so a target created later is refused with a reason.
        uint32_t availableCount{0};
        vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, nullptr);
        std::vector<VkExtensionProperties> available(availableCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, available.data());
        const auto hasExtension{
            [&available](const char *name)
            {
                return std::ranges::any_of(
                    available, [name](const VkExtensionProperties &extension)
                    { return std::string_view{extension.extensionName} == std::string_view{name}; });
            }};
        if (hasExtension(VK_KHR_SURFACE_EXTENSION_NAME))
        {
            exts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
            for (const char *platformExtension : kPlatformSurfaceExtensions)
            {
                if (hasExtension(platformExtension))
                {
                    exts.push_back(platformExtension);
                }
            }
            // A surface with no window behind it, which is how presentation is exercised where
            // there is no window server to give one - the counterpart to a layer that belongs to
            // no window on Apple.
            if (hasExtension(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME))
            {
                exts.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
            }
        }

        std::vector<const char *> layers{};
        if (desc.EnableValidation)
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            exts.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME); // for sync validation below
        }

        VkDebugUtilsMessengerCreateInfoEXT dbgCI{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = _debugCallback,
        };

        // Synchronization validation. Plain validation only checks API usage, so
        // it cannot see a *missing* barrier — the exact class of bug that hides
        // in a hand-rolled barrier layer (a race reads stale data and still
        // "passes" on any GPU that happens to serialize the work). Sync
        // validation models the hazards explicitly and reports them
        // deterministically, so turn it on wherever validation is on.
        constexpr VkValidationFeatureEnableEXT kSyncValidation[]{
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
        VkValidationFeaturesEXT valFeatures{
            .sType                         = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
            .pNext                         = &dbgCI,
            .enabledValidationFeatureCount = static_cast<uint32_t>(std::size(kSyncValidation)),
            .pEnabledValidationFeatures    = kSyncValidation,
        };

        VkInstanceCreateInfo ci{
            .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext                   = desc.EnableValidation ? static_cast<const void *>(&valFeatures) : nullptr,
            .pApplicationInfo        = &ai,
            .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames     = layers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(exts.size()),
            .ppEnabledExtensionNames = exts.data(),
        };
        VK_CHECK(vkCreateInstance(&ci, nullptr, &_instance));

        // Load instance- and device-level entry points through the new instance.
        volkLoadInstance(_instance);

        if (desc.EnableValidation)
        {
            auto fn{
                (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(_instance, "vkCreateDebugUtilsMessengerEXT")};
            if (fn)
            {
                fn(_instance, &dbgCI, nullptr, &_debugMessenger);
            }
        }
        return true;
    }

    bool VulkanDevice::_pickPhysicalDevice()
    {
        assert(_instance != VK_NULL_HANDLE);

        uint32_t count{0};
        VK_CHECK(vkEnumeratePhysicalDevices(_instance, &count, nullptr));
        if (!count)
        {
            ReportRefusal("no Vulkan physical device on this system");
            return false;
        }
        std::vector<VkPhysicalDevice> devs(count);
        VK_CHECK(vkEnumeratePhysicalDevices(_instance, &count, devs.data()));

        VkPhysicalDevice best{VK_NULL_HANDLE};
        int              bestScore{-1};

        for (auto pd : devs)
        {
            // API version
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.apiVersion < VK_API_VERSION_1_3)
            {
                continue;
            }

            // Must support VK_EXT_descriptor_buffer
            uint32_t ec{0};
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, nullptr);
            std::vector<VkExtensionProperties> ep(ec);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, ep.data());
            bool hasDB{false};
            for (const auto &e : ep)
            {
                if (std::string_view{e.extensionName} == VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)
                {
                    hasDB = true;
                    break;
                }
            }
            if (!hasDB)
            {
                continue;
            }

            // Check required features
            VkPhysicalDeviceDescriptorBufferFeaturesEXT dbf{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
            VkPhysicalDeviceVulkan13Features f13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                                 .pNext = &dbf};
            VkPhysicalDeviceVulkan12Features f12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                 .pNext = &f13};
            VkPhysicalDeviceFeatures2        f2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
            vkGetPhysicalDeviceFeatures2(pd, &f2);
            if (!f13.dynamicRendering || !f13.synchronization2 || !f12.bufferDeviceAddress || !f12.timelineSemaphore ||
                !f12.descriptorIndexing || !f12.runtimeDescriptorArray || !f12.descriptorBindingPartiallyBound ||
                !f12.scalarBlockLayout || !dbf.descriptorBuffer)
            {
                continue;
            }

            int score{0};
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                score = 1000;
            }
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            {
                score = 100;
            }
            if (score > bestScore)
            {
                bestScore = score;
                best      = pd;
            }
        }

        if (!best)
        {
            ReportRefusal("no compatible GPU (needs Vulkan 1.3, descriptor indexing, and "
                          "VK_EXT_descriptor_buffer)");
            return false;
        }
        _physDev = best;

        // Cache adapter name and VRAM
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(_physDev, &props);
            _adapterName = props.deviceName;
        }
        {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(_physDev, &mp);
            for (uint32_t i{0}; i < mp.memoryHeapCount; ++i)
            {
                if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                {
                    _videoMemoryBytes += mp.memoryHeaps[i].size;
                }
            }
        }

        // Descriptor buffer properties
        descBufProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &descBufProps};
        vkGetPhysicalDeviceProperties2(_physDev, &p2);
        _maxPushConstantBytes       = p2.properties.limits.maxPushConstantsSize;
        _timestampPeriodNanoseconds = static_cast<double>(p2.properties.limits.timestampPeriod);

        // Ray tracing (VK_KHR_acceleration_structure + VK_KHR_ray_query) —
        // optional, unlike VK_EXT_descriptor_buffer above: detected here but
        // never gates device selection, so a driver/device lacking it still
        // produces a working (non-ray-tracing) VulkanDevice. Callers must
        // check IDevice::SupportsRayTracing() before touching AS methods.
        {
            uint32_t ec{0};
            vkEnumerateDeviceExtensionProperties(_physDev, nullptr, &ec, nullptr);
            std::vector<VkExtensionProperties> ep(ec);
            vkEnumerateDeviceExtensionProperties(_physDev, nullptr, &ec, ep.data());
            auto has{[&](const char *name)
                     {
                         for (const auto &e : ep)
                         {
                             if (std::string_view{e.extensionName} == name)
                             {
                                 return true;
                             }
                         }
                         return false;
                     }};
            // Ray query alone would cover the inline-tracing we expose, but Slang
            // compiles a RayQuery against a bindless AS handle down to SPIR-V that
            // declares SPV_KHR_ray_tracing (for OpConvertUToAccelerationStructureKHR),
            // and that extension is only legal on a device with
            // VK_KHR_ray_tracing_pipeline (VUID-VkShaderModuleCreateInfo-pCode-08742).
            const bool extsPresent{
                has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && has(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) && has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)};
            if (extsPresent)
            {
                VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtp{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
                VkPhysicalDeviceRayQueryFeaturesKHR rq{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, .pNext = &rtp};
                VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, .pNext = &rq};
                VkPhysicalDeviceFeatures2 rtF2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &asf};
                vkGetPhysicalDeviceFeatures2(_physDev, &rtF2);
                _raytracingSupported = asf.accelerationStructure && rq.rayQuery && rtp.rayTracingPipeline;
            }
        }

        // Find queue families
        uint32_t qfc{0};
        vkGetPhysicalDeviceQueueFamilyProperties(_physDev, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qfc);
        vkGetPhysicalDeviceQueueFamilyProperties(_physDev, &qfc, qfp.data());

        for (uint32_t i{0}; i < qfc; ++i)
        {
            const auto &f{qfp[i]};
            if ((f.queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfxQ.family == ~0u)
            {
                gfxQ.family = i;
            }
            // Prefer a dedicated async-compute family (no graphics bit)
            if ((f.queueFlags & VK_QUEUE_COMPUTE_BIT) && !(f.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                computeQ.family == ~0u)
            {
                computeQ.family = i;
            }
            // Prefer a dedicated transfer family
            if ((f.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(f.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                !(f.queueFlags & VK_QUEUE_COMPUTE_BIT) && transferQ.family == ~0u)
            {
                transferQ.family = i;
            }
        }
        if (gfxQ.family == ~0u)
        {
            ReportRefusal("no graphics queue family on this device");
            return false;
        }
        if (computeQ.family == ~0u)
        {
            computeQ.family = gfxQ.family;
        }
        if (transferQ.family == ~0u)
        {
            transferQ.family = gfxQ.family;
        }

        VkPhysicalDeviceVulkan12Features timestampFeatures{.sType =
                                                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2        timestampFeatures2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                                            .pNext = &timestampFeatures};
        vkGetPhysicalDeviceFeatures2(_physDev, &timestampFeatures2);
        const bool hostResetSupported{timestampFeatures.hostQueryReset == VK_TRUE};

        // timestampComputeAndGraphics is a device limit, not a feature: it reports whether every
        // graphics and compute queue family supports timestamps.
        VkPhysicalDeviceProperties deviceProperties{};
        vkGetPhysicalDeviceProperties(_physDev, &deviceProperties);
        const bool computeAndGraphicsSupported{deviceProperties.limits.timestampComputeAndGraphics == VK_TRUE};
        _computeTimestampsSupported =
            hostResetSupported && computeAndGraphicsSupported && qfp[computeQ.family].timestampValidBits > 0;
        return true;
    }

    void VulkanDevice::_createLogicalDevice()
    {
        assert(_physDev != VK_NULL_HANDLE);
        assert(gfxQ.family != ~0u);
        assert(computeQ.family != ~0u);
        assert(transferQ.family != ~0u);

        // Collect unique families
        std::vector<uint32_t> families{};
        for (uint32_t f : {gfxQ.family, computeQ.family, transferQ.family})
        {
            if (std::find(families.begin(), families.end(), f) == families.end())
            {
                families.push_back(f);
            }
        }
        float                                prio{1.f};
        std::vector<VkDeviceQueueCreateInfo> qcis{};
        for (uint32_t f : families)
        {
            qcis.push_back({.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                            .queueFamilyIndex = f,
                            .queueCount       = 1,
                            .pQueuePriorities = &prio});
        }

        // Feature chain (tail → head for pNext)
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtp{
            .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            .rayTracingPipeline = VK_TRUE,
        };
        VkPhysicalDeviceRayQueryFeaturesKHR rq{
            .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            .pNext    = &rtp,
            .rayQuery = VK_TRUE,
        };
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
            .sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            .pNext                 = &rq,
            .accelerationStructure = VK_TRUE,
        };
        VkPhysicalDeviceDescriptorBufferFeaturesEXT dbf{
            .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
            .pNext            = _raytracingSupported ? static_cast<void *>(&asf) : nullptr,
            .descriptorBuffer = VK_TRUE,
        };
        VkPhysicalDeviceVulkan13Features f13{
            .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext            = &dbf,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };
        VkPhysicalDeviceVulkan12Features f12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &f13,
        };
        // Slang emits the DrawParameters capability for SV_VertexID/base-vertex,
        // which requires shaderDrawParameters (core-guaranteed on Vulkan 1.3 GPUs).
        VkPhysicalDeviceVulkan11Features f11{
            .sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext                = &f12,
            .shaderDrawParameters = VK_TRUE,
        };
        f12.descriptorIndexing              = VK_TRUE;
        f12.runtimeDescriptorArray          = VK_TRUE;
        f12.descriptorBindingPartiallyBound = VK_TRUE;
        f12.scalarBlockLayout               = VK_TRUE;
        f12.timelineSemaphore               = VK_TRUE;
        f12.bufferDeviceAddress             = VK_TRUE;
        f12.hostQueryReset                  = _computeTimestampsSupported;
        VkPhysicalDeviceFeatures features{
            .shaderInt64 = VK_TRUE,
        };

        std::vector<const char *> exts{VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME};
        {
            // Presentation is enabled when the adapter can present at all, for the same reason the
            // instance enables surface extensions: no surface exists yet to ask about.
            uint32_t deviceExtensionCount{0};
            vkEnumerateDeviceExtensionProperties(_physDev, nullptr, &deviceExtensionCount, nullptr);
            std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
            vkEnumerateDeviceExtensionProperties(_physDev, nullptr, &deviceExtensionCount, deviceExtensions.data());
            if (std::ranges::any_of(deviceExtensions,
                                    [](const VkExtensionProperties &extension)
                                    {
                                        return std::string_view{extension.extensionName} ==
                                               std::string_view{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
                                    }))
            {
                exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            }
        }
        if (_raytracingSupported)
        {
            exts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            exts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            exts.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        }

        VkDeviceCreateInfo ci{
            .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext                   = &f11,
            .queueCreateInfoCount    = static_cast<uint32_t>(qcis.size()),
            .pQueueCreateInfos       = qcis.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(exts.size()),
            .ppEnabledExtensionNames = exts.data(),
            .pEnabledFeatures        = &features,
        };
        VK_CHECK(vkCreateDevice(_physDev, &ci, nullptr, &_device));

        vkGetDeviceQueue(_device, gfxQ.family, 0, &gfxQ.q);
        vkGetDeviceQueue(_device, computeQ.family, 0, &computeQ.q);
        vkGetDeviceQueue(_device, transferQ.family, 0, &transferQ.q);
    }

    void VulkanDevice::_loadExtensionFunctions()
    {
        assert(_device != VK_NULL_HANDLE);

#define LOAD(pfn, name)                                                                                                \
    pfn = (decltype(pfn))vkGetDeviceProcAddr(_device, name);                                                           \
    if (!pfn)                                                                                                          \
    FailContract("failed to load " name)

        LOAD(pfn_GetDescriptorEXT, "vkGetDescriptorEXT");
        LOAD(pfn_GetLayoutSize, "vkGetDescriptorSetLayoutSizeEXT");
        LOAD(pfn_GetBindingOffset, "vkGetDescriptorSetLayoutBindingOffsetEXT");
        LOAD(pfn_CmdBindDescriptorBuffers, "vkCmdBindDescriptorBuffersEXT");
        LOAD(pfn_CmdSetDescriptorBufferOffsets, "vkCmdSetDescriptorBufferOffsetsEXT");

        if (_raytracingSupported)
        {
            LOAD(pfn_CreateAccelerationStructure, "vkCreateAccelerationStructureKHR");
            LOAD(pfn_DestroyAccelerationStructure, "vkDestroyAccelerationStructureKHR");
            LOAD(pfn_GetAccelerationStructureBuildSizes, "vkGetAccelerationStructureBuildSizesKHR");
            LOAD(pfn_CmdBuildAccelerationStructures, "vkCmdBuildAccelerationStructuresKHR");
            LOAD(pfn_GetAccelerationStructureDeviceAddress, "vkGetAccelerationStructureDeviceAddressKHR");
        }
#undef LOAD

        pfn_CmdBeginDebugUtilsLabel =
            (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(_device, "vkCmdBeginDebugUtilsLabelEXT");
        pfn_CmdEndDebugUtilsLabel =
            (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(_device, "vkCmdEndDebugUtilsLabelEXT");
        pfn_CmdInsertDebugUtilsLabel =
            (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetDeviceProcAddr(_device, "vkCmdInsertDebugUtilsLabelEXT");
    }

    void VulkanDevice::_createAllocator()
    {
        assert(_instance != VK_NULL_HANDLE);
        assert(_physDev != VK_NULL_HANDLE);
        assert(_device != VK_NULL_HANDLE);

        VmaVulkanFunctions vkFns{
            .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
            .vkGetDeviceProcAddr   = vkGetDeviceProcAddr,
        };
        VmaAllocatorCreateInfo ci{
            .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice   = _physDev,
            .device           = _device,
            .pVulkanFunctions = &vkFns,
            .instance         = _instance,
            .vulkanApiVersion = VK_API_VERSION_1_3,
        };
        VK_CHECK(vmaCreateAllocator(&ci, &_allocator));
    }

    void VulkanDevice::_createCommandPools()
    {
        assert(_device != VK_NULL_HANDLE);
        assert(gfxQ.family != ~0u);
        assert(computeQ.family != ~0u);
        assert(transferQ.family != ~0u);

        auto mk{[&](uint32_t family, VkCommandPool &pool)
                {
                    VkCommandPoolCreateInfo ci{
                        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                        .queueFamilyIndex = family,
                    };
                    VK_CHECK(vkCreateCommandPool(_device, &ci, nullptr, &pool));
                }};
        mk(gfxQ.family, gfxPool);
        mk(computeQ.family, computePool);
        mk(transferQ.family, xferPool);
    }

    void VulkanDevice::_createTimeline()
    {
        assert(_device != VK_NULL_HANDLE);

        VkSemaphoreTypeCreateInfo tc{
            .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue  = 0,
        };
        VkSemaphoreCreateInfo ci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tc};
        VK_CHECK(vkCreateSemaphore(_device, &ci, nullptr, &_timeline));
    }

    void VulkanDevice::_createGlobalLayout()
    {
        assert(_device != VK_NULL_HANDLE);
        assert(_heap.EmptyDescriptorSetLayout() != VK_NULL_HANDLE);
        assert(_heap.DescriptorSetLayout() != VK_NULL_HANDLE);

        VkPushConstantRange pc{
            .stageFlags = VK_SHADER_STAGE_ALL,
            .offset     = 0,
            .size       = _maxPushConstantBytes,
        };
        // Slang reserves descriptor set 1 for DescriptorHandle<T>. Keep set 0
        // explicitly empty so the pipeline layout exactly matches that ABI.
        VkDescriptorSetLayout      layouts[2]{_heap.EmptyDescriptorSetLayout(), _heap.DescriptorSetLayout()};
        VkPipelineLayoutCreateInfo ci{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 2,
            .pSetLayouts            = layouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(_device, &ci, nullptr, &_globalLayout));
    }

    // ============================================================================
    // VulkanBindlessHeap
    // ============================================================================

    void VulkanBindlessHeap::Init(VkDevice vkDev, VmaAllocator allocator,
                                  const VkPhysicalDeviceDescriptorBufferPropertiesEXT &properties,
                                  PFN_vkGetDescriptorEXT                               getDescriptor,
                                  PFN_vkGetDescriptorSetLayoutSizeEXT                  getLayoutSize,
                                  PFN_vkGetDescriptorSetLayoutBindingOffsetEXT         getBindingOffset)
    {
        assert(vkDev != VK_NULL_HANDLE);
        assert(allocator != nullptr);
        assert(getDescriptor != nullptr);
        assert(getLayoutSize != nullptr);
        assert(getBindingOffset != nullptr);

        _vkDev            = vkDev;
        _allocator        = allocator;
        _properties       = properties;
        _getDescriptor    = getDescriptor;
        _getLayoutSize    = getLayoutSize;
        _getBindingOffset = getBindingOffset;
        const auto &p{_properties};
        // 1. Slang's DescriptorHandle<T> heap in its system bindless space 1.
        // well-defined Vulkan lowering uses binding 0 for samplers and
        // binding 2 for sampled/resource images. Buffers remain bindless via
        // BDA and acceleration structures via their device addresses.
        VkDescriptorSetLayoutBinding bindings[3]{
            {0, VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers, VK_SHADER_STAGE_ALL, nullptr},
            {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxTextures, VK_SHADER_STAGE_ALL, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        };
        VkDescriptorBindingFlags bindingFlags[3]{
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            0,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount  = 3,
            .pBindingFlags = bindingFlags,
        };
        VkDescriptorSetLayoutCreateInfo lci{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext        = &bindingFlagsInfo,
            .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
            .bindingCount = 3,
            .pBindings    = bindings,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(_vkDev, &lci, nullptr, &_layout));
        VkDescriptorSetLayoutCreateInfo emptyLayoutCI{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(_vkDev, &emptyLayoutCI, nullptr, &_emptyLayout));

        // 2. Layout size and per-binding offsets
        VkDeviceSize layoutSize{};
        _getLayoutSize(_vkDev, _layout, &layoutSize);
        VkDeviceSize align{p.descriptorBufferOffsetAlignment};
        if (align > 1)
        {
            layoutSize = (layoutSize + align - 1) & ~(align - 1);
        }

        _getBindingOffset(_vkDev, _layout, 0, &_smpBindOff);
        _getBindingOffset(_vkDev, _layout, 2, &_texBindOff);
        _getBindingOffset(_vkDev, _layout, 3, &_bufBindOff);

        // Slot-to-address table retained for code that indexes buffers by
        // LightRHI handle. Most production kernels pass BDA values directly.
        VkBufferCreateInfo tableCI{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = static_cast<VkDeviceSize>(kMaxBuffers) * sizeof(uint64_t),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        };
        VmaAllocationCreateInfo tableAllocCI{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        };
        VmaAllocationInfo tableAllocInfo{};
        VK_CHECK(vmaCreateBuffer(_allocator, &tableCI, &tableAllocCI, &_bufTable, &_bufTableAlloc, &tableAllocInfo));
        _bufTableMapped = static_cast<uint64_t *>(tableAllocInfo.pMappedData);
        std::memset(_bufTableMapped, 0, static_cast<std::size_t>(tableCI.size));
        VK_CHECK(vmaFlushAllocation(_allocator, _bufTableAlloc, 0, VK_WHOLE_SIZE));
        VkBufferDeviceAddressInfo tableBDAInfo{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = _bufTable,
        };
        const VkDeviceAddress tableBDA{vkGetBufferDeviceAddress(_vkDev, &tableBDAInfo)};

        // 3. Allocate one persistently mapped descriptor buffer containing
        // the actual sampler and sampled-image descriptor arrays.
        VkBufferCreateInfo dbci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = layoutSize,
            .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                     VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        };
        VmaAllocationCreateInfo daci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        };
        VmaAllocationInfo dai{};
        VK_CHECK(vmaCreateBuffer(_allocator, &dbci, &daci, &_descBuf, &_descAlloc, &dai));
        _descMapped = dai.pMappedData;

        std::memset(_descMapped, 0, static_cast<std::size_t>(layoutSize));

        VkDescriptorAddressInfoEXT tableAddressInfo{
            .sType   = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
            .address = tableBDA,
            .range   = tableCI.size,
            .format  = VK_FORMAT_UNDEFINED,
        };
        VkDescriptorGetInfoEXT tableDescriptorInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .data  = {.pStorageBuffer = &tableAddressInfo},
        };
        _getDescriptor(_vkDev, &tableDescriptorInfo, p.storageBufferDescriptorSize,
                       static_cast<char *>(_descMapped) + _bufBindOff);
        VK_CHECK(vmaFlushAllocation(_allocator, _descAlloc, 0, VK_WHOLE_SIZE));

        // 4. BDA of the descriptor buffer itself (bound once per command list).
        VkBufferDeviceAddressInfo bda{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = _descBuf,
        };
        _heapBDA = GpuAddress{vkGetBufferDeviceAddress(_vkDev, &bda)};
    }

    void VulkanBindlessHeap::Destroy(VkDevice vkDev)
    {
        if (_layout)
        {
            vkDestroyDescriptorSetLayout(vkDev, _layout, nullptr);
            _layout = VK_NULL_HANDLE;
        }
        if (_emptyLayout)
        {
            vkDestroyDescriptorSetLayout(vkDev, _emptyLayout, nullptr);
            _emptyLayout = VK_NULL_HANDLE;
        }
        if (_descBuf)
        {
            vmaDestroyBuffer(_allocator, _descBuf, _descAlloc);
            _descBuf = VK_NULL_HANDLE;
        }
        if (_bufTable)
        {
            vmaDestroyBuffer(_allocator, _bufTable, _bufTableAlloc);
            _bufTable = VK_NULL_HANDLE;
        }
    }

    void VulkanBindlessHeap::RegisterBuffer(uint32_t slot, VkBuffer /*buf*/, uint64_t /*size*/, VkDeviceAddress bda)
    {
        _bufTableMapped[slot] = bda;
        VK_CHECK(vmaFlushAllocation(_allocator, _bufTableAlloc, static_cast<VkDeviceSize>(slot) * sizeof(uint64_t),
                                    sizeof(uint64_t)));
        ++_usedBufs;
    }

    void VulkanBindlessHeap::RegisterTexture(uint32_t slot, VkImageView view)
    {
        VkDescriptorImageInfo imageInfo{
            .imageView   = view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkDescriptorGetInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .data  = {.pSampledImage = &imageInfo},
        };
        const auto &p{_properties};
        _getDescriptor(_vkDev, &info, p.sampledImageDescriptorSize,
                       static_cast<char *>(_descMapped) + _texBindOff +
                           static_cast<VkDeviceSize>(slot) * p.sampledImageDescriptorSize);
        VK_CHECK(vmaFlushAllocation(_allocator, _descAlloc,
                                    _texBindOff + static_cast<VkDeviceSize>(slot) * p.sampledImageDescriptorSize,
                                    p.sampledImageDescriptorSize));
        ++_usedTexs;
    }

    void VulkanBindlessHeap::RegisterSampler(uint32_t slot, VkSampler smp)
    {
        VkDescriptorGetInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type  = VK_DESCRIPTOR_TYPE_SAMPLER,
            .data  = {.pSampler = &smp},
        };
        const auto &p{_properties};
        _getDescriptor(_vkDev, &info, p.samplerDescriptorSize,
                       static_cast<char *>(_descMapped) + _smpBindOff +
                           static_cast<VkDeviceSize>(slot) * p.samplerDescriptorSize);
        VK_CHECK(vmaFlushAllocation(_allocator, _descAlloc,
                                    _smpBindOff + static_cast<VkDeviceSize>(slot) * p.samplerDescriptorSize,
                                    p.samplerDescriptorSize));
        ++_usedSmps;
    }

    // Zero the table entry so a destroyed resource leaves no usable value behind.
    // Critically for buffers: the slot pool recycles indices, and the next buffer
    // to land on this slot may not be a Storage Buffer (RegisterBuffer is only
    // called for those) — without this, the table would still hold the freed
    // buffer's device address for a shader to dereference.
    void VulkanBindlessHeap::UnregisterBuffer(uint32_t slot)
    {
        _bufTableMapped[slot] = 0;
        VK_CHECK(vmaFlushAllocation(_allocator, _bufTableAlloc, static_cast<VkDeviceSize>(slot) * sizeof(uint64_t),
                                    sizeof(uint64_t)));
        if (_usedBufs)
        {
            --_usedBufs;
        }
    }

    void VulkanBindlessHeap::UnregisterTexture(uint32_t slot)
    {
        const auto size{_properties.sampledImageDescriptorSize};
        std::memset(static_cast<char *>(_descMapped) + _texBindOff + static_cast<VkDeviceSize>(slot) * size, 0,
                    static_cast<std::size_t>(size));
        VK_CHECK(
            vmaFlushAllocation(_allocator, _descAlloc, _texBindOff + static_cast<VkDeviceSize>(slot) * size, size));
        if (_usedTexs)
        {
            --_usedTexs;
        }
    }

    void VulkanBindlessHeap::UnregisterSampler(uint32_t slot)
    {
        const auto size{_properties.samplerDescriptorSize};
        std::memset(static_cast<char *>(_descMapped) + _smpBindOff + static_cast<VkDeviceSize>(slot) * size, 0,
                    static_cast<std::size_t>(size));
        VK_CHECK(
            vmaFlushAllocation(_allocator, _descAlloc, _smpBindOff + static_cast<VkDeviceSize>(slot) * size, size));
        if (_usedSmps)
        {
            --_usedSmps;
        }
    }

    // ============================================================================
    // Resource creation / destruction
    // ============================================================================

    VkBufferUsageFlags VulkanDevice::_toBufUsage(BufferUsage u) const noexcept
    {
        VkBufferUsageFlags f{0};
        if (HasUsage(u, BufferUsage::Vertex))
        {
            f |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        }
        if (HasUsage(u, BufferUsage::Index))
        {
            f |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        }
        if (HasUsage(u, BufferUsage::Constant))
        {
            f |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        }
        if (HasUsage(u, BufferUsage::Storage))
        {
            f |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }
        if (HasUsage(u, BufferUsage::IndirectArgs))
        {
            f |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }
        if (HasUsage(u, BufferUsage::TransferSrc))
        {
            f |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        }
        if (HasUsage(u, BufferUsage::TransferDst))
        {
            f |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }
        // The backend records a BDA for every buffer so BufferAddress() is
        // consistently available and bindless registration can use the same path.
        f |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        // All buffers may be registered in the descriptor buffer heap
        f |= VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
        // Any buffer may be fed to an acceleration-structure build as vertex,
        // index or instance data, and the spec requires the bit at creation time
        // (VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673).
        if (_raytracingSupported)
        {
            f |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        }
        return f;
    }

    VkImageUsageFlags VulkanDevice::_toImageUsage(TextureUsage u) const noexcept
    {
        VkImageUsageFlags f{0};
        if (HasUsage(u, TextureUsage::Sampled))
        {
            f |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (HasUsage(u, TextureUsage::Storage))
        {
            f |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if (HasUsage(u, TextureUsage::RenderTarget))
        {
            f |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (HasUsage(u, TextureUsage::DepthStencil))
        {
            f |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }
        if (HasUsage(u, TextureUsage::TransferSrc))
        {
            f |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        if (HasUsage(u, TextureUsage::TransferDst))
        {
            f |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }
        // Always allow clearing / copying in both directions — render targets are
        // routinely read back (CopyTextureToBuffer) and every image can be cleared.
        f |= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        return f;
    }

    VmaMemoryUsage VulkanDevice::_toVmaUsage(MemoryType m) const noexcept
    {
        switch (m)
        {
            case MemoryType::GpuOnly:
                return VMA_MEMORY_USAGE_GPU_ONLY;
            case MemoryType::CpuToGpu:
                return VMA_MEMORY_USAGE_CPU_TO_GPU;
            case MemoryType::GpuToCpu:
                return VMA_MEMORY_USAGE_GPU_TO_CPU;
        }
        return VMA_MEMORY_USAGE_GPU_ONLY;
    }

    VkImageAspectFlags VulkanDevice::_aspectMask(Format f) const noexcept
    {
        switch (f)
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

    std::expected<std::reference_wrapper<VkMemoryHeap_>, PlacementError>
    VulkanDevice::_reservePlacement(const HeapPlacement &placement, uint32_t memoryTypeIndex, uint64_t size,
                                    uint64_t alignment)
    {
        if (!placement.Heap.Valid())
        {
            return std::unexpected(PlacementError::InvalidHeap);
        }
        auto &memoryHeap{_memoryHeaps.get(placement.Heap.Index)};
        if (memoryHeap.memory == VK_NULL_HANDLE)
        {
            return std::unexpected(PlacementError::InvalidHeap);
        }
        if (memoryHeap.memoryTypeIndex != memoryTypeIndex)
        {
            return std::unexpected(PlacementError::IncompatibleMemory);
        }
        if (alignment == 0 || placement.Offset % alignment != 0)
        {
            return std::unexpected(PlacementError::MisalignedOffset);
        }
        const uint64_t end{placement.Offset + size};
        if (end > memoryHeap.size || end < placement.Offset)
        {
            return std::unexpected(PlacementError::OutOfRange);
        }

        // Refuse a range that a live placement already covers. `upper_bound` names the first
        // placement starting after this one, so only its predecessor and itself can overlap.
        const auto next{memoryHeap.liveRanges.upper_bound(placement.Offset)};
        if (next != memoryHeap.liveRanges.end() && next->first < end)
        {
            return std::unexpected(PlacementError::Overlapping);
        }
        if (next != memoryHeap.liveRanges.begin() && std::prev(next)->second > placement.Offset)
        {
            return std::unexpected(PlacementError::Overlapping);
        }

        memoryHeap.liveRanges.emplace(placement.Offset, end);
        return std::ref(memoryHeap);
    }

    VkBufferCreateInfo VulkanDevice::_bufferCreateInfo(const BufferDesc        &d,
                                                       std::array<uint32_t, 3> &queueFamilies) const
    {
        queueFamilies = {gfxQ.family, computeQ.family, transferQ.family};
        const uint32_t queueFamilyCount{uniqueQueueFamilies(queueFamilies)};
        return VkBufferCreateInfo{
            .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size                  = d.Size,
            .usage                 = _toBufUsage(d.Usage),
            .sharingMode           = queueFamilyCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = queueFamilyCount > 1 ? queueFamilyCount : 0,
            .pQueueFamilyIndices   = queueFamilyCount > 1 ? queueFamilies.data() : nullptr,
        };
    }

    void VulkanDevice::_finishBuffer(uint32_t idx, const BufferDesc &d)
    {
        auto &rec{_buffers.get(idx)};
        // Buffer device address (BDA) — always get it so shaders can read data directly
        const VkBufferDeviceAddressInfo bi{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = rec.buffer,
        };
        rec.bda = vkGetBufferDeviceAddress(_device, &bi);

        // Register storage buffers in the bindless heap
        if (HasUsage(d.Usage, BufferUsage::Storage))
        {
            _heap.RegisterBuffer(idx, rec.buffer, rec.size, rec.bda);
        }

        if (!d.DebugName.empty())
        {
            _setDebugName(VK_OBJECT_TYPE_BUFFER, (uint64_t)rec.buffer, d.DebugName);
        }
    }

    std::expected<BufferHandle, DeviceError> VulkanDevice::CreateBuffer(const BufferDesc &d)
    {
        std::array<uint32_t, 3>  queueFamilies{};
        const VkBufferCreateInfo bci{_bufferCreateInfo(d, queueFamilies)};
        VmaAllocationCreateInfo  aci{
            .flags = d.MemoryType != MemoryType::GpuOnly
                         ? (VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
                         : VmaAllocationCreateFlags(0),
            .usage = _toVmaUsage(d.MemoryType),
        };

        uint32_t idx{_buffers.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many buffers as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_buffers.get(idx)};
        rec.size  = d.Size;
        rec.usage = d.Usage;
        VK_CHECK(vmaCreateBuffer(_allocator, &bci, &aci, &rec.buffer, &rec.alloc, nullptr));

        _finishBuffer(idx, d);
        return BufferHandle{idx};
    }

    std::expected<PlacementRequirements, PlacementError>
    VulkanDevice::BufferPlacementRequirements(const BufferDesc &d) const
    {
        // Vulkan reports a buffer's requirements from the buffer, so one is created and destroyed
        // here purely to ask. No memory is bound to it.
        std::array<uint32_t, 3>  queueFamilies{};
        const VkBufferCreateInfo bci{_bufferCreateInfo(d, queueFamilies)};

        VkBuffer buffer{VK_NULL_HANDLE};
        if (vkCreateBuffer(_device, &bci, nullptr, &buffer) != VK_SUCCESS)
        {
            return std::unexpected{PlacementError::CreationFailed};
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(_device, buffer, &requirements);
        vkDestroyBuffer(_device, buffer, nullptr);

        if (requirements.size == 0 || requirements.alignment == 0)
        {
            return std::unexpected{PlacementError::CreationFailed};
        }
        if ((requirements.memoryTypeBits & (1U << _memoryTypeIndexFor(d.MemoryType))) == 0)
        {
            return std::unexpected{PlacementError::IncompatibleMemory};
        }
        return PlacementRequirements{.Size = requirements.size, .Alignment = requirements.alignment};
    }

    std::expected<BufferHandle, PlacementError> VulkanDevice::CreateBuffer(const BufferDesc    &d,
                                                                           const HeapPlacement &placement)
    {
        std::array<uint32_t, 3>  queueFamilies{};
        const VkBufferCreateInfo bci{_bufferCreateInfo(d, queueFamilies)};

        VkBuffer buffer{VK_NULL_HANDLE};
        VK_CHECK(vkCreateBuffer(_device, &bci, nullptr, &buffer));

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(_device, buffer, &requirements);

        const auto refuse{[&](PlacementError error) -> std::expected<BufferHandle, PlacementError>
                          {
                              vkDestroyBuffer(_device, buffer, nullptr);
                              return std::unexpected(error);
                          }};

        const uint32_t memoryTypeIndex{_memoryTypeIndexFor(d.MemoryType)};
        if ((requirements.memoryTypeBits & (1U << memoryTypeIndex)) == 0)
        {
            return refuse(PlacementError::IncompatibleMemory);
        }

        auto reserved{_reservePlacement(placement, memoryTypeIndex, requirements.size, requirements.alignment)};
        if (!reserved)
        {
            return refuse(reserved.error());
        }
        if (vkBindBufferMemory(_device, buffer, reserved->get().memory, placement.Offset) != VK_SUCCESS)
        {
            reserved->get().liveRanges.erase(placement.Offset);
            return refuse(PlacementError::CreationFailed);
        }

        uint32_t idx{_buffers.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many buffers as it can");
            return std::unexpected{PlacementError::CreationFailed};
        }
        auto &rec{_buffers.get(idx)};
        rec.buffer = buffer;
        rec.size   = d.Size;
        rec.usage  = d.Usage;
        rec.heap   = placement.Heap;
        rec.offset = placement.Offset;

        _finishBuffer(idx, d);
        return BufferHandle{idx};
    }

    void VulkanDevice::DestroyBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &rec{_buffers.get(h.Index)};
        if (HasUsage(rec.usage, BufferUsage::Storage))
        {
            _heap.UnregisterBuffer(h.Index);
        }
        if (rec.heap.Valid())
        {
            _memoryHeaps.get(rec.heap.Index).liveRanges.erase(rec.offset);
            vkDestroyBuffer(_device, rec.buffer, nullptr);
        }
        else
        {
            vmaDestroyBuffer(_allocator, rec.buffer, rec.alloc);
        }
        _buffers.free(h.Index);
    }

    GpuAddress VulkanDevice::BufferAddress(BufferHandle h) const
    {
        return h.Valid() ? GpuAddress{_buffers.get(h.Index).bda} : GpuAddress{};
    }

    BufferInfo VulkanDevice::GetBufferInfo(BufferHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        const auto &r{_buffers.get(h.Index)};
        return BufferInfo{.Size = r.size, .Usage = r.usage, .DeviceAddress = GpuAddress{r.bda}};
    }

    TextureInfo VulkanDevice::GetTextureInfo(TextureHandle h) const
    {
        if (!h.Valid())
        {
            return {};
        }
        const TextureDesc &desc{_textures.get(h.Index).desc};
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

    MappedBuffer VulkanDevice::MapBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return {};
        }
        auto &buffer{_buffers.get(h.Index)};
        void *ptr{nullptr};
        VK_CHECK(vmaMapMemory(_allocator, buffer.alloc, &ptr));
        VK_CHECK(vmaInvalidateAllocation(_allocator, buffer.alloc, 0, VK_WHOLE_SIZE));
        return MappedBuffer{ptr, buffer.size};
    }

    void VulkanDevice::UnmapBuffer(BufferHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        const auto allocation{_buffers.get(h.Index).alloc};
        VK_CHECK(vmaFlushAllocation(_allocator, allocation, 0, VK_WHOLE_SIZE));
        vmaUnmapMemory(_allocator, allocation);
    }

    // ============================================================================
    // Presentation
    //
    // The application owns the surface; the device owns the swapchain it builds on that surface and
    // rebuilds whenever the surface's size changes or the platform reports the chain is stale. A
    // frame is one acquired image, waited on by the submission that draws it and signalled back to
    // the presentation engine when that work completes.
    // ============================================================================

    std::expected<TextureHandle, DeviceError> VulkanDevice::AdoptExternalTexture(VkImage image, VkImageView view,
                                                                                const TextureDesc &desc)
    {
        const uint32_t idx{_textures.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many textures as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_textures.get(idx)};
        rec.image    = image;
        rec.view     = view;
        rec.borrowed = true;
        rec.layout   = VK_IMAGE_LAYOUT_UNDEFINED;
        rec.desc     = desc;
        return TextureHandle{idx};
    }

    FenceHandle VulkanDevice::SubmitForPresent(ICommandList &cmd, const SubmitDesc &desc, VkSemaphore waitBinary,
                                               VkSemaphore signalBinary)
    {
        return _submit(cmd, desc, waitBinary, signalBinary);
    }

    VkResult VulkanDevice::PresentImage(VkSwapchainKHR swapchain, uint32_t image, VkSemaphore waitBinary)
    {
        const VkPresentInfoKHR pi{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &waitBinary,
            .swapchainCount     = 1,
            .pSwapchains        = &swapchain,
            .pImageIndices      = &image,
        };
        std::scoped_lock lk{gfxQ.mtx};
        return vkQueuePresentKHR(gfxQ.q, &pi);
    }

    // ============================================================================
    // VulkanPresentingTextureProvider
    //
    // The surface is the application's; the chain built over it is this target's, rebuilt whenever
    // the surface changes size. An image is borrowed for one frame: it takes a texture slot so the
    // rest of the API can name it, while the chain keeps the image and its view.
    // ============================================================================

    VulkanPresentingTextureProvider::VulkanPresentingTextureProvider(VulkanDevice &device, VkInstance instance,
                                                                     VkSurfaceKHR surface, Format requestedFormat,
                                                                     PresentTiming timing) noexcept
        : _device{device}, _instance{instance}, _surface{surface}, _requestedFormat{requestedFormat}, _timing{timing},
          _handle{device.RegisterExternalTextureProvider(*this)}
    {
    }

    VkPresentModeKHR VulkanPresentingTextureProvider::_presentMode() const
    {
        if (_timing != PresentTiming::AsSoonAsReady)
        {
            return VK_PRESENT_MODE_FIFO_KHR;
        }
        uint32_t count{0};
        vkGetPhysicalDeviceSurfacePresentModesKHR(_device.NativePhysicalDevice(), _surface, &count, nullptr);
        std::vector<VkPresentModeKHR> modes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(_device.NativePhysicalDevice(), _surface, &count, modes.data());
        const bool mailbox{std::ranges::find(modes, VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()};
        return mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
    }

    ExternalTextureProviderHandle VulkanPresentingTextureProvider::Handle() const noexcept
    {
        return _handle;
    }

#if METRICS_ENABLED
    std::size_t VulkanPresentingTextureProvider::TakeShownFrames(std::span<ShownFrame> /*frames*/) noexcept
    {
        return 0;
    }
#endif

    ExternalTextureProviderHandle VulkanDevice::RegisterExternalTextureProvider(IFramePresenter &target)
    {
        const uint32_t index{_externalTextureProviders.alloc()};
        if (index == kInvalidIndex)
        {
            ReportRefusal("this device already holds as many external texture providers as it can");
            return {};
        }
        _externalTextureProviders.get(index) = &target;
        return ExternalTextureProviderHandle{.Index = index};
    }

    void VulkanDevice::UnregisterExternalTextureProvider(ExternalTextureProviderHandle handle) noexcept
    {
        if (handle.Valid())
        {
            _externalTextureProviders.free(handle.Index);
        }
    }

    IFramePresenter *VulkanDevice::ResolveExternalTextureProvider(ExternalTextureProviderHandle handle) const noexcept
    {
        return handle.Valid() ? _externalTextureProviders.get(handle.Index) : nullptr;
    }

    VulkanPresentingTextureProvider::~VulkanPresentingTextureProvider()
    {
        // Unregistered first: a present recorded against this handle must not resolve to a target
        // that is already tearing its chain down.
        _device.UnregisterExternalTextureProvider(_handle);
        const VkDevice native{_device.NativeDevice()};
        vkDeviceWaitIdle(native);
        _releaseSurfaceTexture();
        _destroySwapchain();
        if (_swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(native, _swapchain, nullptr);
        }
        if (_surface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(_instance, _surface, nullptr);
        }
    }

    Format VulkanPresentingTextureProvider::TextureFormat() const noexcept
    {
        return _format;
    }

    Extent2D VulkanPresentingTextureProvider::_surfaceExtent() const noexcept
    {
        VkSurfaceCapabilitiesKHR caps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_device.NativePhysicalDevice(), _surface, &caps) != VK_SUCCESS)
        {
            return Extent2D{.Width = 0, .Height = 0};
        }
        if (caps.currentExtent.width == UINT32_MAX)
        {
            // Some platforms leave the size to the swapchain rather than reporting the window's.
            // The surface is the only size authority this API has, so take the largest it allows.
            return Extent2D{.Width = caps.maxImageExtent.width, .Height = caps.maxImageExtent.height};
        }
        return Extent2D{.Width = caps.currentExtent.width, .Height = caps.currentExtent.height};
    }

    bool VulkanPresentingTextureProvider::_createSwapchain()
    {
        const VkDevice           native{_device.NativeDevice()};
        const VkPhysicalDevice   physical{_device.NativePhysicalDevice()};
        VkSurfaceCapabilitiesKHR caps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, _surface, &caps) != VK_SUCCESS)
        {
            return false;
        }
        const Extent2D surfaceExtent{_surfaceExtent()};
        if (surfaceExtent.Width == 0 || surfaceExtent.Height == 0)
        {
            return false; // nothing to build until the surface has area again
        }
        const VkExtent2D chainExtent{.width = surfaceExtent.Width, .height = surfaceExtent.Height};

        uint32_t formatCount{0};
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical, _surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical, _surface, &formatCount, formats.data());
        if (formats.empty())
        {
            return false;
        }

        // The requested format when the surface offers it. Otherwise the surface's own first choice,
        // which is reported back through TextureFormat() as BGRA8Unorm — the format every surface
        // this backend targets presents — so a caller always builds pipelines against the truth.
        VkSurfaceFormatKHR chosen{formats.front()};
        Format             chosenFormat{Format::BGRA8Unorm};
        const VkFormat     wanted{_requestedFormat == Format::Undefined ? VK_FORMAT_UNDEFINED
                                                                        : toVkFormat(_requestedFormat)};
        for (const VkSurfaceFormatKHR &candidate : formats)
        {
            if (candidate.format == wanted)
            {
                chosen       = candidate;
                chosenFormat = _requestedFormat;
                break;
            }
        }

        const uint32_t imageCount{caps.maxImageCount > 0 ? std::min(caps.minImageCount + 1, caps.maxImageCount)
                                                         : caps.minImageCount + 1};
        VkSwapchainKHR previous{_swapchain};
        const VkSwapchainCreateInfoKHR ci{
            .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface          = _surface,
            .minImageCount    = imageCount,
            .imageFormat      = chosen.format,
            .imageColorSpace  = chosen.colorSpace,
            .imageExtent      = chainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform     = caps.currentTransform,
            .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            // FIFO is the only mode every implementation must support; MAILBOX, when asked for and
            // offered, also never tears but lets the newest frame replace one still waiting.
            .presentMode  = _presentMode(),
            .clipped      = VK_TRUE,
            .oldSwapchain = previous,
        };

        VkSwapchainKHR created{VK_NULL_HANDLE};
        const VkResult result{vkCreateSwapchainKHR(native, &ci, nullptr, &created)};
        if (result == VK_ERROR_DEVICE_LOST)
        {
            _device.MarkDeviceLost();
            return false;
        }
        if (result != VK_SUCCESS)
        {
            return false;
        }

        // oldSwapchain lets the new chain reuse the old one's resources; it does not release it.
        // Its images may still be in flight, so wait before tearing the old one down.
        if (previous != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(native);
        }
        _destroySwapchain();
        if (previous != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(native, previous, nullptr);
        }
        _swapchain       = created;
        _swapchainFormat = chosen.format;
        _swapchainExtent = chainExtent;
        _format          = chosenFormat;

        uint32_t count{0};
        vkGetSwapchainImagesKHR(native, _swapchain, &count, nullptr);
        _swapchainImages.resize(count);
        vkGetSwapchainImagesKHR(native, _swapchain, &count, _swapchainImages.data());

        _swapchainViews.resize(count);
        _presentSemaphores.resize(count);
        for (uint32_t i{0}; i < count; ++i)
        {
            const VkImageViewCreateInfo vci{
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = _swapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = _swapchainFormat,
                .subresourceRange =
                    {
                        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0,
                        .levelCount     = 1,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
            };
            VK_CHECK(vkCreateImageView(native, &vci, nullptr, &_swapchainViews[i]));

            const VkSemaphoreCreateInfo sci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VK_CHECK(vkCreateSemaphore(native, &sci, nullptr, &_presentSemaphores[i]));
        }

        // One acquisition semaphore per frame in flight, reused round-robin.
        _acquireSemaphores.resize(count);
        for (uint32_t i{0}; i < count; ++i)
        {
            const VkSemaphoreCreateInfo sci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VK_CHECK(vkCreateSemaphore(native, &sci, nullptr, &_acquireSemaphores[i]));
        }
        _frameSlot    = 0;
        _currentImage = UINT32_MAX;
        return true;
    }

    void VulkanPresentingTextureProvider::_destroySwapchain()
    {
        const VkDevice native{_device.NativeDevice()};
        for (VkImageView view : _swapchainViews)
        {
            vkDestroyImageView(native, view, nullptr);
        }
        for (VkSemaphore semaphore : _presentSemaphores)
        {
            vkDestroySemaphore(native, semaphore, nullptr);
        }
        for (VkSemaphore semaphore : _acquireSemaphores)
        {
            vkDestroySemaphore(native, semaphore, nullptr);
        }
        _swapchainViews.clear();
        _presentSemaphores.clear();
        _acquireSemaphores.clear();
        _swapchainImages.clear();
        // The swapchain handle itself is destroyed by whoever retires it: _createSwapchain when it
        // replaces the chain, the destructor when the target goes away.
    }

    void VulkanPresentingTextureProvider::_releaseSurfaceTexture()
    {
        if (_texture.Valid())
        {
            _device.DestroyTexture(_texture);
            _texture = TextureHandle{};
        }
    }

    std::expected<TextureHandle, ExternalTextureError> VulkanPresentingTextureProvider::NextTexture()
    {
        if (_currentImage != UINT32_MAX)
        {
            // Acquiring again without presenting holds one of the chain's images for nothing, and
            // a chain has few: a frame is taken, drawn and shown.
            FailContract("taking a frame from a surface that is already holding one");
        }
        if (_device.DeviceLost())
        {
            return std::unexpected{ExternalTextureError::Lost};
        }

        const Extent2D extent{_surfaceExtent()};
        if (extent.Width == 0 || extent.Height == 0)
        {
            return std::unexpected{ExternalTextureError::Unavailable}; // nothing to draw into without area
        }
        if (_swapchain == VK_NULL_HANDLE || extent.Width != _swapchainExtent.width ||
            extent.Height != _swapchainExtent.height)
        {
            if (!_createSwapchain())
            {
                return std::unexpected{_device.DeviceLost() ? ExternalTextureError::Lost : ExternalTextureError::Unavailable};
            }
        }

        const VkSemaphore acquireSemaphore{_acquireSemaphores[_frameSlot]};
        uint32_t          imageIndex{0};
        // No wait: when every image is still queued for the display, VK_NOT_READY comes back at once
        // and the caller retries later, instead of blocking its thread for a refresh.
        const VkResult result{vkAcquireNextImageKHR(_device.NativeDevice(), _swapchain, 0U, acquireSemaphore,
                                                    VK_NULL_HANDLE, &imageIndex)};
        if (result == VK_ERROR_DEVICE_LOST)
        {
            _device.MarkDeviceLost();
            return std::unexpected{ExternalTextureError::Lost};
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            // The surface moved on: rebuild and let the caller ask again next frame.
            std::ignore = _createSwapchain();
            return std::unexpected{ExternalTextureError::Unavailable};
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            return std::unexpected{ExternalTextureError::Unavailable};
        }
        _currentImage = imageIndex;

        const TextureDesc desc{
            .Format = _format,
            .Extent = {_swapchainExtent.width, _swapchainExtent.height, 1},
            .Usage  = TextureUsage::RenderTarget,
        };
        auto adopted{_device.AdoptExternalTexture(_swapchainImages[imageIndex], _swapchainViews[imageIndex], desc)};
        if (!adopted)
        {
            // The frame cannot be named, so it is skipped rather than presented half-used.
            _currentImage = UINT32_MAX;
            return std::unexpected{ExternalTextureError::Unavailable};
        }
        _texture = *adopted;
        return _texture;
    }

    FenceHandle VulkanPresentingTextureProvider::SubmitAndPresentHeldFrame(ICommandList &cmd, const SubmitDesc &desc)
    {
        if (_currentImage == UINT32_MAX || !_texture.Valid())
        {
            FailContract("presenting a surface that is holding no frame");
        }

        // The submission that drew the frame carries the ordering: it waits on the acquisition and
        // signals what the present waits on, so showing a frame costs no second submission.
        const VkSemaphore acquired{_acquireSemaphores[_frameSlot]};
        const VkSemaphore drawn{_presentSemaphores[_currentImage]};
        const FenceHandle fence{_device.SubmitForPresent(cmd, desc, acquired, drawn)};
        const VkResult    result{_device.PresentImage(_swapchain, _currentImage, drawn)};

        _releaseSurfaceTexture();
        _currentImage = UINT32_MAX;
        _frameSlot    = static_cast<uint32_t>((_frameSlot + 1) % _acquireSemaphores.size());

        if (result == VK_ERROR_DEVICE_LOST)
        {
            _device.MarkDeviceLost();
        }
        return fence;
    }

    // ============================================================================
    // Memory
    //
    // A memory heap is one VkDeviceMemory allocation the application places textures into. VMA
    // stays in charge of ordinary resources; a heap is deliberately raw, because its suballocation
    // is the caller's job.
    // ============================================================================

    uint32_t VulkanDevice::_memoryTypeIndexFor(MemoryType memory) const
    {
        VkMemoryPropertyFlags required{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
        switch (memory)
        {
            case MemoryType::GpuOnly:
                required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                break;
            case MemoryType::CpuToGpu:
                required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                break;
            case MemoryType::GpuToCpu:
                required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                break;
        }

        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(_physDev, &properties);
        for (uint32_t i{0}; i < properties.memoryTypeCount; ++i)
        {
            if ((properties.memoryTypes[i].propertyFlags & required) == required)
            {
                return i;
            }
        }
        FailContract("no memory type matches the requested MemoryType");
    }

    std::expected<MemoryHeapHandle, DeviceError> VulkanDevice::CreateMemoryHeap(const MemoryHeapDesc &d)
    {
        const uint32_t       memoryTypeIndex{_memoryTypeIndexFor(d.MemoryType)};
        VkMemoryAllocateInfo ai{
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = d.Size,
            .memoryTypeIndex = memoryTypeIndex,
        };

        VkDeviceMemory memory{VK_NULL_HANDLE};
        VK_CHECK(vkAllocateMemory(_device, &ai, nullptr, &memory));

        if (!d.DebugName.empty())
        {
            _setDebugName(VK_OBJECT_TYPE_DEVICE_MEMORY, (uint64_t)memory, d.DebugName);
        }

        uint32_t idx{_memoryHeaps.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many memory heaps as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_memoryHeaps.get(idx)};
        rec.memory          = memory;
        rec.size            = d.Size;
        rec.type            = d.MemoryType;
        rec.memoryTypeIndex = memoryTypeIndex;
        return MemoryHeapHandle{idx};
    }

    void VulkanDevice::DestroyMemoryHeap(MemoryHeapHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        vkFreeMemory(_device, _memoryHeaps.get(h.Index).memory, nullptr);
        _memoryHeaps.free(h.Index);
    }

    VulkanDevice::PlacementPlan VulkanDevice::_planPlacement(const TextureDesc &d, MemoryType memory) const
    {
        const uint32_t memoryTypeIndex{_memoryTypeIndexFor(memory)};

        // Vulkan reports an image's requirements from the image, so one is created and destroyed
        // here purely to ask. No memory is bound to it. Optimal tiling is the fast layout; linear
        // tiling is the fallback host-visible memory accepts on adapters that refuse the fast one
        // there. The first layout this memory type can hold wins.
        for (const VkImageTiling tiling : {VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_TILING_LINEAR})
        {
            std::array<uint32_t, 3> queueFamilies{};
            VkImageViewType         viewType{VK_IMAGE_VIEW_TYPE_2D};
            VkImageAspectFlags      aspect{};
            VkImageCreateInfo       ici{_imageCreateInfo(d, queueFamilies, viewType, aspect)};
            ici.tiling = tiling;

            VkImage image{VK_NULL_HANDLE};
            if (vkCreateImage(_device, &ici, nullptr, &image) != VK_SUCCESS)
            {
                continue; // this layout cannot describe the texture at all
            }
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(_device, image, &requirements);
            vkDestroyImage(_device, image, nullptr);

            if ((requirements.memoryTypeBits & (1U << memoryTypeIndex)) != 0)
            {
                return PlacementPlan{
                    .tiling          = tiling,
                    .memoryTypeIndex = memoryTypeIndex,
                    .size            = requirements.size,
                    .alignment       = requirements.alignment,
                    .viable          = true,
                };
            }
        }
        return PlacementPlan{};
    }

    std::expected<PlacementRequirements, PlacementError>
    VulkanDevice::TexturePlacementRequirements(const TextureDesc &d, MemoryType memory) const
    {
        const PlacementPlan plan{_planPlacement(d, memory)};
        if (!plan.viable)
        {
            return std::unexpected{PlacementError::IncompatibleMemory};
        }
        if (plan.size == 0 || plan.alignment == 0)
        {
            return std::unexpected{PlacementError::CreationFailed};
        }
        return PlacementRequirements{.Size = plan.size, .Alignment = plan.alignment};
    }

    VkImageCreateInfo VulkanDevice::_imageCreateInfo(const TextureDesc &d, std::array<uint32_t, 3> &queueFamilies,
                                                     VkImageViewType &outViewType, VkImageAspectFlags &outAspect) const
    {
        // Image type / view type
        VkImageType imgType{VK_IMAGE_TYPE_2D};
        outViewType = VK_IMAGE_VIEW_TYPE_2D;
        uint32_t cubeFlags{0};
        switch (d.Dimension)
        {
            case TextureDimension::Tex1D:
                imgType     = VK_IMAGE_TYPE_1D;
                outViewType = VK_IMAGE_VIEW_TYPE_1D;
                break;
            case TextureDimension::Tex2D:
                imgType     = VK_IMAGE_TYPE_2D;
                outViewType = VK_IMAGE_VIEW_TYPE_2D;
                break;
            case TextureDimension::Tex3D:
                imgType     = VK_IMAGE_TYPE_3D;
                outViewType = VK_IMAGE_VIEW_TYPE_3D;
                break;
            case TextureDimension::TexCube:
                imgType     = VK_IMAGE_TYPE_2D;
                outViewType = VK_IMAGE_VIEW_TYPE_CUBE;
                cubeFlags   = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                break;
            case TextureDimension::Tex1DArray:
                imgType     = VK_IMAGE_TYPE_1D;
                outViewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
                break;
            case TextureDimension::Tex2DArray:
                imgType     = VK_IMAGE_TYPE_2D;
                outViewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                break;
            case TextureDimension::TexCubeArray:
                imgType     = VK_IMAGE_TYPE_2D;
                outViewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
                cubeFlags   = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                break;
        }

        outAspect = _aspectMask(d.Format);

        queueFamilies = {gfxQ.family, computeQ.family, transferQ.family};
        const uint32_t queueFamilyCount{uniqueQueueFamilies(queueFamilies)};
        return VkImageCreateInfo{
            .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags     = cubeFlags,
            .imageType = imgType,
            .format    = toVkFormat(d.Format),
            .extent = {d.Extent.Width, d.Extent.Height, d.Dimension == TextureDimension::Tex3D ? d.Extent.Depth : 1u},
            .mipLevels             = d.MipLevels,
            .arrayLayers           = d.ArrayLayers,
            .samples               = static_cast<VkSampleCountFlagBits>(d.SampleCount),
            .tiling                = VK_IMAGE_TILING_OPTIMAL,
            .usage                 = _toImageUsage(d.Usage),
            .sharingMode           = queueFamilyCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = queueFamilyCount > 1 ? queueFamilyCount : 0,
            .pQueueFamilyIndices   = queueFamilyCount > 1 ? queueFamilies.data() : nullptr,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }

    void VulkanDevice::_finishTexture(uint32_t idx, const TextureDesc &d, VkImageViewType viewType, VkFormat format,
                                      VkImageAspectFlags aspect)
    {
        auto &rec{_textures.get(idx)};
        // Full-resource default image view — only images that are actually
        // viewed (sampled / storage / attachment) can have one; a transfer-only
        // staging image has no view-compatible usage and would fail validation.
        const bool needsView{HasUsage(d.Usage, TextureUsage::Sampled) || HasUsage(d.Usage, TextureUsage::Storage) ||
                             HasUsage(d.Usage, TextureUsage::RenderTarget) ||
                             HasUsage(d.Usage, TextureUsage::DepthStencil)};
        if (needsView)
        {
            VkImageViewCreateInfo vci{
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = rec.image,
                .viewType = viewType,
                .format   = format,
                .subresourceRange =
                    {
                        .aspectMask     = aspect,
                        .baseMipLevel   = 0,
                        .levelCount     = d.MipLevels,
                        .baseArrayLayer = 0,
                        .layerCount     = d.ArrayLayers,
                    },
            };
            VK_CHECK(vkCreateImageView(_device, &vci, nullptr, &rec.view));
        }

        // Register sampled textures in bindless heap
        if (HasUsage(d.Usage, TextureUsage::Sampled))
        {
            _heap.RegisterTexture(idx, rec.view);
        }

        if (!d.DebugName.empty())
        {
            _setDebugName(VK_OBJECT_TYPE_IMAGE, (uint64_t)rec.image, d.DebugName);
            if (rec.view != VK_NULL_HANDLE)
            {
                _setDebugName(VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)rec.view, std::string{d.DebugName} + ":view");
            }
        }
    }

    std::expected<TextureHandle, DeviceError> VulkanDevice::CreateTexture(const TextureDesc &d)
    {
        std::array<uint32_t, 3> queueFamilies{};
        VkImageViewType         viewType{VK_IMAGE_VIEW_TYPE_2D};
        VkImageAspectFlags      aspect{};
        const VkImageCreateInfo ici{_imageCreateInfo(d, queueFamilies, viewType, aspect)};
        VmaAllocationCreateInfo aci{.usage = VMA_MEMORY_USAGE_GPU_ONLY};

        uint32_t idx{_textures.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many textures as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_textures.get(idx)};
        rec.desc = d;
        VK_CHECK(vmaCreateImage(_allocator, &ici, &aci, &rec.image, &rec.alloc, nullptr));

        _finishTexture(idx, d, viewType, ici.format, aspect);
        return TextureHandle{idx};
    }

    std::expected<TextureHandle, PlacementError> VulkanDevice::CreateTexture(const TextureDesc   &d,
                                                                             const HeapPlacement &placement)
    {
        if (!placement.Heap.Valid())
        {
            return std::unexpected(PlacementError::InvalidHeap);
        }
        auto &memoryHeap{_memoryHeaps.get(placement.Heap.Index)};
        if (memoryHeap.memory == VK_NULL_HANDLE)
        {
            return std::unexpected(PlacementError::InvalidHeap);
        }
        // The layout this adapter can place in the heap's memory: the fast one where it fits, the
        // linear fallback where it does not.
        const PlacementPlan plan{_planPlacement(d, memoryHeap.type)};
        if (!plan.viable)
        {
            return std::unexpected(PlacementError::IncompatibleMemory);
        }

        std::array<uint32_t, 3> queueFamilies{};
        VkImageViewType         viewType{VK_IMAGE_VIEW_TYPE_2D};
        VkImageAspectFlags      aspect{};
        VkImageCreateInfo       ici{_imageCreateInfo(d, queueFamilies, viewType, aspect)};
        ici.tiling = plan.tiling;

        VkImage image{VK_NULL_HANDLE};
        VK_CHECK(vkCreateImage(_device, &ici, nullptr, &image));

        const auto refuse{[&](PlacementError error) -> std::expected<TextureHandle, PlacementError>
                          {
                              vkDestroyImage(_device, image, nullptr);
                              return std::unexpected(error);
                          }};

        if (plan.alignment == 0 || placement.Offset % plan.alignment != 0)
        {
            return refuse(PlacementError::MisalignedOffset);
        }
        const uint64_t end{placement.Offset + plan.size};
        if (end > memoryHeap.size || end < placement.Offset)
        {
            return refuse(PlacementError::OutOfRange);
        }

        // Refuse a range that a live placement already covers. `upper_bound` names the first
        // placement starting after this one, so only its predecessor and itself can overlap.
        const auto next{memoryHeap.liveRanges.upper_bound(placement.Offset)};
        if (next != memoryHeap.liveRanges.end() && next->first < end)
        {
            return refuse(PlacementError::Overlapping);
        }
        if (next != memoryHeap.liveRanges.begin() && std::prev(next)->second > placement.Offset)
        {
            return refuse(PlacementError::Overlapping);
        }

        if (vkBindImageMemory(_device, image, memoryHeap.memory, placement.Offset) != VK_SUCCESS)
        {
            return refuse(PlacementError::CreationFailed);
        }

        uint32_t idx{_textures.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many textures as it can");
            return refuse(PlacementError::CreationFailed);
        }
        auto &rec{_textures.get(idx)};
        rec.desc   = d;
        rec.image  = image;
        rec.heap   = placement.Heap;
        rec.offset = placement.Offset;

        memoryHeap.liveRanges.emplace(placement.Offset, end);
        _finishTexture(idx, d, viewType, ici.format, aspect);
        return TextureHandle{idx};
    }

    void VulkanDevice::DestroyTexture(TextureHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &rec{_textures.get(h.Index)};
        if (HasUsage(rec.desc.Usage, TextureUsage::Sampled))
        {
            _heap.UnregisterTexture(h.Index);
        }
        if (rec.borrowed)
        {
            // The swapchain owns this image and its view; the slot is all that is released here.
        }
        else if (rec.heap.Valid())
        {
            vkDestroyImageView(_device, rec.view, nullptr);
            _memoryHeaps.get(rec.heap.Index).liveRanges.erase(rec.offset);
            vkDestroyImage(_device, rec.image, nullptr);
        }
        else
        {
            vkDestroyImageView(_device, rec.view, nullptr);
            vmaDestroyImage(_allocator, rec.image, rec.alloc);
        }
        _textures.free(h.Index);
    }

    GpuAddress VulkanDevice::TextureAddress(TextureHandle h) const
    {
        // Unlike Metal (a real gpuResourceID) or buffers/AS (a real device
        // address), Vulkan's bindless textures are reached through plain
        // descriptor indexing — the handle's own heap slot index IS the
        // value a DescriptorHandle<Texture2D> needs. The matching sampled
        // image descriptor is written by RegisterTexture.
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{h.Index};
    }

    std::expected<SamplerHandle, DeviceError> VulkanDevice::CreateSampler(const SamplerDesc &d)
    {
        VkSamplerCreateInfo sci{toVkSamplerCreateInfo(d)};
        VkSampler           smp{};
        VK_CHECK(vkCreateSampler(_device, &sci, nullptr, &smp));
        uint32_t idx{_samplers.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many samplers as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        _samplers.get(idx).sampler = smp;
        _heap.RegisterSampler(idx, smp);
        return SamplerHandle{idx};
    }

    void VulkanDevice::DestroySampler(SamplerHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        _heap.UnregisterSampler(h.Index);
        vkDestroySampler(_device, _samplers.get(h.Index).sampler, nullptr);
        _samplers.free(h.Index);
    }

    GpuAddress VulkanDevice::SamplerAddress(SamplerHandle h) const
    {
        // Same descriptor-indexing mechanism as TextureAddress.
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{h.Index};
    }

    // ============================================================================
    // Timestamp queries
    // ============================================================================

    bool VulkanDevice::SupportsComputeTimestamps() const noexcept
    {
        return _computeTimestampsSupported;
    }

    double VulkanDevice::TimestampPeriodNanoseconds() const noexcept
    {
        return _timestampPeriodNanoseconds;
    }

    std::expected<TimestampQueryPoolHandle, DeviceError> VulkanDevice::CreateTimestampQueryPool(uint32_t count)
    {
        if (count == 0)
        {
            return {};
        }

        VkQueryPoolCreateInfo createInfo{
            .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType  = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = count,
        };
        const uint32_t index{_timestampQueryPools.alloc()};
        if (index == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many timestamp query pools as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &record{_timestampQueryPools.get(index)};
        VK_CHECK(vkCreateQueryPool(_device, &createInfo, nullptr, &record.pool));
        record.count = count;
        return TimestampQueryPoolHandle{index};
    }

    void VulkanDevice::DestroyTimestampQueryPool(TimestampQueryPoolHandle pool)
    {
        if (!pool.Valid())
        {
            return;
        }
        vkDestroyQueryPool(_device, _timestampQueryPools.get(pool.Index).pool, nullptr);
        _timestampQueryPools.free(pool.Index);
    }

    void VulkanDevice::ResetTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, uint32_t count)
    {
        if (!pool.Valid() || count == 0)
        {
            return;
        }
        const auto &record{_timestampQueryPools.get(pool.Index)};
        assert(first <= record.count && count <= record.count - first);
        vkResetQueryPool(_device, record.pool, first, count);
    }

    void VulkanDevice::ReadTimestampQueries(TimestampQueryPoolHandle pool, uint32_t first, std::span<uint64_t> results)
    {
        if (!pool.Valid() || results.empty())
        {
            return;
        }
        const auto &record{_timestampQueryPools.get(pool.Index)};
        assert(first <= record.count && results.size() <= record.count - first);
        VK_CHECK(vkGetQueryPoolResults(_device, record.pool, first, static_cast<uint32_t>(results.size()),
                                       results.size_bytes(), results.data(), sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    }

    // ============================================================================
    // Pipeline creation
    // ============================================================================

    VkShaderModule VulkanDevice::_makeShaderModule(std::span<const uint32_t> spirv)
    {
        // Word-alignment of the source bytes is validated once, upstream, in
        // rhi::ToShaderDesc — by the time bytes reach here as a uint32_t span,
        // size_bytes() % 4 == 0 is structurally guaranteed by the element type.
        if (spirv.empty())
        {
            return VK_NULL_HANDLE;
        }
        VkShaderModuleCreateInfo ci{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.size_bytes(),
            .pCode    = spirv.data(),
        };
        VkShaderModule m{};
        VK_CHECK(vkCreateShaderModule(_device, &ci, nullptr, &m));
        return m;
    }

    std::expected<PipelineHandle, DeviceError> VulkanDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc &d)
    {
        if (d.PushConstantBytes > _maxPushConstantBytes)
        {
            return {};
        }
        // Extract SPIR-V spans — std::get_if returns nullptr if the wrong alternative
        // is active (e.g. MetalLib on a misrouted call), which _makeShaderModule
        // handles gracefully by returning VK_NULL_HANDLE for an empty span.
        auto           spirvOf{[](const ShaderDesc &sd) -> std::span<const uint32_t>
                               {
                         auto *p{std::get_if<SpirvBytecode>(&sd.Bytecode)};
                         return p ? p->Words : std::span<const uint32_t>{};
                               }};
        VkShaderModule vs{_makeShaderModule(spirvOf(d.VertexShader))};
        VkShaderModule fs{_makeShaderModule(spirvOf(d.FragmentShader))};

        std::vector<VkPipelineShaderStageCreateInfo> stages{};
        auto addStage{[&](VkShaderModule m, VkShaderStageFlagBits stage, std::string_view ep)
                      {
                          if (m != VK_NULL_HANDLE)
                          {
                              stages.push_back({.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage  = stage,
                                                .module = m,
                                                .pName  = ep.empty() ? "main" : ep.data()});
                          }
                      }};
        addStage(vs, VK_SHADER_STAGE_VERTEX_BIT, d.VertexShader.EntryPoint);
        addStage(fs, VK_SHADER_STAGE_FRAGMENT_BIT, d.FragmentShader.EntryPoint);

        // Bindless-only pipeline: no vertex input bindings
        VkPipelineVertexInputStateCreateInfo vi{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

        // Order MUST match rhi::PrimitiveTopology (TriangleList, TriangleStrip,
        // LineList, LineStrip, PointList) — indexed by the enum value below.
        static const VkPrimitiveTopology kTopTbl[]{
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,    VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        };
        VkPipelineInputAssemblyStateCreateInfo ia{
            .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = kTopTbl[static_cast<int>(d.Topology)],
        };

        static const VkDynamicState       kDyn[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo  dyn{.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                              .dynamicStateCount = 2,
                                              .pDynamicStates    = kDyn};
        VkPipelineViewportStateCreateInfo vp{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};

        auto                                   toCull{[](CullMode m) -> VkCullModeFlags
                                                      {
                        switch (m)
                        {
                            case CullMode::None:
                                return VK_CULL_MODE_NONE;
                            case CullMode::Front:
                                return VK_CULL_MODE_FRONT_BIT;
                            case CullMode::Back:
                                return VK_CULL_MODE_BACK_BIT;
                        }
                        return VK_CULL_MODE_BACK_BIT;
                                                      }};
        VkPipelineRasterizationStateCreateInfo rast{
            .sType            = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = d.Rasterizer.DepthClamp,
            .polygonMode = d.Rasterizer.FillMode == FillMode::Wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL,
            .cullMode    = toCull(d.Rasterizer.CullMode),
            .frontFace   = d.Rasterizer.FrontFace == FrontFace::CounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                                 : VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable         = d.Rasterizer.DepthBiasConstant != 0.f || d.Rasterizer.DepthBiasSlope != 0.f,
            .depthBiasConstantFactor = d.Rasterizer.DepthBiasConstant,
            .depthBiasSlopeFactor    = d.Rasterizer.DepthBiasSlope,
            .lineWidth               = 1.f,
        };

        VkPipelineMultisampleStateCreateInfo ms{
            .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = static_cast<VkSampleCountFlagBits>(d.SampleCount),
        };

        auto                                  toCmpOp{[](CompareOp op) -> VkCompareOp
                                                      {
                         switch (op)
                         {
                             case CompareOp::Never:
                                 return VK_COMPARE_OP_NEVER;
                             case CompareOp::Less:
                                 return VK_COMPARE_OP_LESS;
                             case CompareOp::Equal:
                                 return VK_COMPARE_OP_EQUAL;
                             case CompareOp::LessEqual:
                                 return VK_COMPARE_OP_LESS_OR_EQUAL;
                             case CompareOp::Greater:
                                 return VK_COMPARE_OP_GREATER;
                             case CompareOp::NotEqual:
                                 return VK_COMPARE_OP_NOT_EQUAL;
                             case CompareOp::GreaterEqual:
                                 return VK_COMPARE_OP_GREATER_OR_EQUAL;
                             case CompareOp::Always:
                                 return VK_COMPARE_OP_ALWAYS;
                         }
                         return VK_COMPARE_OP_LESS;
                                                      }};
        VkPipelineDepthStencilStateCreateInfo ds{
            .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable  = d.DepthStencil.DepthTest,
            .depthWriteEnable = d.DepthStencil.DepthWrite,
            .depthCompareOp   = toCmpOp(d.DepthStencil.DepthOp),
        };

        auto                                             toBF{[](BlendFactor f) -> VkBlendFactor
                                                              {
                      switch (f)
                      {
                          case BlendFactor::Zero:
                              return VK_BLEND_FACTOR_ZERO;
                          case BlendFactor::One:
                              return VK_BLEND_FACTOR_ONE;
                          case BlendFactor::SrcColor:
                              return VK_BLEND_FACTOR_SRC_COLOR;
                          case BlendFactor::OneMinusSrcColor:
                              return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                          case BlendFactor::DstColor:
                              return VK_BLEND_FACTOR_DST_COLOR;
                          case BlendFactor::OneMinusDstColor:
                              return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
                          case BlendFactor::SrcAlpha:
                              return VK_BLEND_FACTOR_SRC_ALPHA;
                          case BlendFactor::OneMinusSrcAlpha:
                              return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                          case BlendFactor::DstAlpha:
                              return VK_BLEND_FACTOR_DST_ALPHA;
                          case BlendFactor::OneMinusDstAlpha:
                              return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
                          case BlendFactor::SrcAlphaSaturate:
                              return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
                          default:
                              return VK_BLEND_FACTOR_ONE;
                      }
                                                              }};
        auto                                             toBO{[](BlendOp o) -> VkBlendOp
                                                              {
                      switch (o)
                      {
                          case BlendOp::Add:
                              return VK_BLEND_OP_ADD;
                          case BlendOp::Subtract:
                              return VK_BLEND_OP_SUBTRACT;
                          case BlendOp::ReverseSubtract:
                              return VK_BLEND_OP_REVERSE_SUBTRACT;
                          case BlendOp::Min:
                              return VK_BLEND_OP_MIN;
                          case BlendOp::Max:
                              return VK_BLEND_OP_MAX;
                      }
                      return VK_BLEND_OP_ADD;
                                                              }};
        std::vector<VkPipelineColorBlendAttachmentState> blendAtts{};
        blendAtts.reserve(d.ColorBlend.size());
        for (const auto &b : d.ColorBlend)
        {
            blendAtts.push_back({
                .blendEnable         = b.Enable,
                .srcColorBlendFactor = toBF(b.SrcColor),
                .dstColorBlendFactor = toBF(b.DstColor),
                .colorBlendOp        = toBO(b.ColorOp),
                .srcAlphaBlendFactor = toBF(b.SrcAlpha),
                .dstAlphaBlendFactor = toBF(b.DstAlpha),
                .alphaBlendOp        = toBO(b.AlphaOp),
                .colorWriteMask      = b.WriteMask,
            });
        }

        VkPipelineColorBlendStateCreateInfo blend{
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = static_cast<uint32_t>(blendAtts.size()),
            .pAttachments    = blendAtts.data(),
        };

        // Dynamic rendering — replaces VkRenderPass
        std::vector<VkFormat> colorFmts{};
        colorFmts.reserve(d.ColorFormats.size());
        for (Format f : d.ColorFormats)
        {
            colorFmts.push_back(toVkFormat(f));
        }

        VkPipelineRenderingCreateInfo rendCI{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount    = static_cast<uint32_t>(colorFmts.size()),
            .pColorAttachmentFormats = colorFmts.data(),
            .depthAttachmentFormat =
                d.DepthFormat != Format::Undefined ? toVkFormat(d.DepthFormat) : VK_FORMAT_UNDEFINED,
        };

        VkGraphicsPipelineCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendCI,
            // The command buffer binds the bindless heap as a descriptor buffer,
            // so every pipeline must opt into descriptor-buffer descriptor sets.
            .flags               = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
            .stageCount          = static_cast<uint32_t>(stages.size()),
            .pStages             = stages.data(),
            .pVertexInputState   = &vi,
            .pInputAssemblyState = &ia,
            .pViewportState      = &vp,
            .pRasterizationState = &rast,
            .pMultisampleState   = &ms,
            .pDepthStencilState  = &ds,
            .pColorBlendState    = &blend,
            .pDynamicState       = &dyn,
            .layout              = _globalLayout,
            .renderPass          = VK_NULL_HANDLE, // dynamic rendering = no render pass
        };

        uint32_t idx{_pipelines.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many pipelines as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_pipelines.get(idx)};
        rec.isCompute         = false;
        rec.pushConstantBytes = d.PushConstantBytes;
        VK_CHECK(vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pci, nullptr, &rec.pipeline));

        // Shader modules are no longer needed after pipeline compilation
        if (vs)
        {
            vkDestroyShaderModule(_device, vs, nullptr);
        }
        if (fs)
        {
            vkDestroyShaderModule(_device, fs, nullptr);
        }

        if (!d.DebugName.empty())
        {
            _setDebugName(VK_OBJECT_TYPE_PIPELINE, (uint64_t)rec.pipeline, d.DebugName);
        }

        return PipelineHandle{idx};
    }

    std::expected<PipelineHandle, DeviceError> VulkanDevice::CreateComputePipeline(const ComputePipelineDesc &d)
    {
        if (d.PushConstantBytes > _maxPushConstantBytes)
        {
            return {};
        }
        auto          *spirvPtr{std::get_if<SpirvBytecode>(&d.Shader.Bytecode)};
        VkShaderModule cs{spirvPtr ? _makeShaderModule(spirvPtr->Words) : VK_NULL_HANDLE};
        if (!cs)
        {
            return PipelineHandle{};
        }

        VkPipelineShaderStageCreateInfo stage{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = cs,
            .pName  = d.Shader.EntryPoint.empty() ? "main" : d.Shader.EntryPoint.data(),
        };
        VkComputePipelineCreateInfo pci{
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags  = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
            .stage  = stage,
            .layout = _globalLayout,
        };

        uint32_t idx{_pipelines.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many pipelines as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_pipelines.get(idx)};
        rec.isCompute         = true;
        rec.pushConstantBytes = d.PushConstantBytes;
        VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pci, nullptr, &rec.pipeline));
        vkDestroyShaderModule(_device, cs, nullptr);

        if (!d.DebugName.empty())
        {
            _setDebugName(VK_OBJECT_TYPE_PIPELINE, (uint64_t)rec.pipeline, d.DebugName);
        }

        return PipelineHandle{idx};
    }

    void VulkanDevice::DestroyPipeline(PipelineHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        vkDestroyPipeline(_device, _pipelines.get(h.Index).pipeline, nullptr);
        _pipelines.free(h.Index);
    }

    // ============================================================================
    // Ray tracing
    // ============================================================================

    void VulkanDevice::BuildAccelGeometryInfo(const AccelerationStructureDesc             &desc,
                                              VkDeviceAddress                              instanceBufferAddress,
                                              VkAccelerationStructureGeometryKHR          &outGeometry,
                                              VkAccelerationStructureBuildGeometryInfoKHR &outBuildInfo,
                                              uint32_t                                    &outPrimitiveCount) const
    {
        outGeometry =
            VkAccelerationStructureGeometryKHR{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};

        VkAccelerationStructureTypeKHR asType{};
        if (desc.Type == AccelerationStructureType::BottomLevel)
        {
            asType                         = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            outGeometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            outGeometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
            outGeometry.geometry.triangles = VkAccelerationStructureGeometryTrianglesDataKHR{
                .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                .vertexData   = {.deviceAddress = desc.VertexBufferAddress.Address},
                .vertexStride = desc.VertexStride,
                .maxVertex    = desc.VertexCount > 0 ? desc.VertexCount - 1 : 0,
                .indexType = desc.IndexBufferAddress.Valid()
                                 ? (desc.IndexType == IndexType::Uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32)
                                 : VK_INDEX_TYPE_NONE_KHR,
                .indexData = {.deviceAddress = desc.IndexBufferAddress.Address},
            };
            outPrimitiveCount = desc.IndexBufferAddress.Valid() ? desc.IndexCount / 3 : desc.VertexCount / 3;
        }
        else
        {
            asType                         = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            outGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            outGeometry.geometry.instances = VkAccelerationStructureGeometryInstancesDataKHR{
                .sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                .arrayOfPointers = VK_FALSE,
                .data            = {.deviceAddress = instanceBufferAddress},
            };
            outPrimitiveCount = static_cast<uint32_t>(desc.Instances.size());
        }

        outBuildInfo = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type  = asType,
            .flags = static_cast<VkBuildAccelerationStructureFlagsKHR>(
                desc.PreferFastTrace ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                     : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR),
            .mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = 1,
            .pGeometries   = &outGeometry,
        };
    }

    bool VulkanDevice::SupportsRayTracing() const noexcept
    {
        return _raytracingSupported;
    }

    AccelerationStructureBuildSizes
    VulkanDevice::QueryAccelerationStructureBuildSizes(const AccelerationStructureDesc &desc) const
    {
        if (!_raytracingSupported)
        {
            FailContract("QueryAccelerationStructureBuildSizes: device does not "
                         "support ray tracing (VK_KHR_acceleration_structure / VK_KHR_ray_query "
                         "unavailable)");
        }

        VkAccelerationStructureGeometryKHR          geometry{};
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        uint32_t                                    primCount{0};
        // instanceBufferAddress=0: the build-size query never dereferences
        // geometry data, only counts (primitiveCount below) — a real address
        // is only required for the actual build (ICommandList::BuildAccelerationStructure).
        BuildAccelGeometryInfo(desc, /*instanceBufferAddress=*/0, geometry, buildInfo, primCount);

        VkAccelerationStructureBuildSizesInfoKHR sizes{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_GetAccelerationStructureBuildSizes(_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
                                               &primCount, &sizes);
        return AccelerationStructureBuildSizes{
            .AccelerationStructureSize = sizes.accelerationStructureSize,
            .BuildScratchSize          = sizes.buildScratchSize,
            .UpdateScratchSize         = sizes.updateScratchSize,
        };
    }

    std::expected<AccelerationStructureHandle, DeviceError>
    VulkanDevice::CreateAccelerationStructure(const AccelerationStructureDesc &desc)
    {
        if (!_raytracingSupported)
        {
            FailContract("CreateAccelerationStructure: device does not support ray "
                         "tracing (VK_KHR_acceleration_structure / VK_KHR_ray_query unavailable)");
        }

        auto sizes{QueryAccelerationStructureBuildSizes(desc)};

        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = sizes.AccelerationStructureSize,
            .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        };
        VmaAllocationCreateInfo aci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

        uint32_t idx{_accelStructs.alloc()};
        if (idx == kInvalidIndex)
        {
            ReportRefusal("the device is holding as many acceleration structures as it can");
            return std::unexpected{DeviceError::Exhausted};
        }
        auto &rec{_accelStructs.get(idx)};
        rec.size = sizes.AccelerationStructureSize;
        VK_CHECK(vmaCreateBuffer(_allocator, &bci, &aci, &rec.buffer, &rec.alloc, nullptr));

        VkAccelerationStructureCreateInfoKHR ci{
            .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = rec.buffer,
            .size   = sizes.AccelerationStructureSize,
            .type   = desc.Type == AccelerationStructureType::BottomLevel
                          ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
                          : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        };
        VK_CHECK(pfn_CreateAccelerationStructure(_device, &ci, nullptr, &rec.as));

        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{
            .sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = rec.as,
        };
        rec.deviceAddress = pfn_GetAccelerationStructureDeviceAddress(_device, &addrInfo);

        if (!desc.DebugName.empty())
        {
            _setDebugName(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, (uint64_t)rec.as, desc.DebugName);
        }

        return AccelerationStructureHandle{idx};
    }

    void VulkanDevice::DestroyAccelerationStructure(AccelerationStructureHandle h)
    {
        if (!h.Valid())
        {
            return;
        }
        auto &rec{_accelStructs.get(h.Index)};
        pfn_DestroyAccelerationStructure(_device, rec.as, nullptr);
        vmaDestroyBuffer(_allocator, rec.buffer, rec.alloc);
        _accelStructs.free(h.Index);
    }

    GpuAddress VulkanDevice::AccelerationStructureAddress(AccelerationStructureHandle h) const
    {
        // The AS's bindless GPU handle: its device address
        // (vkGetAccelerationStructureDeviceAddressKHR, cached at create).
        // Shaders receive it in push constants as a
        // DescriptorHandle<RaytracingAccelerationStructure> field, which
        // SPIR-V converts back via OpConvertUToAccelerationStructureKHR —
        // see the doc comment in device.cppm.
        if (!h.Valid())
        {
            return {};
        }
        return GpuAddress{_accelStructs.get(h.Index).deviceAddress};
    }

    // ============================================================================
    // Submission / sync
    // ============================================================================

    void VulkanDevice::WaitForFence(FenceHandle f)
    {
        if (!f.Valid())
        {
            return;
        }
        VkSemaphoreWaitInfo wi{
            .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores    = &_timeline,
            .pValues        = &f.Id,
        };
        VK_CHECK(vkWaitSemaphores(_device, &wi, UINT64_MAX));
    }

    bool VulkanDevice::IsFenceComplete(FenceHandle f)
    {
        if (!f.Valid())
        {
            return true;
        }
        uint64_t val{0};
        vkGetSemaphoreCounterValue(_device, _timeline, &val);
        return val >= f.Id;
    }

    void VulkanDevice::WaitIdle()
    {
        vkDeviceWaitIdle(_device);
    }

    // ============================================================================
    // Upload helpers
    // ============================================================================

    void VulkanDevice::UploadBuffer(BufferHandle dst, std::span<const std::byte> data, uint64_t dstOffset)
    {
        if (!dst.Valid() || data.empty())
        {
            return;
        }
        const uint64_t size{data.size_bytes()};

        // Create a transient CPU-visible staging buffer
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        };
        VmaAllocationCreateInfo aci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        VmaAllocationInfo ai{};
        VkBuffer          staging{};
        VmaAllocation     stagingAlloc{};
        VK_CHECK(vmaCreateBuffer(_allocator, &bci, &aci, &staging, &stagingAlloc, &ai));
        std::memcpy(ai.pMappedData, data.data(), size);
        VK_CHECK(vmaFlushAllocation(_allocator, stagingAlloc, 0, VK_WHOLE_SIZE));

        // Record and Submit copy
        VkCommandBufferAllocateInfo cbai{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = xferPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer cmd{};
        {
            std::scoped_lock lk{xferPoolMtx};
            VK_CHECK(vkAllocateCommandBuffers(_device, &cbai, &cmd));
        }

        VkCommandBufferBeginInfo bi{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &bi);
        VkBufferCopy2 copy{
            .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = 0,
            .dstOffset = dstOffset,
            .size      = size,
        };
        VkCopyBufferInfo2 cbi{
            .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer   = staging,
            .dstBuffer   = _buffers.get(dst.Index).buffer,
            .regionCount = 1,
            .pRegions    = &copy,
        };
        vkCmdCopyBuffer2(cmd, &cbi);
        vkEndCommandBuffer(cmd);

        // Submit and wait on the timeline
        uint64_t signalVal{};
        {
            std::scoped_lock lk{_timelineMtx};
            signalVal = ++_timelineValue;
        }
        VkCommandBufferSubmitInfo csi{
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = cmd,
        };
        VkSemaphoreSubmitInfo ssi{
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = _timeline,
            .value     = signalVal,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        VkSubmitInfo2 si{
            .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .commandBufferInfoCount   = 1,
            .pCommandBufferInfos      = &csi,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos    = &ssi,
        };
        {
            std::scoped_lock lk{transferQ.mtx};
            VK_CHECK(vkQueueSubmit2(transferQ.q, 1, &si, VK_NULL_HANDLE));
        }
        VkSemaphoreWaitInfo wi{
            .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores    = &_timeline,
            .pValues        = &signalVal,
        };
        VK_CHECK(vkWaitSemaphores(_device, &wi, UINT64_MAX));

        {
            std::scoped_lock lk{xferPoolMtx};
            vkFreeCommandBuffers(_device, xferPool, 1, &cmd);
        }
        vmaDestroyBuffer(_allocator, staging, stagingAlloc);
    }

    void VulkanDevice::UploadTexture(TextureHandle dst, std::span<const std::byte> data, uint64_t rowPitch,
                                     uint64_t slicePitch, const TextureCopyRegion &region)
    {
        if (!dst.Valid() || data.empty())
        {
            return;
        }
        auto          &tex{_textures.get(dst.Index)};
        const uint64_t totalSize{slicePitch > 0 ? slicePitch : rowPitch * region.Extent.Height * region.Extent.Depth};
        if (data.size_bytes() < totalSize)
        {
            FailContract("UploadTexture: the source holds " + std::to_string(data.size_bytes()) +
                         " bytes but the region reads " + std::to_string(totalSize));
        }

        // Staging buffer
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = totalSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        };
        VmaAllocationCreateInfo aci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        VmaAllocationInfo ai{};
        VkBuffer          staging{};
        VmaAllocation     stagingAlloc{};
        VK_CHECK(vmaCreateBuffer(_allocator, &bci, &aci, &staging, &stagingAlloc, &ai));
        std::memcpy(ai.pMappedData, data.data(), totalSize);
        VK_CHECK(vmaFlushAllocation(_allocator, stagingAlloc, 0, VK_WHOLE_SIZE));

        // Blit into the image
        VkCommandBufferAllocateInfo cbai{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = xferPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer cmd{};
        {
            std::scoped_lock lk{xferPoolMtx};
            VK_CHECK(vkAllocateCommandBuffers(_device, &cbai, &cmd));
        }

        VkCommandBufferBeginInfo cbbi{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &cbbi);

        // Transition to TRANSFER_DST
        VkImageMemoryBarrier2 toXfer{
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = tex.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                                           : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask       = tex.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                       ? 0
                                       : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout           = tex.layout,
            .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = tex.image,
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, region.MipLevel, 1, region.ArrayLayer, 1},
        };
        VkDependencyInfo dep{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &toXfer,
        };
        vkCmdPipelineBarrier2(cmd, &dep);

        VkBufferImageCopy2 cp{
            .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, region.MipLevel, region.ArrayLayer, 1},
            .imageOffset       = {region.DstOffset.X, region.DstOffset.Y, region.DstOffset.Z},
            .imageExtent = {region.Extent.Width, region.Extent.Height, region.Extent.Depth ? region.Extent.Depth : 1u},
        };
        VkCopyBufferToImageInfo2 cpInfo{
            .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .srcBuffer      = staging,
            .dstImage       = tex.image,
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount    = 1,
            .pRegions       = &cp,
        };
        vkCmdCopyBufferToImage2(cmd, &cpInfo);

        vkEndCommandBuffer(cmd);

        uint64_t signalVal{};
        {
            std::scoped_lock lk{_timelineMtx};
            signalVal = ++_timelineValue;
        }

        VkCommandBufferSubmitInfo csi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd};
        VkSemaphoreSubmitInfo     ssi{.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                      .semaphore = _timeline,
                                      .value     = signalVal,
                                      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        VkSubmitInfo2             si{.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                     .commandBufferInfoCount   = 1,
                                     .pCommandBufferInfos      = &csi,
                                     .signalSemaphoreInfoCount = 1,
                                     .pSignalSemaphoreInfos    = &ssi};
        {
            std::scoped_lock lk{transferQ.mtx};
            VK_CHECK(vkQueueSubmit2(transferQ.q, 1, &si, VK_NULL_HANDLE));
        }

        VkSemaphoreWaitInfo wi{.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                               .semaphoreCount = 1,
                               .pSemaphores    = &_timeline,
                               .pValues        = &signalVal};
        VK_CHECK(vkWaitSemaphores(_device, &wi, UINT64_MAX));

        {
            std::scoped_lock lk{xferPoolMtx};
            vkFreeCommandBuffers(_device, xferPool, 1, &cmd);
        }
        vmaDestroyBuffer(_allocator, staging, stagingAlloc);
        tex.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }

    // ============================================================================
    // Debug naming
    // ============================================================================

    void VulkanDevice::_setDebugName(VkObjectType type, uint64_t handle, std::string_view name)
    {
        if (!_debugMessenger || name.empty())
        {
            return;
        }
        VkDebugUtilsObjectNameInfoEXT ni{
            .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType   = type,
            .objectHandle = handle,
            .pObjectName  = name.data(),
        };
        auto fn{(PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(_device, "vkSetDebugUtilsObjectNameEXT")};
        if (fn)
        {
            fn(_device, &ni);
        }
    }

} // namespace rhi::vulkan
