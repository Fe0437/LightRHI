// compute_fill_bda.slang
// Compute kernel: writes a value to a buffer via a raw GPU address in push constants.
// Tests the BDA path (uint* cast from uint64_t address).

struct PC {
    uint64_t outputAddr;
    uint     value;
    uint     count;
};

// register(b30) forces Metal buffer(30) — matches kPushConstantSlot in the backend.
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);  // Vulkan push constant; register(b30) pins Metal buffer(30)

[shader("compute")]
[numthreads(64, 1, 1)]
void fill_via_bda(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= g_pc.count) return;
    uint* out = (uint*)g_pc.outputAddr;
    out[tid.x] = g_pc.value;
}
