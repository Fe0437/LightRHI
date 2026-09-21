// compute_fill_bda.slang
// Compute kernel: writes a value to a buffer via a raw GPU address in push constants.
// Tests the BDA path (uint* cast from uint64_t address).

#include "light_rhi_shader_abi.slangh"

struct PC {
    uint64_t outputAddr;
    uint     value;
    uint     count;
};

[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(LIGHTRHI_PUSH_CONSTANT_REGISTER);

[shader("compute")]
[numthreads(64, 1, 1)]
void fill_via_bda(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= g_pc.count) return;
    uint* out = (uint*)g_pc.outputAddr;
    out[tid.x] = g_pc.value;
}
