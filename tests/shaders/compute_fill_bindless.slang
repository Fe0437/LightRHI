// compute_fill_bindless.slang
// Compute kernel: looks up a buffer's GPU address from the bindless table using
// a slot index from push constants, then writes a fixed pattern through it.

// Set 1 is Slang's system bindless space. Bindings 0 and 2 are reserved for
// sampler and resource descriptor arrays respectively.
[[vk::binding(3, 1)]] StructuredBuffer<uint64_t> g_bufAddresses;

struct PC { uint slot; };
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);  // Vulkan push constant; register(b30) pins Metal buffer(30)

[shader("compute")]
[numthreads(64, 1, 1)]
void bindless_fill(uint3 tid : SV_DispatchThreadID)
{
    uint* out = (uint*)g_bufAddresses[g_pc.slot];
    out[tid.x] = 0xCAFEBABEu;
}
