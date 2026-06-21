// compute_seq.slang
// Two sequential compute kernels for the barrier test.
// fill_a writes 0xAAAAAAAA; xor_b XORs with 0x55555555 → result 0xFFFFFFFF.
// A UAV barrier between them is inserted by the test to prove ordering.

struct PC {
    uint64_t bufAddr;
    uint     count;
};

[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);  // Vulkan push constant; register(b30) pins Metal buffer(30)

[shader("compute")]
[numthreads(64, 1, 1)]
void fill_a(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= g_pc.count) return;
    uint* buf = (uint*)g_pc.bufAddr;
    buf[tid.x] = 0xAAAAAAAAu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void xor_b(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= g_pc.count) return;
    uint* buf = (uint*)g_pc.bufAddr;
    buf[tid.x] ^= 0x55555555u;
}
