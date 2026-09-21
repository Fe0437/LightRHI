// triangle.slang — minimal bindless graphics pipeline for 02_compute_triangle.
//
// Mirrors the pattern used by the RHI tests (see tests/shaders/*.slang):
//   * set 0, bindings 0/1/2 = the bindless address / texture / sampler tables
//   * LightRHI's reserved push-constant register = the push-constant block
//   * vertices are fetched directly from a raw GPU address (BDA) carried in the
//     push constant — no vertex buffer binding.

#include "light_rhi_shader_abi.slangh"

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> g_bufAddresses;
[[vk::binding(1, 0)]] StructuredBuffer<uint64_t> g_texIds;
[[vk::binding(2, 0)]] StructuredBuffer<uint64_t> g_smpIds;

struct Vertex { float x, y, r, g, b; };

struct PC { uint64_t vertAddr; };
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(LIGHTRHI_PUSH_CONSTANT_REGISTER);

struct V2F
{
    float4 pos : SV_Position;
    float3 col : COLOR0;
};

[shader("vertex")]
V2F vert_main(uint vid : SV_VertexID)
{
    Vertex* verts = (Vertex*)g_pc.vertAddr;
    Vertex  v     = verts[vid];

    V2F o;
    o.pos = float4(v.x, v.y, 0.0f, 1.0f);
    o.col = float3(v.r, v.g, v.b);
    return o;
}

[shader("fragment")]
float4 frag_main(V2F i) : SV_Target
{
    return float4(i.col, 1.0f);
}
