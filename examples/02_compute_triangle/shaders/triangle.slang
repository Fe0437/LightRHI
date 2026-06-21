// triangle.slang — minimal bindless graphics pipeline for 02_compute_triangle.
//
// Mirrors the pattern used by the RHI tests (see tests/shaders/*.slang):
//   * set 0, bindings 0/1/2 = the bindless address / texture / sampler tables
//   * set 0, binding 30      = the push-constant block (register(b30) pins the
//                              Metal buffer slot to kPushConstantSlot)
//   * vertices are fetched directly from a raw GPU address (BDA) carried in the
//     push constant — no vertex buffer binding.

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> g_bufAddresses;
[[vk::binding(1, 0)]] StructuredBuffer<uint64_t> g_texIds;
[[vk::binding(2, 0)]] StructuredBuffer<uint64_t> g_smpIds;

struct Vertex { float x, y, r, g, b; };

struct PC { uint64_t vertAddr; };
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);  // Vulkan push constant; register(b30) pins Metal buffer(30)

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
