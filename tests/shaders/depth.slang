// depth.slang
// Vertex shader placing a fullscreen triangle at a configurable NDC depth.
// Pairs with red_frag / green_frag from fullscreen.slang.

#include "light_rhi_shader_abi.slangh"

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> g_bufAddresses;
[[vk::binding(1, 0)]] StructuredBuffer<uint64_t> g_texIds;
[[vk::binding(2, 0)]] StructuredBuffer<uint64_t> g_smpIds;

struct PC { float depth; };
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(LIGHTRHI_PUSH_CONSTANT_REGISTER);

struct VOut { float4 pos : SV_Position; };

[shader("vertex")]
VOut depth_vert(uint vid : SV_VertexID)
{
    float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2( 3.0f, -1.0f),
        float2(-1.0f,  3.0f),
    };
    VOut o;
    o.pos = float4(positions[vid], g_pc.depth, 1.0f);
    return o;
}
