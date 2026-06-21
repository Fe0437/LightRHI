// quad.slang
// CCW quad vertex shader + blue fragment shader.
// Used by: draw_indexed test (6 uint32 indices, 4 vertices).

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> g_bufAddresses;
[[vk::binding(1, 0)]] StructuredBuffer<uint64_t> g_texIds;
[[vk::binding(2, 0)]] StructuredBuffer<uint64_t> g_smpIds;

struct VertOut { float4 pos : SV_Position; };

[shader("vertex")]
VertOut quad_vert(uint vid : SV_VertexID)
{
    float2 positions[4] = {
        float2(-1.0f, -1.0f),
        float2( 1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f,  1.0f),
    };
    VertOut o;
    o.pos = float4(positions[vid], 0.0f, 1.0f);
    return o;
}

[shader("fragment")]
float4 blue_frag(VertOut i) : SV_Target { return float4(0.0f, 0.0f, 1.0f, 1.0f); }
