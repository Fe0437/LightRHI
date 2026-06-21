// fullscreen.slang
// Fullscreen triangle vertex shader + solid-colour fragment variants.
// Used by: render_triangle_readback, multiple_render_targets, draw_indirect.

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> g_bufAddresses;
[[vk::binding(1, 0)]] StructuredBuffer<uint64_t> g_texIds;
[[vk::binding(2, 0)]] StructuredBuffer<uint64_t> g_smpIds;

struct VertOut { float4 pos : SV_Position; };

// Fullscreen triangle: covers the entire clip-space quad with three vertices.
[shader("vertex")]
VertOut vert_main(uint vid : SV_VertexID)
{
    float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2( 3.0f, -1.0f),
        float2(-1.0f,  3.0f),
    };
    VertOut o;
    o.pos = float4(positions[vid], 0.0f, 1.0f);
    return o;
}

[shader("fragment")]
float4 red_frag(VertOut i) : SV_Target { return float4(1.0f, 0.0f, 0.0f, 1.0f); }

[shader("fragment")]
float4 green_frag(VertOut i) : SV_Target { return float4(0.0f, 1.0f, 0.0f, 1.0f); }
