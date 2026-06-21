// mrt.slang
// Fragment shader writing to two render targets simultaneously.
// Vertex stage reuses vert_main from fullscreen.slang.

struct VOut { float4 pos : SV_Position; };

struct MRTOut {
    float4 color0 : SV_Target0;
    float4 color1 : SV_Target1;
};

[shader("fragment")]
MRTOut mrt_frag(VOut i)
{
    MRTOut o;
    o.color0 = float4(1.0f, 0.0f, 0.0f, 1.0f);
    o.color1 = float4(0.0f, 0.0f, 1.0f, 1.0f);
    return o;
}
