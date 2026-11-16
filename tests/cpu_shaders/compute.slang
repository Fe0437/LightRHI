/**
 * {file}
 * {brief} Exercises the CPU shader ABI through bindless sampled and writable textures.
 */
#include "light_rhi_shader_abi.slangh"
#include "light_rhi_workgroup.slangh"

struct Constants
{
    DescriptorHandle<Texture2D<float4>> Source;
    DescriptorHandle<SamplerState> Sampler;
    DescriptorHandle<RWTexture2D<float4>> Target;
    uint *Result;
    uint Value;
};
[[vk::push_constant]] ConstantBuffer<Constants> g_pc : register(LIGHTRHI_PUSH_CONSTANT_REGISTER);

LIGHTRHI_GROUP_ARRAY(uint, exchange, 4);

[shader("compute")]
[numthreads(2, 2, 1)]
void write(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    LIGHTRHI_GROUP_ARRAY_VIEW(uint, exchange, 0, 4);
    exchange[lane] = g_pc.Value + tid.x + 4 * tid.y;
    LightRhiGroupMemoryBarrier();
    g_pc.Result[tid.x + 4 * tid.y] = exchange[3 - lane];
    const Texture2D<float4> source = g_pc.Source;
    const SamplerState sampler = g_pc.Sampler;
    RWTexture2D<float4> target = g_pc.Target;
    target[tid.xy] = source.SampleLevel(sampler, (float2(tid.xy) + 0.5f) / 4.0f, 0.0f);
}

[shader("compute")]
[numthreads(2, 2, 1)]
void increment(uint3 tid : SV_DispatchThreadID)
{
    g_pc.Result[tid.x + 4 * tid.y] += g_pc.Value;
}
