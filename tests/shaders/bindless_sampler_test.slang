// bindless_sampler_test.slang
// Proves a texture is sampled with no binding at all: the push constants carry the texture and the
// sampler as DescriptorHandles, and plain fields after them keep their places.

#include "light_rhi_shader_abi.slangh"

struct PC
{
    float *output;
    DescriptorHandle<Texture2D>    texture;
    DescriptorHandle<SamplerState> sampler;
    float  firstU;
    float  stepU;
};
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(LIGHTRHI_PUSH_CONSTANT_REGISTER);

[shader("compute")]
[numthreads(4, 1, 1)]
void bindless_sampler_test_main(uint3 tid : SV_DispatchThreadID)
{
    const Texture2D    texture = g_pc.texture;
    const SamplerState sampler = g_pc.sampler;
    const float        u       = g_pc.firstU + g_pc.stepU * float(tid.x);
    g_pc.output[tid.x] = texture.SampleLevel(sampler, float2(u, 0.5f), 0.0f).x;
}
