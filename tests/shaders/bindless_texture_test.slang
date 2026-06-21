// bindless_texture_test.slang
// Proves a sampled texture is fully bindless: the command list receives no
// texture binding. The push constant carries DescriptorHandle<Texture2D>.
// Metal stores the resident texture's GPU resource ID directly in that
// handle; Vulkan stores the sampled-image descriptor-array index.

struct PC
{
    uint64_t outputAddr;
    DescriptorHandle<Texture2D> texture0;
    DescriptorHandle<Texture2D> texture1;
};
// register(b30) forces Metal buffer(30) — matches kPushConstantSlot in the backend.
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);

[shader("compute")]
[numthreads(8, 1, 1)]
void bindless_texture_test_main(uint3 tid : SV_DispatchThreadID)
{
    uint textureIndex = tid.x >> 2;
    Texture2D texture = textureIndex == 0 ? g_pc.texture0 : g_pc.texture1;
    uint texelIndex = tid.x & 3;
    int2 coord = int2(texelIndex & 1, texelIndex >> 1);
    float4 color = texture.Load(int3(coord, 0));

    float *out = (float *)g_pc.outputAddr;
    out[tid.x * 4 + 0] = color.x;
    out[tid.x * 4 + 1] = color.y;
    out[tid.x * 4 + 2] = color.z;
    out[tid.x * 4 + 3] = color.w;
}
