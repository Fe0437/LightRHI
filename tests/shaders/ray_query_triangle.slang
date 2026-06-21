// ray_query_triangle.slang
// Inline ray-query against a single-triangle scene. Writes the hit distance
// (or -1.0 on miss) through a raw BDA output pointer, exactly like the
// existing compute_fill_bda.slang test shader.
//
// The acceleration structure is fully bindless: the push constants carry a
// DescriptorHandle<RaytracingAccelerationStructure> — an 8-byte GPU handle
// obtained from IDevice::AccelerationStructureAddress(), placed in push
// constants exactly like a buffer BDA pointer. No per-dispatch bind call.
//
//   Vulkan: lowers to a uint64 read + OpConvertUToAccelerationStructureKHR
//           (the handle is the AS device address; spirv-val verified).
//   Metal:  lowers to an 8-byte MTL resource ID read with argument-buffer
//           semantics (the handle is MTLAccelerationStructure.gpuResourceID);
//           the backend keeps live acceleration structures resident.
//
// NOTE: slangc 2026.13's Metal target has a codegen bug in exactly this
// configuration (DescriptorHandle + RayQuery): it emits MSL that references
// core-module structs (RayDesc) without defining them. The shader pipeline
// patches that textually — see postprocess_metal_source() in
// tools/compile_shaders.py. Earlier spikes also showed indexed
// `RaytracingAccelerationStructure accels[]` arrays emit invalid MSL
// outright, which is why the handle form (not an array) is the bindless
// mechanism here.

struct PC {
    uint64_t outputAddr;
    DescriptorHandle<RaytracingAccelerationStructure> accel;
};
// register(b30) forces Metal buffer(30) — matches kPushConstantSlot in the backend.
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);

[shader("compute")]
[numthreads(1, 1, 1)]
void ray_query_main(uint3 tid : SV_DispatchThreadID)
{
    RaytracingAccelerationStructure accel = g_pc.accel;

    RayDesc ray;
    ray.Origin    = float3(0.25, 0.25, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin      = 0.0;
    ray.TMax      = 100.0;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(accel, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }

    float result = -1.0;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        result = q.CommittedRayT();

    float *out = (float *)g_pc.outputAddr;
    out[0] = result;
}
