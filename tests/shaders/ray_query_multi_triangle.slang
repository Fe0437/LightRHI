// ray_query_multi_triangle.slang
// Inline ray-query against a 3-triangle scene, one ray per dispatch thread
// (4 threads: one per triangle, plus one deliberate miss), exercising a BLAS
// with more than one triangle and per-primitive hit-distance correctness —
// see ray_query_triangle.slang for the single-triangle baseline and the
// bindless DescriptorHandle design/workaround notes.

struct PC {
    uint64_t outputAddr;
    DescriptorHandle<RaytracingAccelerationStructure> accel;
};
// register(b30) forces Metal buffer(30) — matches kPushConstantSlot in the backend.
[[vk::push_constant]] ConstantBuffer<PC> g_pc : register(b30);

[shader("compute")]
[numthreads(4, 1, 1)]
void ray_query_main(uint3 tid : SV_DispatchThreadID)
{
    // Matches the 3 non-overlapping triangles built by the host test
    // (each triangle occupies X in [2*i, 2*i+1], Y in [0,1], Z=0), plus a
    // 4th ray aimed well outside all of them to exercise the miss path.
    float3 origins[4] = {
        float3(0.25, 0.25, -1.0),
        float3(2.25, 0.25, -1.0),
        float3(4.25, 0.25, -1.0),
        float3(10.0, 10.0, -1.0),
    };

    RayDesc ray;
    ray.Origin    = origins[tid.x];
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin      = 0.0;
    ray.TMax      = 100.0;

    RaytracingAccelerationStructure accel = g_pc.accel;
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
    out[tid.x] = result;
}
