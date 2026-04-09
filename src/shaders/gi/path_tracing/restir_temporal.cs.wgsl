// restir_temporal.cs.wgsl
// ReSTIR DI — Pass 2: Temporal Resampling
// Combines current-frame reservoir with previous-frame reservoir for the
// same surface point (found via reprojection).
// Clamps temporal M to prevent unbounded reservoir staleness.
//
// Does NOT rely on ray_buffer for position/direction (shade has modified it).

@group(0) @binding(0)  var<uniform>              pt:                 PTUniforms;
@group(0) @binding(1)  var<uniform>              restir:             ReSTIRUniforms;
@group(0) @binding(2)  var<uniform>              camera:             CameraUniforms;
@group(0) @binding(3)  var<uniform>              prev_camera:        CameraUniforms;
@group(0) @binding(4)  var<storage, read_write>   reservoir_buffer:   array<Reservoir>;
@group(0) @binding(5)  var<storage, read>         prev_reservoir:     array<Reservoir>;
@group(0) @binding(6)  var<storage, read>         hit_buffer:         array<HitRecord>;
@group(0) @binding(7)  var<storage, read>         bvh_pos:            array<vec4f>;
@group(0) @binding(8)  var<storage, read>         bvh_normals:        array<vec4f>;
// Previous frame's per-pixel data stored in storage buffers
@group(0) @binding(9)  var<storage, read>         prev_pixel_data:    array<vec4f>;
// prev_pixel_data[pixel_id] = (normal.xyz, linear_depth)

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let total_pixels  = pt.width * pt.height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    var current_reservoir = reservoir_buffer[pixel_id];

    // If no valid reservoir, nothing to merge
    if (current_reservoir.M == 0u) { return; }

    let hit = hit_buffer[pixel_id];
    if (hit.did_hit == 0u) { return; }

    // ============================================================
    // Reconstruct surface info from BVH (NOT ray_buffer)
    // ============================================================
    let bw = vec3f(hit.bary.x, hit.bary.y, 1.0 - hit.bary.x - hit.bary.y);
    let p0 = bvh_pos[hit.idx0].xyz;
    let p1 = bvh_pos[hit.idx1].xyz;
    let p2 = bvh_pos[hit.idx2].xyz;
    let hit_pos = bw.x * p0 + bw.y * p1 + bw.z * p2;

    let n0 = bvh_normals[hit.idx0].xyz;
    let n1 = bvh_normals[hit.idx1].xyz;
    let n2 = bvh_normals[hit.idx2].xyz;
    var surface_N = normalize(bw.x * n0 + bw.y * n1 + bw.z * n2);
    if (hit.side < 0.0) {
        surface_N = -surface_N;
    }
    let linear_depth = hit.dist;

    // ============================================================
    // Reproject to previous frame
    // ============================================================
    let clip_prev = prev_camera.view_proj_mat * vec4f(hit_pos, 1.0);
    let ndc_prev  = clip_prev.xyz / clip_prev.w;

    // NDC to pixel coords in previous frame
    let uv_prev = vec2f(ndc_prev.x * 0.5 + 0.5, -ndc_prev.y * 0.5 + 0.5);
    let px_prev = vec2i(i32(uv_prev.x * f32(pt.width)), i32(uv_prev.y * f32(pt.height)));

    // Check within bounds
    if (px_prev.x < 0 || px_prev.x >= i32(pt.width) ||
        px_prev.y < 0 || px_prev.y >= i32(pt.height)) {
        // Out of screen — no temporal data, just keep current
        reservoirFinalize(&current_reservoir, targetPDF(
            hit_pos, surface_N,
            current_reservoir.sample_pos, current_reservoir.sample_Le,
            current_reservoir.sample_normal, current_reservoir.sample_type,
            current_reservoir.sample_dist
        ));
        reservoir_buffer[pixel_id] = current_reservoir;
        return;
    }

    let prev_pixel_id = u32(px_prev.y) * pt.width + u32(px_prev.x);

    // ============================================================
    // Geometry consistency check
    // ============================================================
    let prev_data    = prev_pixel_data[prev_pixel_id];
    let prev_normal  = prev_data.xyz;
    let prev_depth   = prev_data.w;

    if (!geometryConsistent(surface_N, prev_normal, linear_depth, prev_depth)) {
        // Geometry mismatch — reject temporal data
        reservoirFinalize(&current_reservoir, targetPDF(
            hit_pos, surface_N,
            current_reservoir.sample_pos, current_reservoir.sample_Le,
            current_reservoir.sample_normal, current_reservoir.sample_type,
            current_reservoir.sample_dist
        ));
        reservoir_buffer[pixel_id] = current_reservoir;
        return;
    }

    // ============================================================
    // Load and clamp previous reservoir
    // ============================================================
    var prev_res = prev_reservoir[prev_pixel_id];

    // Clamp M to prevent staleness: prev.M <= temporal_max_M * current.M
    let max_prev_M = restir.temporal_max_M * current_reservoir.M;
    if (prev_res.M > max_prev_M) {
        let scale = f32(max_prev_M) / f32(prev_res.M);
        prev_res.w_sum *= scale;
        prev_res.M = max_prev_M;
    }

    // ============================================================
    // Combine: merge prev into current via WRS
    // ============================================================
    var rng = initRNG(pixel_id ^ 0x5678u, restir.frame_index);

    let p_hat_prev = targetPDF(
        hit_pos, surface_N,
        prev_res.sample_pos, prev_res.sample_Le,
        prev_res.sample_normal, prev_res.sample_type,
        prev_res.sample_dist
    );
    reservoirCombine(&current_reservoir, prev_res, p_hat_prev, &rng);

    // Finalize with current selected sample's p_hat
    let final_p_hat = targetPDF(
        hit_pos, surface_N,
        current_reservoir.sample_pos, current_reservoir.sample_Le,
        current_reservoir.sample_normal, current_reservoir.sample_type,
        current_reservoir.sample_dist
    );
    reservoirFinalize(&current_reservoir, final_p_hat);

    reservoir_buffer[pixel_id] = current_reservoir;
}
