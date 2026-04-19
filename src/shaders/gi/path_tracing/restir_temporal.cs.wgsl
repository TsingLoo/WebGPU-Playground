// restir_temporal.cs.wgsl
// ReSTIR DI — Pass 2: Temporal Resampling (Unbiased)
//
// Implements unbiased temporal combination (Bitterli et al. 2020, Algorithm 6).
// After merging the current and previous reservoirs via WRS, the final weight is:
//
//   W = w_sum / (Z * p̂_current(y_s))
//
// where Z = sum of M_i for pixels i where p̂_i(y_s) > 0.
// For two-pixel temporal combination, Z has at most two terms:
//   - M_current  (if p̂ at current frame surface > 0)
//   - M_prev     (if p̂ at previous frame surface > 0)
//
// The previous frame's world-space surface position is reconstructed from the
// stored pixel coordinates, depth (prev_pixel_data.w = ray distance), and the
// previous camera's inv_view_proj_mat.
//
// Clamps temporal M to prevent unbounded reservoir staleness.

@group(0) @binding(0)  var<uniform>              pt:                 PTUniforms;
@group(0) @binding(1)  var<uniform>              restir:             ReSTIRUniforms;
@group(0) @binding(2)  var<uniform>              camera:             CameraUniforms;
@group(0) @binding(3)  var<uniform>              prev_camera:        CameraUniforms;
@group(0) @binding(4)  var<storage, read_write>   reservoir_buffer:   array<Reservoir>;
@group(0) @binding(5)  var<storage, read>         prev_reservoir:     array<Reservoir>;
@group(0) @binding(6)  var<storage, read>         hit_buffer:         array<HitRecord>;
@group(0) @binding(7)  var<storage, read>         bvh_pos:            array<vec4f>;
@group(0) @binding(8)  var<storage, read>         bvh_normals:        array<vec4f>;
// prev_pixel_data[pixel_id] = vec4f(normal.xyz, linear_depth)
@group(0) @binding(9)  var<storage, read>         prev_pixel_data:    array<vec4f>;

// Reconstruct world-space surface position for a previous-frame pixel.
// Uses the previous camera's inverse VP matrix to unproject the pixel into world space,
// then walks along the ray to the stored depth.
fn reconstructPrevSurfacePos(px_x: u32, px_y: u32, depth: f32) -> vec3f {
    let uv  = (vec2f(f32(px_x), f32(px_y)) + 0.5) / vec2f(f32(pt.width), f32(pt.height));
    let ndc = vec3f(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 1.0);
    let clip_far = prev_camera.inv_view_proj_mat * vec4f(ndc, 1.0);
    let far_world = clip_far.xyz / clip_far.w;
    let ray_dir = normalize(far_world - prev_camera.camera_pos.xyz);
    return prev_camera.camera_pos.xyz + ray_dir * depth;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let total_pixels = pt.width * pt.height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    var current_reservoir = reservoir_buffer[pixel_id];

    if (current_reservoir.M == 0u) { return; }

    let hit = hit_buffer[pixel_id];
    if (hit.did_hit == 0u) { return; }

    // ================================================================
    // Reconstruct current surface from BVH
    // ================================================================
    let bw = vec3f(hit.bary.x, hit.bary.y, 1.0 - hit.bary.x - hit.bary.y);
    let hit_pos = bw.x * bvh_pos[hit.idx0].xyz
                + bw.y * bvh_pos[hit.idx1].xyz
                + bw.z * bvh_pos[hit.idx2].xyz;

    var surface_N = normalize(
        bw.x * bvh_normals[hit.idx0].xyz +
        bw.y * bvh_normals[hit.idx1].xyz +
        bw.z * bvh_normals[hit.idx2].xyz
    );
    if (hit.side < 0.0) { surface_N = -surface_N; }
    let linear_depth = hit.dist;

    // ================================================================
    // Reproject to previous frame
    // ================================================================
    let clip_prev = prev_camera.view_proj_mat * vec4f(hit_pos, 1.0);
    let ndc_prev  = clip_prev.xyz / clip_prev.w;

    let uv_prev = vec2f(ndc_prev.x * 0.5 + 0.5, -ndc_prev.y * 0.5 + 0.5);
    let px_prev = vec2i(i32(uv_prev.x * f32(pt.width)), i32(uv_prev.y * f32(pt.height)));

    if (px_prev.x < 0 || px_prev.x >= i32(pt.width) ||
        px_prev.y < 0 || px_prev.y >= i32(pt.height)) {
        // Out of screen — no temporal data; finalize with unbiased Z = M_current only
        let p_hat_only = targetPDF(
            hit_pos, surface_N,
            current_reservoir.sample_pos, current_reservoir.sample_Le,
            current_reservoir.sample_normal, current_reservoir.sample_type,
            current_reservoir.sample_dist
        );
        if (p_hat_only < 1e-10) {
            current_reservoir.W = 0.0;
        } else {
            current_reservoir.W = current_reservoir.w_sum / (f32(current_reservoir.M) * p_hat_only);
        }
        reservoir_buffer[pixel_id] = current_reservoir;
        return;
    }

    let prev_pixel_id = u32(px_prev.y) * pt.width + u32(px_prev.x);

    // ================================================================
    // Geometry consistency check
    // ================================================================
    let prev_data   = prev_pixel_data[prev_pixel_id];
    let prev_normal = prev_data.xyz;
    let prev_depth  = prev_data.w;

    if (!geometryConsistent(surface_N, prev_normal, linear_depth, prev_depth)) {
        // Geometry mismatch — reject temporal; finalize with Z = M_current only
        let p_hat_only = targetPDF(
            hit_pos, surface_N,
            current_reservoir.sample_pos, current_reservoir.sample_Le,
            current_reservoir.sample_normal, current_reservoir.sample_type,
            current_reservoir.sample_dist
        );
        if (p_hat_only < 1e-10) {
            current_reservoir.W = 0.0;
        } else {
            current_reservoir.W = current_reservoir.w_sum / (f32(current_reservoir.M) * p_hat_only);
        }
        reservoir_buffer[pixel_id] = current_reservoir;
        return;
    }

    // ================================================================
    // Load and clamp previous reservoir
    // ================================================================
    var prev_res = prev_reservoir[prev_pixel_id];

    // Clamp M to prevent staleness: prev.M <= temporal_max_M * current.M
    let max_prev_M = restir.temporal_max_M * current_reservoir.M;
    if (prev_res.M > max_prev_M) {
        let scale = f32(max_prev_M) / f32(prev_res.M);
        prev_res.w_sum *= scale;
        prev_res.M = max_prev_M;
    }

    // ================================================================
    // Combine current + previous via WRS
    // ================================================================
    // Save current M before combination (needed for unbiased Z)
    let M_current = current_reservoir.M;

    var rng = initRNG(pixel_id ^ 0x5678u, restir.frame_index);

    let p_hat_prev = targetPDF(
        hit_pos, surface_N,
        prev_res.sample_pos, prev_res.sample_Le,
        prev_res.sample_normal, prev_res.sample_type,
        prev_res.sample_dist
    );
    reservoirCombine(&current_reservoir, prev_res, p_hat_prev, &rng);

    // ================================================================
    // Unbiased Z computation (Bitterli 2020, Alg. 6)
    // ================================================================
    let final_p_hat = targetPDF(
        hit_pos, surface_N,
        current_reservoir.sample_pos, current_reservoir.sample_Le,
        current_reservoir.sample_normal, current_reservoir.sample_type,
        current_reservoir.sample_dist
    );

    var Z = 0u;

    // Current pixel: p̂_current(y_s)
    if (final_p_hat > 0.0) {
        Z += M_current;
    }

    // Previous pixel: reconstruct world-space position, then evaluate p̂_prev(y_s)
    let prev_surface_pos = reconstructPrevSurfacePos(
        u32(px_prev.x), u32(px_prev.y), prev_depth
    );

    let p_hat_at_prev = targetPDF(
        prev_surface_pos, prev_normal,
        current_reservoir.sample_pos, current_reservoir.sample_Le,
        current_reservoir.sample_normal, current_reservoir.sample_type,
        current_reservoir.sample_dist
    );
    if (p_hat_at_prev > 0.0) {
        Z += prev_res.M;
    }

    // ================================================================
    // Unbiased W = w_sum / (Z * p̂_current)
    // ================================================================
    if (Z == 0u || final_p_hat < 1e-10) {
        current_reservoir.W = 0.0;
        current_reservoir.M = 0u;
    } else {
        current_reservoir.W = current_reservoir.w_sum / (f32(Z) * final_p_hat);
        current_reservoir.M = Z;
    }

    reservoir_buffer[pixel_id] = current_reservoir;
}
