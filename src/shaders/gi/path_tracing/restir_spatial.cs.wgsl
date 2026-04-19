// restir_spatial.cs.wgsl
// ReSTIR DI — Pass 3: Spatial Resampling (Unbiased)
//
// Implements the unbiased combination from Bitterli et al. 2020, Algorithm 6.
// Key difference from the biased version: after combining K neighbors into the
// center reservoir, we compute Z = sum of M_i for all pixels i where the
// selected sample y has non-zero target PDF p̂_i(y).  The final weight becomes:
//
//   W = w_sum / (Z * p̂_center(y))   [unbiased]
//
// instead of the biased:
//
//   W = w_sum / (M_total * p̂_center(y))
//
// This eliminates darkening artifacts at depth discontinuities caused by
// counting contributions from pixels that cannot "see" the selected sample.
//
// Implementation uses two separate RNG streams so that neighbor positions
// are reproducible in the Z-counting pass:
//   rng_pos — drives neighbor disk sampling (replayed identically in pass 2)
//   rng_wrs — drives WRS acceptance (only needed in pass 1)
//
// K is capped at 32 (fits in the u32 valid_mask bitmask).

@group(0) @binding(0)  var<uniform>              pt:                 PTUniforms;
@group(0) @binding(1)  var<uniform>              restir:             ReSTIRUniforms;
@group(0) @binding(2)  var<storage, read>         reservoir_in:       array<Reservoir>;
@group(0) @binding(3)  var<storage, read_write>   reservoir_out:      array<Reservoir>;
@group(0) @binding(4)  var<storage, read>         hit_buffer:         array<HitRecord>;
@group(0) @binding(5)  var<storage, read>         bvh_pos:            array<vec4f>;
@group(0) @binding(6)  var<storage, read>         bvh_normals:        array<vec4f>;
// Current frame pixel data for geometry checks on neighbors
@group(0) @binding(7)  var<storage, read>         pixel_data:         array<vec4f>;

// Reconstruct world-space hit position and shading normal from a HitRecord.
fn reconstructSurface(h: HitRecord) -> vec4f {
    // returns vec4f(pos.xyz, packed_sign) — normal is computed separately
    let bw = vec3f(h.bary.x, h.bary.y, 1.0 - h.bary.x - h.bary.y);
    let pos = bw.x * bvh_pos[h.idx0].xyz
            + bw.y * bvh_pos[h.idx1].xyz
            + bw.z * bvh_pos[h.idx2].xyz;
    return vec4f(pos, h.side);
}

fn reconstructNormal(h: HitRecord) -> vec3f {
    let bw = vec3f(h.bary.x, h.bary.y, 1.0 - h.bary.x - h.bary.y);
    var N = normalize(
        bw.x * bvh_normals[h.idx0].xyz +
        bw.y * bvh_normals[h.idx1].xyz +
        bw.z * bvh_normals[h.idx2].xyz
    );
    if (h.side < 0.0) { N = -N; }
    return N;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let total_pixels = pt.width * pt.height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let hit = hit_buffer[pixel_id];

    // Default: copy input to output
    var result = reservoir_in[pixel_id];

    if (hit.did_hit == 0u || result.M == 0u) {
        reservoir_out[pixel_id] = result;
        return;
    }

    // ================================================================
    // Reconstruct center pixel surface from BVH
    // ================================================================
    let surf       = reconstructSurface(hit);
    let hit_pos    = surf.xyz;
    let surface_N  = reconstructNormal(hit);
    let linear_depth = hit.dist;

    let px_x = pixel_id % pt.width;
    let px_y = pixel_id / pt.width;

    let K = min(restir.spatial_count, 32u);  // capped at 32 to fit in valid_mask u32
    let R = f32(restir.spatial_radius);

    // Two independent RNG streams — separate position sampling from WRS acceptance
    // so that pass 2 can replay the exact same neighbor positions.
    var rng_pos = initRNG(pixel_id ^ 0xABCDu, restir.frame_index);
    var rng_wrs = initRNG(pixel_id ^ 0xF00Du, restir.frame_index);

    // ================================================================
    // Pass 1: Combination
    // Merges K neighbors into the center reservoir via WRS.
    // Tracks which neighbors were valid in valid_mask (bit ki = neighbor ki passed).
    // ================================================================
    var combined   = result;   // start with center pixel's own initial reservoir
    var valid_mask = 0u;       // bitmask of accepted neighbors

    for (var ki = 0u; ki < K; ki++) {
        // Sample random neighbor in disk of radius R — uses rng_pos only
        let r_radius = sqrt(rand(&rng_pos)) * R;
        let r_angle  = rand(&rng_pos) * 2.0 * PI;
        let offset_x = i32(round(r_radius * cos(r_angle)));
        let offset_y = i32(round(r_radius * sin(r_angle)));

        let nb_x = i32(px_x) + offset_x;
        let nb_y = i32(px_y) + offset_y;

        if (nb_x < 0 || nb_x >= i32(pt.width) || nb_y < 0 || nb_y >= i32(pt.height)) { continue; }
        if (offset_x == 0 && offset_y == 0) { continue; }

        let nb_pixel_id = u32(nb_y) * pt.width + u32(nb_x);

        // Geometry consistency check using cached pixel_data
        let nb_data = pixel_data[nb_pixel_id];
        if (!geometryConsistent(surface_N, nb_data.xyz, linear_depth, nb_data.w)) { continue; }

        let nb_reservoir = reservoir_in[nb_pixel_id];
        if (nb_reservoir.M == 0u) { continue; }

        // Evaluate neighbor's sample at center pixel's shading point
        let p_hat_nb = targetPDF(
            hit_pos, surface_N,
            nb_reservoir.sample_pos, nb_reservoir.sample_Le,
            nb_reservoir.sample_normal, nb_reservoir.sample_type,
            nb_reservoir.sample_dist
        );

        // Combine into running reservoir — rng_wrs drives the accept/reject
        reservoirCombine(&combined, nb_reservoir, p_hat_nb, &rng_wrs);

        // Mark this neighbor as valid for the Z pass
        valid_mask |= (1u << ki);
    }

    // ================================================================
    // Pass 2: Compute Z (unbiased MIS denominator)
    //
    // Z = sum of M_i over all pixels i (center + valid neighbors) where
    //     p̂_i(y_s) > 0  (i.e., the selected sample is "reachable" from pixel i)
    //
    // We replay rng_pos with the same seed to regenerate the exact same
    // neighbor positions.  rng_wrs is NOT replayed — it was only needed
    // for the accept/reject decisions in pass 1.
    // ================================================================

    // Evaluate target PDF at center pixel for the selected sample y_s
    let p_hat_center = targetPDF(
        hit_pos, surface_N,
        combined.sample_pos, combined.sample_Le,
        combined.sample_normal, combined.sample_type,
        combined.sample_dist
    );

    var Z = 0u;
    // Center pixel contributes its M if sample is reachable from here
    if (p_hat_center > 0.0) {
        Z += result.M;
    }

    // Reset position RNG to the same seed used in pass 1
    var rng_pos2 = initRNG(pixel_id ^ 0xABCDu, restir.frame_index);

    for (var ki = 0u; ki < K; ki++) {
        // Consume same two rand() values to regenerate the same neighbor position
        let r_radius = sqrt(rand(&rng_pos2)) * R;
        let r_angle  = rand(&rng_pos2) * 2.0 * PI;

        // Only process neighbors that were valid in pass 1
        if ((valid_mask & (1u << ki)) == 0u) { continue; }

        let offset_x = i32(round(r_radius * cos(r_angle)));
        let offset_y = i32(round(r_radius * sin(r_angle)));
        let nb_pixel_id = u32(i32(px_y) + offset_y) * pt.width + u32(i32(px_x) + offset_x);

        // Reconstruct neighbor's exact world-space surface from BVH (same method as center)
        let nb_hit = hit_buffer[nb_pixel_id];
        let nb_pos = reconstructSurface(nb_hit).xyz;
        let nb_N   = reconstructNormal(nb_hit);

        // Check if the selected sample y_s has non-zero contribution at this neighbor's surface
        let p_hat_at_nb = targetPDF(
            nb_pos, nb_N,
            combined.sample_pos, combined.sample_Le,
            combined.sample_normal, combined.sample_type,
            combined.sample_dist
        );

        if (p_hat_at_nb > 0.0) {
            Z += reservoir_in[nb_pixel_id].M;
        }
    }

    // ================================================================
    // Unbiased finalization: W = w_sum / (Z * p̂_center)
    // ================================================================
    var final_res = combined;
    if (Z == 0u || p_hat_center < 1e-10) {
        final_res.W = 0.0;
        final_res.M = 0u;
    } else {
        final_res.W = final_res.w_sum / (f32(Z) * p_hat_center);
        final_res.M = Z;
    }

    reservoir_out[pixel_id] = final_res;
}
