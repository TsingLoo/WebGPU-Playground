// restir_spatial.cs.wgsl
// ReSTIR DI — Pass 3: Spatial Resampling
// Combines reservoirs from K random neighbor pixels within screen-space radius R.
// Uses geometry rejection (normal + depth consistency) and MIS-weighted combination.
//
// Uses two separate buffers: reads from reservoir_in, writes to reservoir_out
// for safe parallel execution (no read-after-write hazard).

@group(0) @binding(0)  var<uniform>              pt:                 PTUniforms;
@group(0) @binding(1)  var<uniform>              restir:             ReSTIRUniforms;
@group(0) @binding(2)  var<storage, read>         reservoir_in:       array<Reservoir>;
@group(0) @binding(3)  var<storage, read_write>   reservoir_out:      array<Reservoir>;
@group(0) @binding(4)  var<storage, read>         ray_buffer:         array<PTRay>;
@group(0) @binding(5)  var<storage, read>         hit_buffer:         array<HitRecord>;
@group(0) @binding(6)  var<storage, read>         bvh_normals:        array<vec4f>;
// Current frame pixel data for geometry checks
@group(0) @binding(7)  var<storage, read>         pixel_data:         array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let ray = ray_buffer[pixel_id];
    let hit = hit_buffer[pixel_id];

    // Default: copy input to output
    var result = reservoir_in[pixel_id];

    if (ray.ray_active == 0u || hit.did_hit == 0u || result.M == 0u) {
        reservoir_out[pixel_id] = result;
        return;
    }

    // ============================================================
    // Reconstruct surface info for center pixel
    // ============================================================
    let hit_pos = ray.origin + ray.direction * hit.dist;
    let bw = vec3f(hit.bary.x, hit.bary.y, 1.0 - hit.bary.x - hit.bary.y);
    let n0 = bvh_normals[hit.idx0].xyz;
    let n1 = bvh_normals[hit.idx1].xyz;
    let n2 = bvh_normals[hit.idx2].xyz;
    var surface_N = normalize(bw.x * n0 + bw.y * n1 + bw.z * n2);
    if (dot(surface_N, ray.direction) > 0.0) {
        surface_N = -surface_N;
    }
    let linear_depth = hit.dist;

    var rng = initRNG(pixel_id ^ 0xABCDu, restir.frame_index);

    // Pixel X,Y for neighbor sampling
    let px_x = pixel_id % render_width;
    let px_y = pixel_id / render_width;

    let K = restir.spatial_count;
    let R = f32(restir.spatial_radius);

    // Start: combined reservoir begins as the center pixel's reservoir
    var combined = result;

    for (var ki = 0u; ki < K; ki++) {
        // Random neighbor within disk of radius R
        let r_radius = sqrt(rand(&rng)) * R;
        let r_angle  = rand(&rng) * 2.0 * PI;
        let offset_x = i32(round(r_radius * cos(r_angle)));
        let offset_y = i32(round(r_radius * sin(r_angle)));

        let nb_x = i32(px_x) + offset_x;
        let nb_y = i32(px_y) + offset_y;

        // Skip out-of-bounds
        if (nb_x < 0 || nb_x >= i32(render_width) || nb_y < 0 || nb_y >= i32(render_height)) {
            continue;
        }
        // Skip self
        if (offset_x == 0 && offset_y == 0) {
            continue;
        }

        let nb_pixel_id = u32(nb_y) * render_width + u32(nb_x);

        // Geometry consistency check
        let nb_data   = pixel_data[nb_pixel_id];
        let nb_normal = nb_data.xyz;
        let nb_depth  = nb_data.w;

        if (!geometryConsistent(surface_N, nb_normal, linear_depth, nb_depth)) {
            continue;
        }

        // Load neighbor reservoir
        let nb_reservoir = reservoir_in[nb_pixel_id];
        if (nb_reservoir.M == 0u) { continue; }

        // Evaluate target PDF of neighbor's sample at CENTER pixel's shading point
        let p_hat_nb = targetPDF(
            hit_pos, surface_N,
            nb_reservoir.sample_pos, nb_reservoir.sample_Le,
            nb_reservoir.sample_normal, nb_reservoir.sample_type,
            nb_reservoir.sample_dist
        );

        // MIS-weighted combination
        reservoirCombine(&combined, nb_reservoir, p_hat_nb, &rng);
    }

    // Finalize combined reservoir
    let final_p_hat = targetPDF(
        hit_pos, surface_N,
        combined.sample_pos, combined.sample_Le,
        combined.sample_normal, combined.sample_type,
        combined.sample_dist
    );
    reservoirFinalize(&combined, final_p_hat);

    reservoir_out[pixel_id] = combined;
}
