// restir_initial.cs.wgsl
// ReSTIR DI — Pass 1: Initial Candidate Generation (RIS)
// Per pixel: draws M candidate light samples, performs Weighted Reservoir
// Sampling (WRS) to keep the best one. Writes to reservoir_buffer.
//
// Runs after shade (bounce 0). Reads hit_buffer (unchanged) for surface info.
// Does NOT rely on ray_buffer for position/direction (shade has modified it).

@group(0) @binding(0)  var<uniform>             pt:                 PTUniforms;
@group(0) @binding(1)  var<uniform>             restir:             ReSTIRUniforms;
@group(0) @binding(2)  var<storage, read>        ray_buffer:         array<PTRay>;
@group(0) @binding(3)  var<storage, read>        hit_buffer:         array<HitRecord>;
@group(0) @binding(4)  var<storage, read>        materials:          array<vec4f>;
@group(0) @binding(5)  var<uniform>              sun_light:          SunLight;
@group(0) @binding(6)  var<storage, read_write>  reservoir_buffer:   array<Reservoir>;
@group(0) @binding(7)  var<storage, read>        emissive_indices:   array<vec4u>;
@group(0) @binding(8)  var<storage, read>        bvh_pos:            array<vec4f>;
@group(0) @binding(9)  var<storage, read>        bvh_normals:        array<vec4f>;
@group(0) @binding(10) var<storage, read>        bvh_uvs:            array<vec4f>;
@group(0) @binding(11) var<storage, read_write>  pixel_data_out:     array<vec4f>;
@group(0) @binding(12) var                       emissive_tex:       texture_2d_array<f32>;
@group(0) @binding(13) var                       tex_sampler:        sampler;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let total_pixels  = pt.width * pt.height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let hit = hit_buffer[pixel_id];

    // If ray missed, write empty reservoir
    if (hit.did_hit == 0u) {
        reservoir_buffer[pixel_id] = reservoirEmpty();
        pixel_data_out[pixel_id] = vec4f(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // ============================================================
    // Reconstruct surface point from BVH vertex data (NOT ray_buffer)
    // ============================================================
    let bw = vec3f(hit.bary.x, hit.bary.y, 1.0 - hit.bary.x - hit.bary.y);

    let p0 = bvh_pos[hit.idx0].xyz;
    let p1 = bvh_pos[hit.idx1].xyz;
    let p2 = bvh_pos[hit.idx2].xyz;
    let hit_pos = bw.x * p0 + bw.y * p1 + bw.z * p2;

    // Smooth normal interpolation
    let n0 = bvh_normals[hit.idx0].xyz;
    let n1 = bvh_normals[hit.idx1].xyz;
    let n2 = bvh_normals[hit.idx2].xyz;
    var surface_N = normalize(bw.x * n0 + bw.y * n1 + bw.z * n2);
    // Flip normal if back-face (use hit.side flag from intersect)
    if (hit.side < 0.0) {
        surface_N = -surface_N;
    }

    var rng = initRNG(pixel_id ^ 0x1337u, restir.frame_index);

    // ============================================================
    // Generate M candidates via WRS
    // ============================================================
    var reservoir = reservoirEmpty();
    let M = restir.candidate_count;

    for (var ci = 0u; ci < M; ci++) {
        var cand_pos:    vec3f;
        var cand_type:   u32;
        var cand_Le:     vec3f;
        var cand_dist:   f32;
        var cand_normal: vec3f;
        var cand_pdf_s:  f32;

        // Decide: sun vs emissive triangle
        // Determine availability of each light type upfront to avoid wasting candidate slots
        let sun_enabled  = sun_light.color.a >= 0.5;
        let has_emissive = restir.emissive_tri_count > 0u;

        // Compute selection probability so we never waste a slot on a disabled source
        // p_sun: probability of picking sun this candidate
        var p_sun: f32;
        if (sun_enabled && has_emissive) {
            p_sun = 0.5;
        } else if (sun_enabled) {
            p_sun = 1.0;
        } else if (has_emissive) {
            p_sun = 0.0;
        } else {
            // No lights at all
            continue;
        }

        let pick_sun = (p_sun >= 1.0) || (p_sun > 0.0 && rand(&rng) < p_sun);

        if (pick_sun) {
            // ----- Sun candidate -----
            let sun_dir = normalize(sun_light.direction.xyz);
            let sun_intensity = sun_light.direction.w;

            // Perturb for soft shadow disk
            let up_vec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(sun_dir.x) > 0.9);
            let sun_tangent   = normalize(cross(sun_dir, up_vec));
            let sun_bitangent = cross(sun_dir, sun_tangent);
            let r2            = rand2(&rng);
            let radius        = sqrt(r2.x) * 0.02;
            let theta_angle   = 2.0 * PI * r2.y;
            let perturbed_dir = normalize(
                sun_dir + sun_tangent * radius * cos(theta_angle) + sun_bitangent * radius * sin(theta_angle)
            );

            cand_pos    = hit_pos + perturbed_dir * 10000.0;
            cand_type   = 1u;
            cand_Le     = sun_light.color.rgb * sun_intensity;
            cand_dist   = 10000.0;
            cand_normal = -perturbed_dir;
            cand_pdf_s  = p_sun; // correct probability accounting for disabled sources
        } else {
            // ----- Emissive triangle candidate -----
            let tri_local_idx = randU32(&rng) % restir.emissive_tri_count;
            let tri_global = emissive_indices[tri_local_idx]; // xyz = vertex indices, w = matId

            let v0 = bvh_pos[tri_global.x].xyz;
            let v1 = bvh_pos[tri_global.y].xyz;
            let v2 = bvh_pos[tri_global.z].xyz;

            // Random point on triangle
            let u1 = rand(&rng);
            let u2 = rand(&rng);
            let su1 = sqrt(u1);
            let bary_light = vec3f(1.0 - su1, su1 * (1.0 - u2), su1 * u2);
            cand_pos = bary_light.x * v0 + bary_light.y * v1 + bary_light.z * v2;

            let edge1     = v1 - v0;
            let edge2     = v2 - v0;
            let cross_e   = cross(edge1, edge2);
            cand_normal   = normalize(cross_e);
            let tri_area  = 0.5 * length(cross_e);

            // Emissive radiance from material
            let mat = unpackPTMaterial(&materials, tri_global.w);
            var final_emission = mat.emissive;
            if (mat.emissive_tex_layer >= 0) {
                let uv0 = bvh_uvs[tri_global.x].xy;
                let uv1 = bvh_uvs[tri_global.y].xy;
                let uv2 = bvh_uvs[tri_global.z].xy;
                let light_uv = bary_light.x * uv0 + bary_light.y * uv1 + bary_light.z * uv2;
                let em_sample = textureSampleLevel(emissive_tex, tex_sampler, light_uv, mat.emissive_tex_layer, 0.0).rgb;
                final_emission *= em_sample;
            }
            cand_Le = final_emission;

            let to_light = cand_pos - hit_pos;
            cand_dist = length(to_light);

            // Source PDF = P(pick emissive) * 1/(count * area)
            cand_pdf_s = (1.0 - p_sun) / (f32(restir.emissive_tri_count) * max(tri_area, 1e-8));
            cand_type = 2u;
        }

        // Evaluate target PDF
        let p_hat = targetPDF(hit_pos, surface_N, cand_pos, cand_Le, cand_normal, cand_type, cand_dist);

        // WRS weight = p_hat / p_source
        let w_i = select(0.0, p_hat / max(cand_pdf_s, 1e-10), p_hat > 0.0);

        reservoirUpdate(&reservoir, cand_pos, cand_type, cand_Le, cand_dist, cand_normal, cand_pdf_s, w_i, &rng);
    }

    // Finalize: compute W
    let final_p_hat = targetPDF(
        hit_pos, surface_N,
        reservoir.sample_pos, reservoir.sample_Le,
        reservoir.sample_normal, reservoir.sample_type,
        reservoir.sample_dist
    );
    reservoirFinalize(&reservoir, final_p_hat);

    reservoir_buffer[pixel_id] = reservoir;
    // Write per-pixel data for spatial resampling geometry checks
    pixel_data_out[pixel_id] = vec4f(surface_N, hit.dist);
}
