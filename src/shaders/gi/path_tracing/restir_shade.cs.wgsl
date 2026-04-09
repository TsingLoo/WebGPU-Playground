// restir_shade.cs.wgsl
// ReSTIR DI — Pass 4: Shade From Reservoir → Shadow Ray
// Evaluates the final reservoir's selected light sample through full PBR BRDF,
// multiplied by the reservoir weight W. Generates a shadow ray for visibility testing.
//
// This replaces the NEE section of shade.cs.wgsl when ReSTIR is enabled.
// Does NOT rely on ray_buffer for position/direction (shade has modified it).
// Reconstructs everything from BVH + camera.

@group(0) @binding(0)  var<uniform>              pt:                 PTUniforms;
@group(0) @binding(1)  var<uniform>              restir:             ReSTIRUniforms;
@group(0) @binding(2)  var<uniform>              camera:             CameraUniforms;
@group(0) @binding(3)  var<storage, read>         hit_buffer:         array<HitRecord>;
@group(0) @binding(4)  var<storage, read>         reservoir_buffer:   array<Reservoir>;
@group(0) @binding(5)  var<storage, read_write>   shadow_buffer:      array<ShadowRay>;
@group(0) @binding(6)  var<storage, read_write>   accum_buffer:       array<vec4f>;
@group(0) @binding(7)  var<storage, read>         materials:          array<vec4f>;
@group(0) @binding(8)  var                        base_color_tex:     texture_2d_array<f32>;
@group(0) @binding(9)  var                        tex_sampler:        sampler;
@group(0) @binding(10) var                        mr_tex:             texture_2d_array<f32>;
@group(0) @binding(11) var<storage, read>         bvh_pos:            array<vec4f>;
@group(0) @binding(12) var<storage, read>         bvh_uvs:            array<vec4f>;
@group(0) @binding(13) var<storage, read>         bvh_normals:        array<vec4f>;
@group(0) @binding(14) var<storage, read>         bvh_tangents:       array<vec4f>;
@group(0) @binding(15) var                        normal_map_tex:     texture_2d_array<f32>;
// Pixel data output — for temporal resampling next frame
@group(0) @binding(16) var<storage, read_write>   pixel_data_out:     array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let total_pixels  = pt.width * pt.height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    // Default: deactivate shadow ray
    var shadow: ShadowRay;
    shadow.shadow_active = 0u;

    let hit = hit_buffer[pixel_id];

    if (hit.did_hit == 0u) {
        shadow_buffer[pixel_id] = shadow;
        pixel_data_out[pixel_id] = vec4f(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // ============================================================
    // Reconstruct hit attributes from BVH (NOT ray_buffer)
    // ============================================================
    let bw = vec3f(hit.bary.x, hit.bary.y, 1.0 - hit.bary.x - hit.bary.y);

    let p0 = bvh_pos[hit.idx0].xyz;
    let p1 = bvh_pos[hit.idx1].xyz;
    let p2 = bvh_pos[hit.idx2].xyz;
    let hit_pos = bw.x * p0 + bw.y * p1 + bw.z * p2;

    // UV interpolation
    let uv0 = bvh_uvs[hit.idx0].xy;
    let uv1 = bvh_uvs[hit.idx1].xy;
    let uv2 = bvh_uvs[hit.idx2].xy;
    let hit_uv = bw.x * uv0 + bw.y * uv1 + bw.z * uv2;

    // Smooth normal
    let sn0 = bvh_normals[hit.idx0].xyz;
    let sn1 = bvh_normals[hit.idx1].xyz;
    let sn2 = bvh_normals[hit.idx2].xyz;
    var smooth_N = normalize(bw.x * sn0 + bw.y * sn1 + bw.z * sn2);
    if (hit.side < 0.0) {
        smooth_N = -smooth_N;
    }

    // Tangent interpolation
    let t0 = bvh_tangents[hit.idx0];
    let t1 = bvh_tangents[hit.idx1];
    let t2 = bvh_tangents[hit.idx2];
    let interp_tangent_dir = normalize(bw.x * t0.xyz + bw.y * t1.xyz + bw.z * t2.xyz);
    let handedness = t0.w;

    // Material
    let mat = unpackPTMaterial(&materials, hit.mat_id);

    var albedo = mat.albedo;
    if (mat.tex_layer >= 0) {
        let tex_col = textureSampleLevel(base_color_tex, tex_sampler, hit_uv, mat.tex_layer, 0.0);
        albedo *= tex_col.xyz;
    }

    var roughness = mat.roughness;
    var metallic  = mat.metallic;
    if (mat.mr_tex_layer >= 0) {
        let mr_sample = textureSampleLevel(mr_tex, tex_sampler, hit_uv, mat.mr_tex_layer, 0.0);
        roughness *= mr_sample.g;
        metallic  *= mr_sample.b;
    }
    roughness = max(roughness, 0.04);

    // Normal mapping
    var N = smooth_N;
    if (mat.normal_tex_layer >= 0) {
        let T_raw = interp_tangent_dir - N * dot(interp_tangent_dir, N);
        let T_len = length(T_raw);
        var T = vec3f(0.0);
        if (T_len > 0.001) {
            T = T_raw / T_len;
        } else {
            let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
            T = normalize(cross(N, refVec));
        }
        let h_sign = select(handedness, 1.0, abs(handedness) < 0.5);
        let B = normalize(cross(N, T)) * h_sign;
        let tbn = mat3x3f(T, B, N);
        let normal_sample = textureSampleLevel(normal_map_tex, tex_sampler, hit_uv, mat.normal_tex_layer, 0.0).rgb;
        let tangent_normal = normal_sample * 2.0 - 1.0;
        N = normalize(tbn * tangent_normal);
    }

    // ============================================================
    // Write pixel data for next frame's temporal resampling
    // ============================================================
    pixel_data_out[pixel_id] = vec4f(N, hit.dist);

    // ============================================================
    // Evaluate reservoir
    // ============================================================
    let reservoir = reservoir_buffer[pixel_id];

    if (reservoir.M == 0u || reservoir.W < 1e-10) {
        shadow_buffer[pixel_id] = shadow;
        return;
    }

    // View direction from camera position (NOT from ray_buffer!)
    let V = normalize(camera.camera_pos.xyz - hit_pos);
    let NdotV = max(dot(N, V), 0.0001);

    let a  = roughness * roughness;
    let a2 = a * a;
    let F0 = mix(vec3f(0.04), albedo, metallic);

    // Direction to light
    var L: vec3f;
    var light_dist: f32;
    if (reservoir.sample_type == 1u) {
        // Sun: direction is toward far pos
        L = normalize(reservoir.sample_pos - hit_pos);
        light_dist = 1000.0; // effectively infinite
    } else {
        // Area light
        L = normalize(reservoir.sample_pos - hit_pos);
        light_dist = length(reservoir.sample_pos - hit_pos);
    }

    let NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        shadow_buffer[pixel_id] = shadow;
        return;
    }

    // Full PBR BRDF evaluation
    let H_vec   = normalize(V + L);
    let NdotH   = max(dot(N, H_vec), 0.0);
    let VdotH   = max(dot(V, H_vec), 0.0);
    let F_brdf  = fresnelSchlickV(VdotH, F0);
    let G_brdf  = geometrySchlickGGX_pt(NdotV, roughness) * geometrySchlickGGX_pt(NdotL, roughness);
    let D_brdf  = distributionGGX_pt(NdotH, a2);
    let spec_brdf = F_brdf * D_brdf * G_brdf / max(4.0 * NdotV * NdotL, 0.0001);
    let diff_brdf = (vec3f(1.0) - F_brdf) * (1.0 - metallic) * albedo / PI;

    let brdf_eval = diff_brdf + spec_brdf;

    // ReSTIR contribution = BRDF * Le * W * NdotL
    // At bounce 0, throughput is (1,1,1) so we don't need ray.throughput
    let Lo_restir = brdf_eval * reservoir.sample_Le * reservoir.W * NdotL;

    // Generate shadow ray
    shadow.origin       = hit_pos + smooth_N * 0.002;
    shadow.max_dist     = light_dist - 0.01;
    shadow.direction    = L;
    shadow.pixel_id     = pixel_id;
    shadow.Li           = clamp(Lo_restir, vec3f(0.0), vec3f(pt.clamp_radiance));
    shadow.shadow_active = 1u;

    shadow_buffer[pixel_id] = shadow;
}
