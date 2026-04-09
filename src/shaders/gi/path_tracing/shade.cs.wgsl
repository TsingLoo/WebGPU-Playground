// shade.cs.wgsl
// Wavefront Path Tracing — Pass 3: Material Shading + NEE
// Full PBR with normal mapping + metallic-roughness texture support.

@group(0) @binding(0)  var<uniform>             camera:             CameraUniforms;
@group(0) @binding(1)  var<uniform>             pt:                 PTUniforms;
@group(0) @binding(2)  var<storage, read_write>  ray_buffer:         array<PTRay>;
@group(0) @binding(3)  var<storage, read_write>  hit_buffer:         array<HitRecord>;
@group(0) @binding(4)  var<storage, read_write>  shadow_buffer:      array<ShadowRay>;
@group(0) @binding(5)  var<storage, read_write>  accum_buffer:       array<vec4f>;
@group(0) @binding(6)  var<storage, read>        materials:          array<vec4f>;
@group(0) @binding(7)  var<uniform>              sun_light:          SunLight;
@group(0) @binding(8)  var                       base_color_tex:     texture_2d_array<f32>;
@group(0) @binding(9)  var                       base_color_sampler: sampler;
@group(0) @binding(10) var                       normal_map_tex:     texture_2d_array<f32>;
@group(0) @binding(11) var                       normal_map_sampler: sampler;
@group(0) @binding(12) var                       mr_tex:             texture_2d_array<f32>;
@group(0) @binding(13) var                       mr_sampler:         sampler;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    // Default: deactivate shadow ray
    var shadow: ShadowRay;
    shadow.shadow_active = 0u;
    shadow_buffer[pixel_id] = shadow;

    var ray = ray_buffer[pixel_id];
    if (ray.ray_active == 0u) { return; }

    let hit = hit_buffer[pixel_id];

    // Miss: leave ray state for the miss shader to handle
    if (hit.did_hit == 0u) {
        ray_buffer[pixel_id] = ray;
        return;
    }

    let mat = unpackPTMaterial(&materials, hit.mat_id);

    var albedo = mat.albedo;
    if (mat.tex_layer >= 0) {
        let tex_col = textureSampleLevel(base_color_tex, base_color_sampler, hit.uv, mat.tex_layer, 0.0);
        albedo *= tex_col.xyz;
    }

    // --- Metallic-Roughness map sampling ---
    var roughness = mat.roughness;
    var metallic  = mat.metallic;
    if (mat.mr_tex_layer >= 0) {
        let mr_sample = textureSampleLevel(mr_tex, mr_sampler, hit.uv, mat.mr_tex_layer, 0.0);
        // glTF spec: G = roughness, B = metallic
        roughness *= mr_sample.g;
        metallic  *= mr_sample.b;
    }
    roughness = max(roughness, 0.04);

    // --- Normal mapping ---
    var N = hit.normal; // smooth interpolated vertex normal
    if (mat.normal_tex_layer >= 0) {
        // Build TBN matrix from interpolated normal and tangent
        let T_raw = hit.tangent.xyz - N * dot(hit.tangent.xyz, N); // Gram-Schmidt re-orthogonalize
        let T_len = length(T_raw);
        var T = vec3f(0.0);
        if (T_len > 0.001) {
            T = T_raw / T_len;
        } else {
            // Fallback: generate tangent from normal
            let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
            T = normalize(cross(N, refVec));
        }
        let handedness = select(hit.tangent.w, 1.0, abs(hit.tangent.w) < 0.5);
        let B = normalize(cross(N, T)) * handedness;
        let tbn = mat3x3f(T, B, N);

        let normal_sample = textureSampleLevel(normal_map_tex, normal_map_sampler, hit.uv, mat.normal_tex_layer, 0.0).rgb;
        let tangent_normal = normal_sample * 2.0 - 1.0;
        N = normalize(tbn * tangent_normal);
    }

    // Use geometric normal for ray offset to avoid self-intersection
    let geom_N = hit.geom_normal;

    var rng = initRNG(pixel_id ^ (ray.bounce * 1009u), pt.frame_index);

    let V     = -ray.direction;
    let NdotV = max(dot(N, V), 0.0001);

    // Emissive
    if (any(mat.emissive > vec3f(0.001))) {
        let prev = accum_buffer[pixel_id];
        let em   = clamp(ray.throughput * mat.emissive, vec3f(0.0), vec3f(pt.clamp_radiance));
        accum_buffer[pixel_id] = vec4f(prev.xyz + em, prev.w);
    }

    // =================================================================
    // Glass / Transmissive
    // =================================================================
    if (mat.transmission > 0.5) {
        let eta_in  = select(mat.ior, 1.0, hit.side > 0.0);
        let eta_out = select(1.0, mat.ior, hit.side > 0.0);
        let cos_i   = max(dot(N, V), 0.0001);
        let F       = fresnelDielectric(cos_i, eta_in, eta_out);

        var new_dir: vec3f;
        var new_ior: f32;
        if (rand(&rng) < F) {
            new_dir = reflect(-V, N);
            new_ior = ray.ior;
        } else {
            let eta       = eta_in / eta_out;
            let refracted = refract(-V, N * hit.side, eta);
            if (all(refracted == vec3f(0.0))) {
                new_dir = reflect(-V, N);
                new_ior = ray.ior;
            } else {
                new_dir = normalize(refracted);
                new_ior = eta_out;
            }
        }
        ray.throughput      *= albedo;
        ray.origin           = hit.pos + new_dir * 0.001;
        ray.direction        = new_dir;
        ray.ior              = new_ior;
        ray.bounce          += 1u;
        ray.ray_active       = select(0u, 1u, ray.bounce < pt.max_bounces);
        ray.specular_bounce  = 1u;
        ray_buffer[pixel_id] = ray;
        return;
    }

    // =================================================================
    // Opaque PBR
    // =================================================================
    let a  = roughness * roughness;
    let a2 = a * a;
    let F0 = mix(vec3f(0.04), albedo, metallic);

    // Russian Roulette (only after bounce 2)
    let rr_max = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));
    let rr_survive = clamp(rr_max, 0.05, 1.0);
    if (ray.bounce >= 2u && rand(&rng) > rr_survive) {
        ray.ray_active = 0u;
        ray_buffer[pixel_id] = ray;
        return;
    }
    let rr_weight = select(1.0 / rr_survive, 1.0, ray.bounce < 2u);

    // Stochastic lobe selection: specular vs diffuse
    let F_avg = fresnelSchlickV(NdotV, F0);
    let spec_weight = max(F_avg.x, max(F_avg.y, F_avg.z));
    let spec_prob   = clamp(spec_weight, 0.1, 0.9);
    let use_specular = rand(&rng) < spec_prob;

    var new_dir: vec3f;
    var new_throughput: vec3f;
    var is_specular: bool = false;

    if (use_specular) {
        // GGX importance sampling
        let H      = sampleGGX(N, roughness, &rng);
        let L      = normalize(reflect(-V, H));
        let NdotL  = dot(N, L);
        if (NdotL <= 0.0) {
            ray.ray_active = 0u;
            ray_buffer[pixel_id] = ray;
            return;
        }
        let NdotH  = max(dot(N, H), 0.0001);
        let VdotH  = max(dot(V, H), 0.0001);
        let NdotL_ = max(NdotL, 0.0001);

        let Fspec  = fresnelSchlickV(VdotH, F0);
        let G      = geometrySchlickGGX_pt(NdotV, roughness)
                   * geometrySchlickGGX_pt(NdotL_, roughness);

        // BRDF/PDF weight for GGX importance sampling of H:
        // weight = F * G * VdotH / (NdotH * NdotV)
        // Divide by spec_prob for unbiased MIS
        let spec_brdf_weight = Fspec * G * VdotH / max(NdotH * NdotV, 0.0001);
        new_dir       = L;
        new_throughput = ray.throughput * spec_brdf_weight * (rr_weight / spec_prob);
        is_specular   = (roughness < 0.05);
    } else {
        // Cosine-weighted hemisphere sampling for diffuse
        let L       = sampleCosineHemisphere(N, &rng);
        let kD      = (vec3f(1.0) - fresnelSchlickV(NdotV, F0)) * (1.0 - metallic);
        new_dir       = L;
        // Divide by (1-spec_prob) for unbiased MIS
        new_throughput = ray.throughput * albedo * kD * (rr_weight / (1.0 - spec_prob));
    }

    // Clamp throughput to prevent fireflies
    let tp_max = max(new_throughput.x, max(new_throughput.y, new_throughput.z));
    if (tp_max > 20.0) {
        new_throughput *= 20.0 / tp_max;
    }

    // NEE — direct sun light
    if (sun_light.color.a > 0.5 && !is_specular) {
        let base_sun_dir = normalize(sun_light.direction.xyz);

        // Soft shadow: jitter sun direction within a small cone
        let up_vec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(base_sun_dir.x) > 0.9);
        let sun_tangent = normalize(cross(base_sun_dir, up_vec));
        let sun_bitangent = cross(base_sun_dir, sun_tangent);
        let r2 = rand2(&rng);
        let r = sqrt(r2.x) * 0.02;
        let theta = 2.0 * PI * r2.y;
        let sun_dir = normalize(base_sun_dir + sun_tangent * r * cos(theta) + sun_bitangent * r * sin(theta));

        let NdotL_s  = max(dot(N, sun_dir), 0.0);
        if (NdotL_s > 0.0) {
            let H_s     = normalize(V + sun_dir);
            let NdotH_s = max(dot(N, H_s), 0.0);
            let VdotH_s = max(dot(V, H_s), 0.0);
            let F_s     = fresnelSchlickV(VdotH_s, F0);
            let G_s     = geometrySchlickGGX_pt(NdotV, roughness)
                        * geometrySchlickGGX_pt(NdotL_s, roughness);
            let D_s     = distributionGGX_pt(NdotH_s, a2);
            let spec_s  = F_s * D_s * G_s / max(4.0 * NdotV * NdotL_s, 0.0001);
            let diff_s  = (vec3f(1.0) - F_s) * (1.0 - metallic) * albedo / PI;
            let sun_rad = sun_light.color.rgb * sun_light.direction.w;
            let Lo_sun  = (diff_s + spec_s) * sun_rad * NdotL_s;

            shadow.origin       = hit.pos + geom_N * 0.002;
            shadow.max_dist     = 1000.0;
            shadow.direction    = sun_dir;
            shadow.pixel_id     = pixel_id;
            shadow.Li           = clamp(ray.throughput * Lo_sun, vec3f(0.0), vec3f(pt.clamp_radiance));
            shadow.shadow_active = 1u;
        }
    }
    shadow_buffer[pixel_id] = shadow;

    // Spawn next bounce
    ray.throughput      = new_throughput;
    ray.origin          = hit.pos + geom_N * 0.001;
    ray.direction       = new_dir;
    ray.bounce         += 1u;
    ray.ray_active      = select(0u, 1u, ray.bounce < pt.max_bounces);
    ray.specular_bounce = select(0u, 1u, is_specular);

    if (max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z)) < 0.001) {
        ray.ray_active = 0u;
    }
    ray_buffer[pixel_id] = ray;
}
