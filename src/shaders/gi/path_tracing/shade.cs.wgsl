// shade.cs.wgsl
// Wavefront Path Tracing — Pass 3: Material Shading + NEE

@group(0) @binding(0)  var<uniform>             camera:        CameraUniforms;
@group(0) @binding(1)  var<uniform>             pt:            PTUniforms;
@group(0) @binding(2)  var<storage, read_write>  ray_buffer:    array<PTRay>;
@group(0) @binding(3)  var<storage, read_write>  hit_buffer:    array<HitRecord>;
@group(0) @binding(4)  var<storage, read_write>  shadow_buffer: array<ShadowRay>;
@group(0) @binding(5)  var<storage, read_write>  accum_buffer:  array<vec4f>;
@group(0) @binding(6)  var<storage, read>        materials:     array<vec4f>;
@group(0) @binding(7)  var<uniform>              sun_light:     SunLight;
@group(0) @binding(8)  var                       base_color_tex: texture_2d_array<f32>;
@group(0) @binding(9)  var                       base_color_sampler: sampler;

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

    // Miss: terminate
    if (hit.did_hit == 0u) {
        ray.ray_active = 0u;
        ray_buffer[pixel_id] = ray;
        return;
    }

    let mat = unpackPTMaterial(&materials, hit.mat_id);

    var albedo = mat.albedo;
    if (mat.tex_layer >= 0) {
        let tex_col = textureSampleLevel(base_color_tex, base_color_sampler, hit.uv, mat.tex_layer, 0.0);
        albedo *= tex_col.xyz;
    }

    var rng = initRNG(pixel_id ^ (ray.bounce * 1009u), pt.frame_index);

    let N     = hit.normal;
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
    let a  = mat.roughness * mat.roughness;
    let a2 = a * a;
    let F0 = mix(vec3f(0.04), albedo, mat.metallic);
    let F_dir     = fresnelSchlickV(NdotV, F0);
    let diffuse_kD = (vec3f(1.0) - F_dir) * (1.0 - mat.metallic);

    // Russian Roulette
    let rr_max = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));
    let rr_survive = clamp(rr_max, 0.05, 1.0);
    if (ray.bounce >= 2u && rand(&rng) > rr_survive) {
        ray.ray_active = 0u;
        ray_buffer[pixel_id] = ray;
        return;
    }
    let rr_weight = select(1.0 / rr_survive, 1.0, ray.bounce < 2u);

    let spec_prob    = clamp(mat.metallic + 0.5 * (1.0 - mat.metallic), 0.0, 1.0);
    let use_specular = (rand(&rng) < spec_prob) && (mat.roughness < 0.95);

    var new_dir: vec3f;
    var new_throughput: vec3f;
    var is_specular: bool = false;

    if (use_specular) {
        let H      = sampleGGX(N, max(mat.roughness, 0.04), &rng);
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
        let G      = geometrySchlickGGX_pt(NdotV, mat.roughness)
                   * geometrySchlickGGX_pt(NdotL_, mat.roughness);
        let brdf_pdf = Fspec * G * VdotH / max(NdotH * NdotV, 0.0001);
        new_dir       = L;
        new_throughput = ray.throughput * brdf_pdf * rr_weight;
        is_specular   = (mat.roughness < 0.05);
    } else {
        let L         = sampleCosineHemisphere(N, &rng);
        new_dir       = L;
        new_throughput = ray.throughput * albedo * diffuse_kD * rr_weight;
    }

    // NEE — direct sun light
    if (sun_light.color.a > 0.5 && !is_specular) {
        let base_sun_dir = normalize(sun_light.direction.xyz);
        
        let up_vec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(base_sun_dir.x) > 0.9);
        let sun_tangent = normalize(cross(base_sun_dir, up_vec));
        let sun_bitangent = cross(base_sun_dir, sun_tangent);
        let r2 = rand2(&rng);
        let r = sqrt(r2.x) * 0.05; // 0.05 is sun angular radius
        let theta = 2.0 * PI * r2.y;
        let sun_dir = normalize(base_sun_dir + sun_tangent * r * cos(theta) + sun_bitangent * r * sin(theta));

        let NdotL_s  = max(dot(N, sun_dir), 0.0);
        if (NdotL_s > 0.0) {
            let H_s    = normalize(V + sun_dir);
            let NdotH_s = max(dot(N, H_s), 0.0);
            let VdotH_s = max(dot(V, H_s), 0.0);
            let F_s    = fresnelSchlickV(VdotH_s, F0);
            let G_s    = geometrySchlickGGX_pt(NdotV, mat.roughness)
                       * geometrySchlickGGX_pt(NdotL_s, mat.roughness);
            let D_s    = distributionGGX_pt(NdotH_s, a2);
            let spec_s = F_s * D_s * G_s / max(4.0 * NdotV * NdotL_s, 0.0001);
            let diff_s = (vec3f(1.0) - F_s) * (1.0 - mat.metallic) * albedo / PI;
            let sun_rad = sun_light.color.rgb * sun_light.direction.w;
            let Lo_sun  = (diff_s + spec_s) * sun_rad * NdotL_s;

            shadow.origin       = hit.pos + N * 0.002;
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
    ray.origin          = hit.pos + N * 0.001;
    ray.direction       = new_dir;
    ray.bounce         += 1u;
    ray.ray_active      = select(0u, 1u, ray.bounce < pt.max_bounces);
    ray.specular_bounce = select(0u, 1u, is_specular);

    if (max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z)) < 0.001) {
        ray.ray_active = 0u;
    }
    ray_buffer[pixel_id] = ray;
}
