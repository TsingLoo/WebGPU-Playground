// restir_common.wgsl
// ReSTIR DI — Shared structs, WRS helpers, target PDF, light sampling.
// Included AFTER pt_common.wgsl by all ReSTIR compute shaders.
// Reference: Bitterli et al. 2020 — Spatiotemporal Reservoir Resampling

// ============================================================
// ReSTIR Uniforms (per-frame constants, uploaded from CPU)
// ============================================================
struct ReSTIRUniforms {
    candidate_count:    u32,    // M = number of initial light candidates per pixel
    spatial_radius:     u32,    // R = pixel radius for spatial resampling
    spatial_count:      u32,    // K = number of spatial neighbors to sample
    temporal_max_M:     u32,    // temporal M clamp (e.g. 20 * current M)
    enabled:            u32,    // 1 = ReSTIR active, 0 = fallback to plain NEE
    frame_index:        u32,    // global frame counter (for RNG seeding)
    emissive_tri_count: u32,    // number of emissive triangles available
    _pad0:              u32,
};

// ============================================================
// Reservoir — 48 bytes per pixel
// ============================================================
struct Reservoir {
    // ----- Selected light sample -----
    sample_pos:     vec3f,      // 12  light position (or far point for directional)
    sample_type:    u32,        //  4  0=invalid, 1=sun, 2=emissive_tri

    sample_Le:      vec3f,      // 12  outgoing radiance from light toward surface
    sample_dist:    f32,        //  4  distance from surface to light

    sample_normal:  vec3f,      // 12  light surface normal (area lights)
    sample_pdf:     f32,        //  4  source PDF p_s with which sample was generated

    // ----- Reservoir state -----
    w_sum:          f32,        //  4  sum of weights (for WRS)
    M:              u32,        //  4  number of candidates seen
    W:              f32,        //  4  normalization weight = w_sum / (M * p_hat(y))
    _pad:           u32,        //  4

    // Total = 64 bytes
};

// ============================================================
// Helper: create an empty reservoir
// ============================================================
fn reservoirEmpty() -> Reservoir {
    var r: Reservoir;
    r.sample_pos    = vec3f(0.0);
    r.sample_type   = 0u;
    r.sample_Le     = vec3f(0.0);
    r.sample_dist   = 0.0;
    r.sample_normal = vec3f(0.0, 1.0, 0.0);
    r.sample_pdf    = 0.0;
    r.w_sum         = 0.0;
    r.M             = 0u;
    r.W             = 0.0;
    r._pad          = 0u;
    return r;
}

// ============================================================
// WRS Update: consider a new sample with weight w_i
// Returns true if the sample was accepted (reservoir updated)
// ============================================================
fn reservoirUpdate(
    r: ptr<function, Reservoir>,
    sample_pos:    vec3f,
    sample_type:   u32,
    sample_Le:     vec3f,
    sample_dist:   f32,
    sample_normal: vec3f,
    sample_pdf:    f32,
    w_i:           f32,          // weight = p_hat / p_source
    rng:           ptr<function, u32>
) -> bool {
    (*r).w_sum += w_i;
    (*r).M    += 1u;

    // Accept with probability w_i / w_sum
    let accept = rand(rng) * (*r).w_sum < w_i;
    if (accept) {
        (*r).sample_pos    = sample_pos;
        (*r).sample_type   = sample_type;
        (*r).sample_Le     = sample_Le;
        (*r).sample_dist   = sample_dist;
        (*r).sample_normal = sample_normal;
        (*r).sample_pdf    = sample_pdf;
    }
    return accept;
}

// ============================================================
// Reservoir Combine: merge reservoir 'other' into 'self'
// p_hat_of_other = target PDF evaluated at self's shading point for other's sample
// ============================================================
fn reservoirCombine(
    dst:   ptr<function, Reservoir>,
    other: Reservoir,
    p_hat_of_other: f32,
    rng:   ptr<function, u32>
) {
    let w_i = p_hat_of_other * other.W * f32(other.M);
    (*dst).w_sum += w_i;
    (*dst).M     += other.M;

    let accept = rand(rng) * (*dst).w_sum < w_i;
    if (accept) {
        (*dst).sample_pos    = other.sample_pos;
        (*dst).sample_type   = other.sample_type;
        (*dst).sample_Le     = other.sample_Le;
        (*dst).sample_dist   = other.sample_dist;
        (*dst).sample_normal = other.sample_normal;
        (*dst).sample_pdf    = other.sample_pdf;
    }
}

// ============================================================
// Finalize reservoir W after all updates/combinations
// p_hat = target PDF of the currently selected sample
// ============================================================
fn reservoirFinalize(r: ptr<function, Reservoir>, p_hat: f32) {
    if ((*r).M == 0u || p_hat < 1e-10) {
        (*r).W = 0.0;
    } else {
        (*r).W = (*r).w_sum / (f32((*r).M) * p_hat);
    }
}

// ============================================================
// Target PDF p̂(x): unshadowed contribution estimate
// Measures how valuable a light sample is for a given surface point.
// p̂ = |Le| * G_term = |Le| * max(cos_θ_surface, 0) * max(cos_θ_light, 0) / dist²
// For directional sun: G_term = max(cos_θ_surface, 0) (infinite distance, no falloff)
// ============================================================
fn targetPDF(
    surface_pos:    vec3f,
    surface_normal: vec3f,
    light_pos:      vec3f,
    light_Le:       vec3f,
    light_normal:   vec3f,
    light_type:     u32,
    light_dist:     f32,
) -> f32 {
    let luminance = dot(light_Le, vec3f(0.2126, 0.7152, 0.0722));
    if (luminance < 1e-8) { return 0.0; }

    if (light_type == 1u) {
        // Sun: directional light, treat light_pos as direction TO the light
        let L = normalize(light_pos - surface_pos);
        let cos_surface = max(dot(surface_normal, L), 0.0);
        return luminance * cos_surface;
    } else if (light_type == 2u) {
        // Emissive triangle: area light
        let to_light = light_pos - surface_pos;
        let dist2 = dot(to_light, to_light);
        if (dist2 < 1e-8) { return 0.0; }
        let dist = sqrt(dist2);
        let L = to_light / dist;
        let cos_surface = max(dot(surface_normal, L), 0.0);
        let cos_light   = max(dot(-L, light_normal), 0.0);
        return luminance * cos_surface * cos_light / dist2;
    }
    return 0.0;
}

// ============================================================
// Sun Light Sampling: sample with small angular radius for soft shadows
// ============================================================
fn sampleSunLight(
    sun_dir_normalized: vec3f,
    sun_color:          vec3f,
    sun_intensity:      f32,
    surface_pos:        vec3f,
    rng:                ptr<function, u32>
) -> Reservoir {
    var r = reservoirEmpty();

    // Soft sun disk: perturb direction slightly
    let up_vec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(sun_dir_normalized.x) > 0.9);
    let sun_tangent   = normalize(cross(sun_dir_normalized, up_vec));
    let sun_bitangent = cross(sun_dir_normalized, sun_tangent);
    let r2            = rand2(rng);
    let radius        = sqrt(r2.x) * 0.02; // ~1° angular radius
    let theta_angle   = 2.0 * PI * r2.y;
    let perturbed_dir = normalize(
        sun_dir_normalized
        + sun_tangent   * radius * cos(theta_angle)
        + sun_bitangent * radius * sin(theta_angle)
    );

    // "Position" is far away along the perturbed direction
    let far_pos = surface_pos + perturbed_dir * 10000.0;

    r.sample_pos    = far_pos;
    r.sample_type   = 1u; // sun
    r.sample_Le     = sun_color * sun_intensity;
    r.sample_dist   = 10000.0;
    r.sample_normal = -perturbed_dir; // light "faces" toward us
    r.sample_pdf    = 1.0; // uniform directional for a single sun
    r.M             = 1u;

    return r;
}

// ============================================================
// Emissive Triangle Sampling: uniformly sample one emissive tri
// emissive_indices: array of triangle indices (into bvh_indices)
// ============================================================
fn sampleEmissiveTriangle(
    surface_pos:       vec3f,
    emissive_tri_count: u32,
    rng:               ptr<function, u32>,
    // BVH data passed as function params to avoid global binding confusion
    tri_idx:           u32,  // pre-selected random emissive triangle index
    v0: vec3f, v1: vec3f, v2: vec3f,
    emissive_Le:       vec3f,
) -> Reservoir {
    var r = reservoirEmpty();

    // Random barycentric point on triangle
    let u1 = rand(rng);
    let u2 = rand(rng);
    let su1 = sqrt(u1);
    let bary = vec3f(1.0 - su1, su1 * (1.0 - u2), su1 * u2);

    let light_pos  = bary.x * v0 + bary.y * v1 + bary.z * v2;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let light_normal = normalize(cross(edge1, edge2));
    let tri_area = 0.5 * length(cross(edge1, edge2));

    // Source PDF = 1 / (num_emissive_tris * tri_area)
    let pdf_source = 1.0 / (f32(emissive_tri_count) * max(tri_area, 1e-8));

    let to_light = light_pos - surface_pos;
    let dist = length(to_light);

    r.sample_pos    = light_pos;
    r.sample_type   = 2u; // emissive triangle
    r.sample_Le     = emissive_Le;
    r.sample_dist   = dist;
    r.sample_normal = light_normal;
    r.sample_pdf    = pdf_source;
    r.M             = 1u;

    return r;
}

// ============================================================
// Geometry consistency check for temporal/spatial resampling
// Returns true if the two surface points are "similar enough"
// ============================================================
fn geometryConsistent(
    n1:      vec3f,   // surface normal at pixel 1
    n2:      vec3f,   // surface normal at pixel 2
    depth1:  f32,     // linear depth at pixel 1
    depth2:  f32,     // linear depth at pixel 2
) -> bool {
    let normal_threshold = 0.906; // cos(25°)
    let depth_threshold  = 0.1;   // 10% relative depth difference

    let n_dot = dot(n1, n2);
    if (n_dot < normal_threshold) { return false; }

    let depth_ratio = abs(depth1 - depth2) / max(depth1, 1e-4);
    if (depth_ratio > depth_threshold) { return false; }

    return true;
}
