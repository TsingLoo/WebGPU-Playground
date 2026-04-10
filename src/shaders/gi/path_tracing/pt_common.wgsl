// pt_common.wgsl
// Shared structs, PCG RNG, and utilities for Wavefront Path Tracing.
// Included AFTER common.wgsl and bvh.wgsl by all PT compute shaders.
// NOTE: bvh.wgsl already defines `struct Ray { origin, direction }`.

// ============================================================
// sRGB / Linear Color Space Conversion
// ============================================================
fn srgbToLinear(c: vec3f) -> vec3f {
    // Approximate sRGB EOTF: gamma 2.2 decode
    return pow(max(c, vec3f(0.0)), vec3f(2.2));
}

// ============================================================
// Path Tracing Structs
// ============================================================

// 64 bytes — one per pixel, persists across bounce passes
struct PTRay {
    origin:          vec3f,   // 12
    ior:             f32,     //  4  current medium IOR (1.0=air)
    direction:       vec3f,   // 12
    pixel_id:        u32,     //  4  flat pixel index (y * width + x)
    throughput:      vec3f,   // 12  spectral weight (starts at 1,1,1)
    bounce:          u32,     //  4  current bounce depth
    ray_active:      u32,     //  4  0=inactive, 1=active  (named ray_active to avoid WGSL reserved 'active')
    specular_bounce: u32,     //  4  1 = last event was purely specular
    _pad:            vec2u,   //  8
}; // 64 bytes

// 48 bytes — compact hit; shade reconstructs pos/normal/tangent/uv from BVH
struct HitRecord {
    dist:    f32,    //  4  ray t-parameter
    side:    f32,    //  4  +1.0 = front, -1.0 = back
    bary:    vec2f,  //  8  barycentric (x,y); z = 1-x-y
    idx0:    u32,    //  4  vertex index 0
    idx1:    u32,    //  4  vertex index 1
    idx2:    u32,    //  4  vertex index 2
    mat_id:  u32,    //  4  material id
    did_hit: u32,    //  4  0=miss, 1=hit
    _pad0:   u32,    //  4
    _pad1:   u32,    //  4
    _pad2:   u32,    //  4
}; // 48 bytes

// 48 bytes — NEE shadow ray, one per active pixel
struct ShadowRay {
    origin:       vec3f,  // 12
    max_dist:     f32,    //  4
    direction:    vec3f,  // 12
    pixel_id:     u32,    //  4
    Li:           vec3f,  // 12
    shadow_active: u32,   //  4  (named shadow_active to avoid WGSL reserved 'active')
}; // 48 bytes

// Unpacked material (from global materials buffer, 16 floats = 4 × vec4f per entry)
struct PTMaterial {
    albedo:           vec3f,
    alpha:            f32,
    roughness:        f32,
    metallic:         f32,
    tex_layer:        i32,     // index into baseColor texture array (-1 = none)
    transmission:     f32,     // 0=opaque, 1=fully transmissive
    ior:              f32,
    emissive:         vec3f,
    normal_tex_layer: i32,     // index into normal map texture array (-1 = none)
    mr_tex_layer:     i32,     // index into metallic-roughness texture array (-1 = none)
    alpha_cutoff:     f32,     // alpha test threshold
    alpha_mode:       u32,     // 0=OPAQUE, 1=MASK, 2=BLEND
};

// ============================================================
// NRC Training Struct
// ============================================================
struct NRCWavefrontTrainData {
    features: array<f32, 15>,
    _pad0: f32, // Padding to 64 bytes
    throughput: vec3f,
    _pad1: f32, // Padding to 80 bytes
    primary_radiance: vec3f,
    pixel_id: u32,
    is_active: u32,
    _pad2: vec3u, // Padding to 112 bytes? Wait, let's just make it cleanly aligned
};

// ============================================================
// PT Uniforms (per-frame constants)
// ============================================================
struct PTUniforms {
    width:          u32,
    height:         u32,
    frame_index:    u32,
    sample_count:   u32,
    max_bounces:    u32,
    clamp_radiance: f32,
    pixel_scale:    f32,
    restir_enabled: u32,  // 1 = ReSTIR active (shade skips NEE for bounce 0)
};

// ============================================================
// PCG Hash — fast pseudorandom
// ============================================================
fn pcgHash(state: u32) -> u32 {
    let s = state * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn initRNG(pixel_id: u32, frame_index: u32) -> u32 {
    return pcgHash(pixel_id ^ pcgHash(frame_index + 1u));
}

fn randU32(state: ptr<function, u32>) -> u32 {
    *state = pcgHash(*state);
    return *state;
}

fn rand(state: ptr<function, u32>) -> f32 {
    return f32(randU32(state)) / 4294967296.0;
}

fn rand2(state: ptr<function, u32>) -> vec2f {
    return vec2f(rand(state), rand(state));
}

// ============================================================
// Sampling helpers
// ============================================================

fn sampleCosineHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
    let r1 = rand(rng);
    let r2 = rand(rng);
    let cosTheta = sqrt(r1);
    let sinTheta = sqrt(max(0.0, 1.0 - r1));
    let phi = 2.0 * PI * r2;
    // Use Z axis when n is near Y axis to avoid degenerate cross product
    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let tangent   = normalize(cross(n, up));
    let bitangent = cross(n, tangent);
    return normalize(tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + n * cosTheta);
}

fn sampleGGX(n: vec3f, roughness: f32, rng: ptr<function, u32>) -> vec3f {
    let r1  = rand(rng);
    let r2  = rand(rng);
    let a   = roughness * roughness;
    let theta = atan(a * sqrt(r1) / sqrt(max(1.0 - r1, 1e-6)));
    let phi   = 2.0 * PI * r2;
    let lx = sin(theta) * cos(phi);
    let ly = sin(theta) * sin(phi);
    let lz = cos(theta);
    // Use Z axis when n is near Y axis to avoid degenerate cross product
    let up = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let tangent   = normalize(cross(n, up));
    let bitangent = cross(n, tangent);
    return normalize(tangent * lx + bitangent * ly + n * lz);
}

// ============================================================
// Material unpack (4 × vec4f = 16 floats per material entry)
fn unpackPTMaterial(
    mats: ptr<storage, array<vec4f>, read>,
    mat_id: u32
) -> PTMaterial {
    let base = mat_id * 4u;
    let r0 = (*mats)[base + 0u];
    let r1 = (*mats)[base + 1u];
    let r2 = (*mats)[base + 2u];
    let r3 = (*mats)[base + 3u];
    var m: PTMaterial;
    m.albedo           = r0.xyz;
    m.alpha            = r0.w;
    m.roughness        = r1.x;
    m.metallic         = r1.y;
    m.tex_layer        = i32(r1.z);
    m.transmission     = r1.w;
    m.ior              = r2.x;
    m.emissive         = r2.yzw;
    m.normal_tex_layer = i32(r3.x);
    m.mr_tex_layer     = i32(r3.y);
    m.alpha_cutoff     = r3.z;
    m.alpha_mode       = u32(r3.w);
    return m;
}

// ============================================================
// Fresnel & GGX helpers (suffixed _pt to avoid collision)
// ============================================================
fn fresnelSchlickV(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnelDielectric(cosI: f32, etaI: f32, etaT: f32) -> f32 {
    let sinT2 = (etaI / etaT) * (etaI / etaT) * (1.0 - cosI * cosI);
    if (sinT2 >= 1.0) { return 1.0; }
    let cosT = sqrt(max(0.0, 1.0 - sinT2));
    let rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    let rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return 0.5 * (rs * rs + rp * rp);
}

fn distributionGGX_pt(NdotH: f32, a2: f32) -> f32 {
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX_pt(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
