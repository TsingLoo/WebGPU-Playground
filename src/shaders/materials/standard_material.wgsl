// ==========================================
// PBR Material Bindings and Evaluation
// ==========================================

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    has_normal_texture: f32,
    base_color_factor: vec4f,
    emissive_factor: vec3f,
    has_emissive_texture: f32,
}
@group(${bindGroup_material}) @binding(2) var<uniform> pbrParams: PBRParams;
@group(${bindGroup_material}) @binding(3) var metallicRoughnessTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(4) var metallicRoughnessTexSampler: sampler;
@group(${bindGroup_material}) @binding(5) var normalTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(6) var normalTexSampler: sampler;
@group(${bindGroup_material}) @binding(7) var emissiveTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(8) var emissiveTexSampler: sampler;

struct SurfaceData {
    albedo: vec3f,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    N: vec3f,
    emissive: vec3f,
    shadingModelId: f32, // SHADING_MODEL_PBR, SHADING_MODEL_UNLIT, etc.
}

// Unified material property evaluation
fn evaluateMaterial(uv: vec2f, geometryNormal: vec3f, tangentWorld: vec4f) -> SurfaceData {
    var surf: SurfaceData;
    
    // 1. Albedo & Alpha
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, uv) * pbrParams.base_color_factor;
    surf.albedo = diffuseColor.rgb;
    surf.alpha = diffuseColor.a;

    // 2. Metallic & Roughness (from ORM texture if available)
    surf.metallic = pbrParams.metallic;
    surf.roughness = pbrParams.roughness;
    if (pbrParams.has_mr_texture > 0.5) {
        let mrSample = textureSample(metallicRoughnessTex, metallicRoughnessTexSampler, uv);
        // glTF spec: G = roughness, B = metallic
        surf.roughness = surf.roughness * mrSample.g;
        surf.metallic = surf.metallic * mrSample.b;
    }
    // Clamp roughness to avoid division by zero / singularity
    surf.roughness = max(surf.roughness, 0.04);

    // 3. Normal Mapping
    var N = normalize(geometryNormal);
    if (pbrParams.has_normal_texture > 0.5) {
        let T_raw = tangentWorld.xyz - N * dot(tangentWorld.xyz, N);
        let T_len = length(T_raw);
        var T = vec3f(0.0);
        if (T_len > 0.001) {
            T = T_raw / T_len;
        } else {
            let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
            T = normalize(cross(N, refVec));
        }
        let handedness = select(tangentWorld.w, 1.0, abs(tangentWorld.w) < 0.5);
        let B = normalize(cross(N, T)) * handedness;
        let tbn = mat3x3f(T, B, N);
        let normalSample = textureSample(normalTex, normalTexSampler, uv).rgb;
        let tangentNormal = normalSample * 2.0 - 1.0;
        N = normalize(tbn * tangentNormal);
    }
    surf.N = N;

    // 4. Emissive Mapping
    surf.emissive = pbrParams.emissive_factor;
    if (pbrParams.has_emissive_texture > 0.5) {
        let emissiveColor = textureSample(emissiveTex, emissiveTexSampler, uv).rgb;
        surf.emissive = surf.emissive * emissiveColor;
    }

    surf.shadingModelId = SHADING_MODEL_PBR;

    return surf;
}
