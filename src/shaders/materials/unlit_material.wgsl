// ==========================================
// Unlit Material Bindings and Evaluation
// ==========================================

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    has_normal_texture: f32,
    base_color_factor: vec4f,
    _reserved: vec4f,
}
@group(${bindGroup_material}) @binding(2) var<uniform> pbrParams: PBRParams;

// Keep these bindings for layout compatibility, even if unused by Unlit
@group(${bindGroup_material}) @binding(3) var metallicRoughnessTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(4) var metallicRoughnessTexSampler: sampler;
@group(${bindGroup_material}) @binding(5) var normalTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(6) var normalTexSampler: sampler;

// Must match the SurfaceData struct expected by forward_plus.fs.wgsl
struct SurfaceData {
    albedo: vec3f,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    N: vec3f,
}

// Unified material property evaluation
fn evaluateMaterial(uv: vec2f, geometryNormal: vec3f, tangentWorld: vec4f) -> SurfaceData {
    var surf: SurfaceData;
    
    // 1. Albedo & Alpha
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, uv) * pbrParams.base_color_factor;
    surf.albedo = diffuseColor.rgb;
    surf.alpha = diffuseColor.a;

    // 2. Unlit properties
    surf.metallic = 0.0;
    surf.roughness = 1.0;

    // 3. No Normal Mapping
    surf.N = normalize(geometryNormal);

    return surf;
}
