@group(2) @binding(0) var diffuseTex: texture_2d<f32>;
@group(2) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput {
    @location(2) uv: vec2f, 
}

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    has_normal_texture: f32,
    base_color_factor: vec4f,
    _reserved: vec4f,
}
@group(2) @binding(2) var<uniform> pbrParams: PBRParams;

@fragment
fn main(in: FragmentInput) {
    let rawAlpha = textureSample(diffuseTex, diffuseTexSampler, in.uv).a;
    let alpha = rawAlpha * pbrParams.base_color_factor.a;
    
    if (alpha < 0.5) {
        discard;
    }
}