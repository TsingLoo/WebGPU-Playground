struct FragmentInput {
    @location(2) uv: vec2f, 
}

@fragment
fn main(in: FragmentInput) {
    let rawAlpha = textureSample(diffuseTex, diffuseTexSampler, in.uv).a;
    let alpha = rawAlpha * pbrParams.base_color_factor.a;
    
    if (alpha < 0.5) {
        discard;
    }
}