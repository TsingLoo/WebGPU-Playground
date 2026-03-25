@group(${bindGroup_scene}) @binding(0) var ssaoTex: texture_2d<f32>;

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) f32 {
    let fragCoord_i2 = vec2i(fragCoord.xy);
    let dims = textureDimensions(ssaoTex);
    
    var result = 0.0;
    var weightSum = 0.0;
    for (var x = -2; x <= 2; x += 1) {
        for (var y = -2; y <= 2; y += 1) {
            let offset = vec2i(x, y);
            let sample_coord = clamp(fragCoord_i2 + offset, vec2i(0), vec2i(dims) - vec2i(1));
            result += textureLoad(ssaoTex, sample_coord, 0).r;
            weightSum += 1.0;
        }
    }
    result /= weightSum;
    return result;
}
