// Depth Copy Shader: reads depth24plus texture and writes raw depth to r32float render target
// Used by Frame Warp to copy the scene depth buffer into the reprojection history

@group(0) @binding(0) var depthTex: texture_depth_2d;

struct FragmentInput {
    @location(0) uv: vec2f,
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f {
    let flipped_uv = vec2f(in.uv.x, 1.0 - in.uv.y);
    let dims = textureDimensions(depthTex);
    let coord = vec2i(flipped_uv * vec2f(dims));
    let clamped = clamp(coord, vec2i(0), vec2i(dims) - vec2i(1));
    let depth = textureLoad(depthTex, clamped, 0);
    return vec4f(depth, 0.0, 0.0, 1.0);
}
