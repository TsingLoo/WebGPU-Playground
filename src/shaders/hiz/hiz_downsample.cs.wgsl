@group(0) @binding(0) var inDepthTex: texture_depth_2d;
@group(0) @binding(1) var outHizTexCopy: texture_storage_2d<rg32float, write>;

// Phase 0: Copy standard depth to Mip 0
@compute @workgroup_size(8, 8, 1)
fn copy_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(textureDimensions(inDepthTex));
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    // Read reverse-Z depth
    let depth = textureLoad(inDepthTex, vec2i(gid.xy), 0);
    
    // Store as MinMax Depth: R=far(min), G=near(max) in reverse-Z
    // Initially, both min and max are exactly the sampled depth.
    textureStore(outHizTexCopy, vec2i(gid.xy), vec4f(depth, depth, 0.0, 1.0));
}

@group(0) @binding(2) var inHizTex: texture_2d<f32>;
@group(0) @binding(3) var outHizTexDownsample: texture_storage_2d<rg32float, write>;

// Phase 1..N: Downsample Mip N-1 to Mip N
@compute @workgroup_size(8, 8, 1)
fn downsample_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(textureDimensions(outHizTexDownsample));
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    // Source texel top-left
    let srcCoord = vec2i(gid.xy) * 2;
    let sDims = vec2i(textureDimensions(inHizTex));
    
    // Sample 2x2 texels safely
    let d0 = textureLoad(inHizTex, clamp(srcCoord + vec2i(0, 0), vec2i(0), sDims - 1), 0).rg;
    let d1 = textureLoad(inHizTex, clamp(srcCoord + vec2i(1, 0), vec2i(0), sDims - 1), 0).rg;
    let d2 = textureLoad(inHizTex, clamp(srcCoord + vec2i(0, 1), vec2i(0), sDims - 1), 0).rg;
    let d3 = textureLoad(inHizTex, clamp(srcCoord + vec2i(1, 1), vec2i(0), sDims - 1), 0).rg;
    
    // In Reverse Z:
    // Far (R channel): lower value (towards 0), so we take minimum of the far values.
    // Near (G channel): higher value (towards 1), so we take maximum of the near values.
    let farZ = min(min(d0.r, d1.r), min(d2.r, d3.r));
    let nearZ = max(max(d0.g, d1.g), max(d2.g, d3.g));
    
    textureStore(outHizTexDownsample, vec2i(gid.xy), vec4f(farZ, nearZ, 0.0, 1.0));
}
