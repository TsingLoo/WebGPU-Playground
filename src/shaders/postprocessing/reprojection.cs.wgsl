// Frame Warp Reprojection Compute Shader
// Takes the previous frame's color + depth, reprojects to the latest camera view,
// fills disocclusion holes, and outputs the warped image.

struct ReprojUniforms {
    prev_view_proj: mat4x4f,      // VP matrix used to render the history buffer
    curr_inv_view_proj: mat4x4f,  // inverse of the LATEST VP (current mouse input)
    screen_width: f32,
    screen_height: f32,
    warp_enabled: u32,
    near_plane: f32,
    far_plane: f32,
}

@group(0) @binding(0) var<uniform> reproj: ReprojUniforms;
@group(0) @binding(1) var historyColor: texture_2d<f32>;
@group(0) @binding(2) var historyDepth: texture_2d<f32>;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

// Reconstruct world position from screen UV + depth using inverse view-projection
fn reconstructWorldPos(uv: vec2f, depth: f32) -> vec3f {
    // UV [0,1] → NDC [-1,1], note Y is flipped
    let ndc = vec4f(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );
    // Transform from NDC back to world space using the HISTORY frame's inverse VP
    // We need the inverse of prevViewProj to go from history NDC → world
    // But we have currInvViewProj. We need to think about this differently.
    //
    // The history buffer was rendered with prevViewProj.
    // We want to show it from the perspective of currViewProj.
    //
    // For each pixel in OUTPUT (current view):
    //   1. Unproject using currInvViewProj → world position
    //   2. Project using prevViewProj → where was this point in the history buffer?
    //   3. Sample history color at that location
    //
    // This is a REVERSE mapping approach (iterate output pixels, sample input)
    let worldPos = reproj.curr_inv_view_proj * ndc;
    return worldPos.xyz / worldPos.w;
}

// Linearize reverse-Z depth to view-space distance
fn linearizeDepth(d: f32, near: f32, far: f32) -> f32 {
    // Reverse-Z: z_ndc=1 at near, z_ndc=0 at far
    // For reverse-Z perspective: linear = near * far / (far * d + near * (1-d))
    // Simplified: near / d (when far >> near with reverse-Z)
    if (d <= 0.0) { return far; }
    return near * far / (far * d + near * (1.0 - d));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u32(reproj.screen_width), u32(reproj.screen_height));

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let pixelCoord = vec2i(gid.xy);

    // If warp is disabled, just pass through the history color
    if (reproj.warp_enabled == 0u) {
        let color = textureLoad(historyColor, pixelCoord, 0);
        textureStore(outputTex, pixelCoord, color);
        return;
    }

    // ============================
    // Reverse mapping: for each OUTPUT pixel, find where to sample in the HISTORY buffer
    // ============================

    // Output pixel UV (in current view)
    let uv = vec2f(
        (f32(gid.x) + 0.5) / reproj.screen_width,
        (f32(gid.y) + 0.5) / reproj.screen_height
    );

    // We need a depth value to unproject. Use the history depth at this pixel as initial guess.
    let histDepthRaw = textureLoad(historyDepth, pixelCoord, 0).r;

    // Unproject current pixel to world space using current camera's inverse VP
    let ndcCurr = vec4f(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        histDepthRaw,
        1.0
    );
    let worldPos4 = reproj.curr_inv_view_proj * ndcCurr;
    let worldPos = worldPos4.xyz / worldPos4.w;

    // Project world position into history buffer's screen space
    let histClip = reproj.prev_view_proj * vec4f(worldPos, 1.0);
    let histNDC = histClip.xyz / histClip.w;

    // NDC → UV for history buffer sampling
    let histUV = vec2f(
        histNDC.x * 0.5 + 0.5,
        1.0 - (histNDC.y * 0.5 + 0.5)
    );

    // Check if the reprojected position is within bounds
    let inBounds = histUV.x >= 0.0 && histUV.x <= 1.0 && histUV.y >= 0.0 && histUV.y <= 1.0;

    if (inBounds) {
        // Sample history color at the reprojected position
        let sampleCoord = vec2i(
            i32(histUV.x * reproj.screen_width),
            i32(histUV.y * reproj.screen_height)
        );

        // Clamp to valid range
        let clampedCoord = clamp(sampleCoord, vec2i(0), vec2i(dims) - vec2i(1));

        // Depth consistency check: compare the depth at the reprojected position
        // with what we'd expect. If they differ significantly, it's a disocclusion.
        let sampledDepth = textureLoad(historyDepth, clampedCoord, 0).r;
        let expectedDepth = histNDC.z;

        let linearSampled = linearizeDepth(sampledDepth, reproj.near_plane, reproj.far_plane);
        let linearExpected = linearizeDepth(expectedDepth, reproj.near_plane, reproj.far_plane);

        let depthDiff = abs(linearSampled - linearExpected) / max(linearSampled, 0.001);

        if (depthDiff < 0.1) {
            // Depth is consistent → valid reprojection
            let color = textureLoad(historyColor, clampedCoord, 0);
            textureStore(outputTex, pixelCoord, color);
        } else {
            // Disocclusion detected → try to fill from nearby valid samples
            var bestColor = vec4f(0.0);
            var bestWeight = 0.0;

            // 3×3 neighborhood search for valid samples
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    let neighbor = clampedCoord + vec2i(dx, dy);
                    let nClamped = clamp(neighbor, vec2i(0), vec2i(dims) - vec2i(1));
                    let nDepth = textureLoad(historyDepth, nClamped, 0).r;
                    let nLinear = linearizeDepth(nDepth, reproj.near_plane, reproj.far_plane);
                    let nDiff = abs(nLinear - linearExpected) / max(nLinear, 0.001);

                    if (nDiff < 0.15) {
                        let dist = f32(dx * dx + dy * dy);
                        let w = exp(-dist * 0.5);
                        bestColor += textureLoad(historyColor, nClamped, 0) * w;
                        bestWeight += w;
                    }
                }
            }

            if (bestWeight > 0.0) {
                textureStore(outputTex, pixelCoord, bestColor / bestWeight);
            } else {
                // Complete disocclusion — use the history color at the OUTPUT pixel directly
                // (slightly wrong but better than a black hole)
                let fallback = textureLoad(historyColor, pixelCoord, 0);
                textureStore(outputTex, pixelCoord, fallback);
            }
        }
    } else {
        // Out of bounds — use history color at output pixel as fallback
        let fallback = textureLoad(historyColor, pixelCoord, 0);
        textureStore(outputTex, pixelCoord, fallback);
    }
}
