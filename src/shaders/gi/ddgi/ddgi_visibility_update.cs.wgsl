// DDGI Visibility (Depth) Probe Update
// Each workgroup processes one probe. Each thread handles one texel in the visibility map.
// Stores (mean_distance, mean_distance^2) for Chebyshev visibility test.

@group(0) @binding(0) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(1) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(2) var<storage, read> rayData: array<vec4f>;
@group(0) @binding(3) var visibilityAtlasRead: texture_2d<f32>;
@group(0) @binding(4) var visibilityAtlasWrite: texture_storage_2d<rgba16float, write>;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const VISIBILITY_TEXELS: u32 = ${ddgiVisibilityTexels}u;
const VISIBILITY_WITH_BORDER: u32 = VISIBILITY_TEXELS + 2u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

fn fibonacciSphereDir(index: u32, total: u32) -> vec3f {
    let i = f32(index);
    let n = f32(total);
    let theta = 2.0 * PI * i / GOLDEN_RATIO;
    let phi = acos(1.0 - 2.0 * (i + 0.5) / n);
    return vec3f(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );
}

var<workgroup> sharedDist: array<vec2f, 256>; // 16x16 interior

@compute @workgroup_size(${ddgiVisibilityTexels}, ${ddgiVisibilityTexels}, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let probeIndex = i32(wgid.z);
    let totalProbes = ddgi.grid_count.w;
    if (probeIndex >= totalProbes) { return; }

    let probesPerRow = ddgi.grid_count.x;
    let probeRow = probeIndex / probesPerRow;
    let probeCol = probeIndex % probesPerRow;

    // lid.x and lid.y are in [0..15].
    let texelX = lid.x;
    let texelY = lid.y;
    let linearIdx = lid.y * 16u + lid.x;

    let dstPixel = vec2i(
        probeCol * 18 + 1 + i32(texelX),
        probeRow * 18 + 1 + i32(texelY)
    );

    let octUV = vec2f(
        (f32(texelX) + 0.5) / 16.0,
        (f32(texelY) + 0.5) / 16.0
    );
    let texelDir = octDecode(octUV);

    var weightedDist = 0.0;
    var weightedDist2 = 0.0;
    var totalWeight = 0.0;

    let rayBaseIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE;

    for (var r = 0u; r < DDGI_RAYS_PER_PROBE; r++) {
        let baseDir = fibonacciSphereDir(r, DDGI_RAYS_PER_PROBE);
        let rayDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);

        let rayResult = rayData[rayBaseIdx + r];
        let dist = rayResult.w;

        let weight = max(dot(texelDir, rayDir), 0.0);

        if (weight > 0.0) {
            weightedDist += dist * weight;
            weightedDist2 += dist * dist * weight;
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0) {
        weightedDist /= totalWeight;
        weightedDist2 /= totalWeight;
    }

    // Hysteresis blending
    let prevVal = textureLoad(visibilityAtlasRead, dstPixel, 0).rg;
    let hysteresis = ddgi.hysteresis.y;
    let blendedDist = mix(weightedDist, prevVal.x, hysteresis);
    let blendedDist2 = mix(weightedDist2, prevVal.y, hysteresis);

    // Save to shared memory
    sharedDist[linearIdx] = vec2f(blendedDist, blendedDist2);

    // Write interior directly
    textureStore(visibilityAtlasWrite, dstPixel, vec4f(blendedDist, blendedDist2, 0.0, 1.0));

    // Barrier before reading for border copy
    workgroupBarrier();

    // 68 border pixels for an 18x18 block containing a 16x16 interior
    if (linearIdx < 68u) {
        var bx: u32;
        var by: u32;
        if (linearIdx < 18u) {
            bx = linearIdx;
            by = 0u;
        } else if (linearIdx < 36u) {
            bx = linearIdx - 18u;
            by = 17u;
        } else if (linearIdx < 52u) {
            bx = 0u;
            by = 1u + (linearIdx - 36u);
        } else {
            bx = 17u;
            by = 1u + (linearIdx - 52u);
        }

        var src_x = bx;
        var src_y = by;

        if (bx == 0u && by == 0u) { src_x = 1u; src_y = 1u; }
        else if (bx == 17u && by == 0u) { src_x = 16u; src_y = 1u; }
        else if (bx == 0u && by == 17u) { src_x = 1u; src_y = 16u; }
        else if (bx == 17u && by == 17u) { src_x = 16u; src_y = 16u; }
        else if (bx == 0u) { src_x = 1u; src_y = 17u - by; }
        else if (bx == 17u) { src_x = 16u; src_y = 17u - by; }
        else if (by == 0u) { src_y = 1u; src_x = 17u - bx; }
        else if (by == 17u) { src_y = 16u; src_x = 17u - bx; }

        // Convert [1..16] padded coordinate down to [0..15] shared memory coordinate
        let int_x = src_x - 1u;
        let int_y = src_y - 1u;
        
        let srcLinearIdx = int_y * 16u + int_x;
        let bVal = sharedDist[srcLinearIdx];

        let bDstX = probeCol * 18 + i32(bx);
        let bDstY = probeRow * 18 + i32(by);

        textureStore(visibilityAtlasWrite, vec2i(bDstX, bDstY), vec4f(bVal, 0.0, 1.0));
    }
}
