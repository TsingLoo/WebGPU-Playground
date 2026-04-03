// DDGI Irradiance Probe Update
// Each workgroup processes one probe. Each thread handles one texel in the octahedral irradiance map.

@group(0) @binding(0) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(1) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(2) var<storage, read> rayData: array<vec4f>;
@group(0) @binding(3) var irradianceAtlasRead: texture_2d<f32>;
@group(0) @binding(4) var irradianceAtlasWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> probeData: array<vec4f>;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const IRRADIANCE_TEXELS: u32 = ${ddgiIrradianceTexels}u;
const IRRADIANCE_WITH_BORDER: u32 = IRRADIANCE_TEXELS + 2u;
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

var<workgroup> sharedIrradiance: array<vec3f, 100>; // 10x10

@compute @workgroup_size(10, 10, 1)
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

    let dstPixel = vec2i(
        probeCol * 10 + i32(lid.x),
        probeRow * 10 + i32(lid.y)
    );
    let linearIdx = lid.y * 10u + lid.x;

    let pData = probeData[probeIndex];
    if (pData.w > 0.5) { // sleeping
        let oldColor = textureLoad(irradianceAtlasRead, dstPixel, 0);
        textureStore(irradianceAtlasWrite, dstPixel, oldColor);
        return;
    }

    let isBorder = (lid.x == 0u || lid.x == 9u || lid.y == 0u || lid.y == 9u);

    if (!isBorder) {
        // Interior pixel: compute irradiance
        let texelX = lid.x - 1u;
        let texelY = lid.y - 1u;
        
        // Convert texel to octahedral UV [0,1]^2 for the 8x8 interior
        let octUV = vec2f(
            (f32(texelX) + 0.5) / 8.0,
            (f32(texelY) + 0.5) / 8.0
        );
        let texelDir = octDecode(octUV);

        var weightedIrradiance = vec3f(0.0);
        var totalWeight = 0.0;
        let rayBaseIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE;

        for (var r = 0u; r < DDGI_RAYS_PER_PROBE; r++) {
            let baseDir = fibonacciSphereDir(r, DDGI_RAYS_PER_PROBE);
            let rayDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);
            let rayResult = rayData[rayBaseIdx + r];
            let weight = max(dot(texelDir, rayDir), 0.0);
            if (weight > 0.0) {
                weightedIrradiance += rayResult.xyz * weight;
                totalWeight += weight;
            }
        }
        if (totalWeight > 0.0) {
            weightedIrradiance /= totalWeight;
        }

        // Encode to perceptual space
        let GAMMA = 1.0 / 5.0;
        let newEncoded = pow(max(weightedIrradiance, vec3f(0.0)), vec3f(GAMMA));

        let prevColor = textureLoad(irradianceAtlasRead, dstPixel, 0).rgb;
        let hysteresis = ddgi.hysteresis.x;
        let blended = mix(newEncoded, prevColor, hysteresis);

        sharedIrradiance[linearIdx] = blended;
    }

    workgroupBarrier();

    if (isBorder) {
        // Border pixel: mirror from interior
        var src_x = lid.x;
        var src_y = lid.y;

        if (lid.x == 0u && lid.y == 0u) { src_x = 1u; src_y = 1u; }
        else if (lid.x == 9u && lid.y == 0u) { src_x = 8u; src_y = 1u; }
        else if (lid.x == 0u && lid.y == 9u) { src_x = 1u; src_y = 8u; }
        else if (lid.x == 9u && lid.y == 9u) { src_x = 8u; src_y = 8u; }
        else if (lid.x == 0u) { src_x = 1u; src_y = 9u - lid.y; }
        else if (lid.x == 9u) { src_x = 8u; src_y = 9u - lid.y; }
        else if (lid.y == 0u) { src_y = 1u; src_x = 9u - lid.x; }
        else if (lid.y == 9u) { src_y = 8u; src_x = 9u - lid.x; }

        let srcIdx = src_y * 10u + src_x;
        sharedIrradiance[linearIdx] = sharedIrradiance[srcIdx];
    }
    
    // Everyone writes to atlas
    textureStore(irradianceAtlasWrite, dstPixel, vec4f(sharedIrradiance[linearIdx], 1.0));
}
