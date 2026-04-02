

@group(0) @binding(0) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(1) var<uniform> randomRotation: mat3x3f;
@group(0) @binding(2) var<storage, read> rayData: array<vec4f>; // Just read radiance/hitDist
@group(0) @binding(3) var<storage, read_write> probeData: array<vec4f>;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

fn sphericalFibonacci(i: u32, n: u32) -> vec3f {
    let fraction = (f32(i) * GOLDEN_RATIO) % 1.0;
    let phi = 2.0 * PI * fraction;
    let cosTheta = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let sinTheta = sqrt(clamp(1.0 - cosTheta * cosTheta, 0.0, 1.0));
    return vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let probeIndex = global_id.x;
    if (probeIndex >= u32(ddgi.grid_count.w)) {
        return;
    }

    var currentData = probeData[probeIndex];
    let oldOffset = currentData.xyz;
    let oldState = currentData.w;

    var furthestHitDist = 0.0;
    var furthestHitDir = vec3f(0.0);
    var closestHitDist = 9999.0;
    var closeHitCount = 0u;

    var pushDir = vec3f(0.0);
    
    // We analyze the rays sent in the PREVIOUS pass
    // Note: hitDist > 0, 1000 = sky
    for (var i = 0u; i < DDGI_RAYS_PER_PROBE; i++) {
        let rayIdx = probeIndex * DDGI_RAYS_PER_PROBE + i;
        let dist = rayData[rayIdx].w;
        let dir = randomRotation * sphericalFibonacci(i, DDGI_RAYS_PER_PROBE);

        if (dist < closestHitDist) {
            closestHitDist = dist;
        }

        if (dist > furthestHitDist) {
            furthestHitDist = dist;
            furthestHitDir = dir;
        }

        // If distance is extremely small, it's hitting a backface or is embedded
        if (dist > 0.0 && dist < 0.2) {
            closeHitCount = closeHitCount + 1u;
            // Weigh the push direction more strongly for very close hits
            let weight = 1.0 / max(dist, 0.01);
            pushDir -= dir * weight; 
        }
    }

    // State classification (Probe Sleep)
    var newState = 0.0; // active
    // If almost all rays instantly hit geometry, we are likely embedded in solid space
    if (f32(closeHitCount) / f32(DDGI_RAYS_PER_PROBE) > 0.9) {
        newState = 1.0; // sleep
    }

    // Relocation
    var newOffset = oldOffset;
    let spacing = ddgi.grid_spacing.xyz;
    let minSpacing = min(min(spacing.x, spacing.y), spacing.z);
    let maxOffset = minSpacing * 0.45; // Max allowed displacement is slightly less than half spacing

    if (newState == 0.0) {
        if (closeHitCount > 0u) {
            // Push probe away from close surfaces
            if (length(pushDir) > 0.0001) {
                newOffset += normalize(pushDir) * (maxOffset * 0.05); // move gradually
            }
        } else {
            // Spring back to origin slowly if there's no threat
            newOffset *= 0.9;
        }
    }

    // Clamp offset
    if (length(newOffset) > maxOffset) {
        newOffset = normalize(newOffset) * maxOffset;
    }

    probeData[probeIndex] = vec4f(newOffset, newState);
}
