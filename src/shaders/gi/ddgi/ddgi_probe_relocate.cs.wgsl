

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
    var backfaceCount = 0u;

    var pushDir = vec3f(0.0);
    
    // We analyze the rays sent in the PREVIOUS pass
    // Note: positive dist = frontface/sky, negative dist = backface penalty
    for (var i = 0u; i < DDGI_RAYS_PER_PROBE; i++) {
        let rayIdx = probeIndex * DDGI_RAYS_PER_PROBE + i;
        let dist = rayData[rayIdx].w;
        let dir = randomRotation * sphericalFibonacci(i, DDGI_RAYS_PER_PROBE);

        if (dist < 0.0) {
            backfaceCount = backfaceCount + 1u;
        }

        let absDist = abs(dist);

        if (absDist < closestHitDist) {
            closestHitDist = absDist;
        }
        if (absDist > furthestHitDist) {
            furthestHitDist = absDist;
            furthestHitDir = dir;
        }

        // If distance is extremely small (frontface), we push away to avoid geometry
        if (dist > 0.0 && dist < 0.2) {
            closeHitCount = closeHitCount + 1u;
            // Weigh the push direction more strongly for very close hits
            let weight = 1.0 / max(dist, 0.01);
            pushDir -= dir * weight; 
        }
    }

    // State classification (Probe Sleep / Dead)
    // Industrial Standard (RTXGI / McGuire): If > 25% of rays hit backfaces, 
    // the probe is considered embedded inside solid geometry and should be put to sleep
    var newState = 0.0; // active
    let backfaceRatio = f32(backfaceCount) / f32(DDGI_RAYS_PER_PROBE);
    if (backfaceRatio > 0.25) {
        newState = 1.0; // sleep (embedded)
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
                // Move gradually based on maxOffset
                newOffset += normalize(pushDir) * (maxOffset * 0.02); 
            }
        } else if (closestHitDist > 0.4) {
            // Only spring back to origin if we are significantly far from ANY geometry
            // This creates a deadzone (0.2 < dist < 0.4) where the probe doesn't move,
            // preventing the probe from oscillating when random rays miss the geometry slightly.
            newOffset *= 0.98; // Slower, smoother spring back
        }
    }

    // Clamp offset
    if (length(newOffset) > maxOffset) {
        newOffset = normalize(newOffset) * maxOffset;
    }

    probeData[probeIndex] = vec4f(newOffset, newState);
}
