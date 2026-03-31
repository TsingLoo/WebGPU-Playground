// 3D Voxel Radiance Cascades - Optimized Inline Merge
// Evaluates a 3-level cascade directly into the octahedral irradiance atlas.

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> rcParams: RCUniforms;
@group(0) @binding(2) var voxelGrid: texture_3d<f32>;
@group(0) @binding(3) var envMap: texture_cube<f32>;
@group(0) @binding(4) var envSampler: sampler;
@group(0) @binding(5) var<uniform> sunLight: SunLight;
@group(0) @binding(6) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(7) var<uniform> vsmUniforms: VSMUniforms;
@group(0) @binding(8) var rcAtlasRead: texture_2d<f32>;
@group(0) @binding(9) var rcAtlasWrite: texture_storage_2d<rgba16float, write>;

const RC_RAYS_TOTAL: u32 = 84u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

fn fibonacciSphereDir(index: u32, total: u32) -> vec3f {
    let i = f32(index);
    let n = f32(total);
    let theta = 2.0 * PI * i / GOLDEN_RATIO;
    let phi = acos(1.0 - 2.0 * (i + 0.5) / n);
    return vec3f(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
}

// Global configurations
const VX = 128;
const VY = 128;
const VZ = 128;

// PCG hash based random utility
fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

fn rotationMatrix(axis: vec3f, angle: f32) -> mat3x3f {
    let s = sin(angle);
    let c = cos(angle);
    let oc = 1.0 - c;
    return mat3x3f(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
    );
}

// Workgroup shared memory for the current probe's 84 evaluated rays.
// .rgb = radiance, .a = hit flag (1.0 = hit or sky, -1.0 = miss)
var<workgroup> shared_radiance: array<vec4f, 84>;
var<workgroup> probe_rotation: mat3x3f;

// Evaluates lighting for a hit point, incorporating backface rejection.
fn evaluateLighting(pos: vec3f, voxelData: vec4f, rayDir: vec3f) -> vec3f {
    let normal = normalize(voxelData.rgb * 2.0 - 1.0);

    // Strictly reject backface hits. If rayDir and normal point in the same hemisphere,
    // the ray struck the INSIDE of the 3D geometry structure. Embedded probes must be pitch black.
    if (dot(rayDir, normal) > 0.0) {
        return vec3f(0.0);
    }
    
    var hitLighting = vec3f(0.0);
    
    if (sunLight.color.a > 0.5) {
        let sunShadow = calculateShadowVSMSimple(vsmPhysAtlas, vsmUniforms, sunLight, pos, normal);
        let sunL = normalize(sunLight.direction.xyz);
        let sunNdotL = max(dot(normal, sunL), 0.0);
        hitLighting += sunLight.color.rgb * sunLight.direction.w * sunNdotL * sunShadow;
    }
    
    hitLighting += vec3f(rcParams.params.z); // rc_ambient
    return hitLighting;
}

// Raymarching interval function
fn raymarchInterval(probeWorldPos: vec3f, rayDir: vec3f, startDist: f32, endDist: f32) -> vec4f {
    let vMin = vec3f(-15.0, 0.0, -10.0);
    let vMax = vec3f(15.0, 15.0, 10.0);
    let textureDims = vec3f(128.0, 128.0, 128.0);
    let vExtent = vMax - vMin;

    var tMin = 0.0;
    var tMax = 100.0;
    for (var j = 0; j < 3; j++) {
        if (abs(rayDir[j]) > 0.0001) {
            let invD = 1.0 / rayDir[j];
            let t0 = (vMin[j] - probeWorldPos[j]) * invD;
            let t1 = (vMax[j] - probeWorldPos[j]) * invD;
            tMin = max(tMin, min(t0, t1));
            tMax = min(tMax, max(t0, t1));
        } else {
            if (probeWorldPos[j] < vMin[j] || probeWorldPos[j] > vMax[j]) {
                tMax = -1.0; 
            }
        }
    }

    if (tMax >= tMin && tMax > 0.0) {
        let maxRayDist = min(tMax, endDist);
        let rayStart = max(startDist, tMin) + 0.05; 
        let stepSize = min(min(vExtent.x, vExtent.y), vExtent.z) / 256.0; 
        
        var t = rayStart;
        while (t <= maxRayDist) {
            let pos = probeWorldPos + rayDir * t;
            let uvw = (pos - vMin) / vExtent;
            let coord = vec3i(uvw * textureDims);
            
            if (coord.x >= 0 && coord.y >= 0 && coord.z >= 0 && coord.x < 128 && coord.y < 128 && coord.z < 128) {
                let voxel = textureLoad(voxelGrid, coord, 0);
                if (voxel.a > 0.5) {
                    let hitLighting = evaluateLighting(pos, voxel, rayDir);
                    return vec4f(vec3f(0.5) * hitLighting, t); // rgb=radiance, a=hitDist
                }
            }
            t += stepSize;
        }
    }
    return vec4f(0.0, 0.0, 0.0, -1.0); // Miss
}

// Map spherical direction to octahedral UV [0..1]
fn dirToOctahedral(dir: vec3f) -> vec2f {
    let absDir = abs(dir);
    let invL1 = 1.0 / (absDir.x + absDir.y + absDir.z);
    let p = dir.xy * invL1;
    if (dir.z <= 0.0) {
        let signP = sign(p);
        return (vec2f(1.0) - abs(p.yx)) * signP;
    }
    return p;
}

// Reconstruct direction from octahedral UV (UV in [-1..1] usually, but let's take [0..1] mapped to [-1..1])
fn octahedralToDir(uv: vec2f) -> vec3f {
    let p = uv * 2.0 - 1.0;
    var v = vec3f(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (v.z < 0.0) {
        let signV = sign(v.xy);
        v.x = (1.0 - abs(v.y)) * signV.x;
        v.y = (1.0 - abs(v.x)) * signV.y;
    }
    return normalize(v);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let gridX = u32(rcParams.grid_count.x);
    let probeIndex = wgid.y * gridX + wgid.x;
    
    let totalProbes = u32(rcParams.grid_count.w);
    if (probeIndex >= totalProbes) { return; }

    let gridY = u32(rcParams.grid_count.y);
    let pz = probeIndex / (gridX * gridY);
    let py = (probeIndex % (gridX * gridY)) / gridX;
    let px = probeIndex % gridX;

    // Probe position mapping
    let spacing = rcParams.grid_spacing;
    let minPos = rcParams.grid_min;
    let probeWorldPos = vec3f(
        minPos.x + f32(px) * spacing.x,
        minPos.y + f32(py) * spacing.y,
        minPos.z + f32(pz) * spacing.z
    );

    let threadIdx = lid.x; // 0..63
    
    if (threadIdx == 0u) {
        let frameCount = camera.frame_count;
        let seed1 = probeIndex + frameCount * 114514u;
        let seed2 = pcg_hash(seed1);
        let u1 = select(0.0, rand_float(seed1), rcParams.params.x > 0.0);
        let u2 = select(0.0, rand_float(seed2), rcParams.params.x > 0.0); // Only jitter if hysteresis > 0

        let z = u1 * 2.0 - 1.0;
        let r = max(0.0, sqrt(1.0 - z * z));
        let theta_axis = u2 * 2.0 * 3.14159265;
        let axis = vec3f(r * cos(theta_axis), r * sin(theta_axis), z);
        
        let angle = select(0.0, rand_float(pcg_hash(seed2)) * 2.0 * 3.14159265, rcParams.params.x > 0.0);
        probe_rotation = rotationMatrix(axis, angle);
    }
    workgroupBarrier();

    // Phase 1: Compute 84 rays across 64 threads. 
    // Thread i handles ray i. Threads 0..19 also handle rays 64..83.
    var rayIndices = array<u32, 2>(threadIdx, 999u);
    if (threadIdx < 20u) {
        rayIndices[1] = threadIdx + 64u;
    }

    for (var i = 0u; i < 2u; i++) {
        let rIdx = rayIndices[i];
        if (rIdx < 84u) {
            var startDist = 0.0;
            var endDist = 1000.0;
            var rayDir = vec3f(0.0);
            
            // Radiance Cascades configurations
            if (rIdx < 4u) {
                // Cascade 0
                rayDir = fibonacciSphereDir(rIdx, 4u);
                endDist = 4.0; 
            } else if (rIdx < 20u) {
                // Cascade 1
                rayDir = fibonacciSphereDir(rIdx - 4u, 16u);
                startDist = 4.0;
                endDist = 12.0;
            } else {
                // Cascade 2
                rayDir = fibonacciSphereDir(rIdx - 20u, 64u);
                startDist = 12.0;
                endDist = 1000.0;
            }
            
            // Apply spatiotemporal rotation jitter
            rayDir = normalize(probe_rotation * rayDir);

            // Raymarch
            let hit = raymarchInterval(probeWorldPos, rayDir, startDist, endDist);
            
            if (hit.w < 0.0 && rIdx >= 20u) {
                // Miss on Cascade 2, sample environment map!
                let envSample = textureSampleLevel(envMap, envSampler, rayDir, 0.0).rgb;
                shared_radiance[rIdx] = vec4f(min(envSample, vec3f(rcParams.params.y * 3.0)), 1.0);
            } else if (hit.w >= 0.0) {
                shared_radiance[rIdx] = vec4f(hit.rgb * rcParams.params.y, 1.0);
            } else {
                shared_radiance[rIdx] = vec4f(0.0, 0.0, 0.0, -1.0); // Missed on near cascades
            }
        }
    }

    // Barrier ensures all 84 rays are computed.
    workgroupBarrier();

    // Phase 2: Compute Octahedral integral. 
    // We update an 8x8 region inside the 10x10 atlas block.
    // threadIdx maps to the 8x8 texels.
    let texX = threadIdx % 8u;
    let texY = threadIdx / 8u;

    // UV in [0,1]
    let uv = vec2f(f32(texX) + 0.5, f32(texY) + 0.5) / 8.0;
    let evalDir = octahedralToDir(uv);

    // Merge cascades (Cosine weighted integral)
    // By strictly rejecting misses from near cascades, we effectively "merge" cascades 
    // without expensive recursion: outer cascades provide the missing data.
    var finalRad = vec3f(0.0);
    var weightSum = 0.0;
    
    // C0, C1, C2 inline merge
    for(var i=0u; i<84u; i++) {
        var dir: vec3f;
        if (i < 4u) { dir = fibonacciSphereDir(i, 4u); }
        else if (i < 20u) { dir = fibonacciSphereDir(i - 4u, 16u); }
        else { dir = fibonacciSphereDir(i - 20u, 64u); }
        
        // Rotate integration direction by the identical matrix used for trace
        dir = normalize(probe_rotation * dir);

        let w = max(0.0, dot(evalDir, dir));
        let radHit = shared_radiance[i];

        // Only incorporate rays that hit geometry OR hit the sky (C2 miss)
        if (radHit.w >= 0.0) {
            finalRad += radHit.rgb * w;
            weightSum += w;
        }
    }
    
    if (weightSum > 0.0) {
        finalRad /= weightSum;
    }
    
    // Temporal blend with hysteresis
    let texelsPerProbe = 10u;
    let probesPerRow = 800u;
    
    let baseTexelX = (probeIndex % probesPerRow) * texelsPerProbe + 1u;
    let baseTexelY = (probeIndex / probesPerRow) * texelsPerProbe + 1u;
    let atlasCoord = vec2i(i32(baseTexelX + texX), i32(baseTexelY + texY));
    
    let prevRad = textureLoad(rcAtlasRead, atlasCoord, 0).rgb;
    let blendedRad = mix(finalRad, prevRad, rcParams.params.x); // hysteresis
    
    textureStore(rcAtlasWrite, atlasCoord, vec4f(blendedRad, 1.0));

    workgroupBarrier();

    // Phase 3: Border Updates for seamless bilinear filtering
    // Border updates are handled in a separate pass (rc_border.cs.wgsl)
}
