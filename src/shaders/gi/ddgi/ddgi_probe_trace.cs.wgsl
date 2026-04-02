// DDGI Probe Ray Tracing — World Space Coarse Scene Voxel DDA
// Each thread casts one ray for one probe

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(2) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(3) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read> bvhPositions: array<vec4f>;
@group(0) @binding(5) var<storage, read> bvhIndices: array<vec4u>;

struct MaterialData {
    baseColor: vec4f,
    roughness: f32,
    metallic: f32,
    pad0: f32,
    pad1: f32,
}
@group(0) @binding(6) var<storage, read> materials: array<MaterialData>;

@group(0) @binding(7) var ddgiIrrAtlas: texture_2d<f32>;
@group(0) @binding(8) var ddgiVisAtlas: texture_2d<f32>;
@group(0) @binding(9) var ddgiSampler: sampler;

@group(0) @binding(10) var envMap: texture_cube<f32>;
@group(0) @binding(11) var envSampler: sampler;
@group(0) @binding(12) var<storage, read_write> rayData: array<vec4f>; // [radiance.rgb, hitDist]
@group(0) @binding(13) var<uniform> sunLight: SunLight;
@group(0) @binding(14) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(15) var<uniform> vsmUniforms: VSMUniforms;
@group(0) @binding(16) var<storage, read> bvhUVs: array<vec4f>;
@group(0) @binding(17) var baseColorTexArray: texture_2d_array<f32>;
@group(0) @binding(18) var baseColorSampler: sampler;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

fn fibonacciSphereDir(index: u32, total: u32) -> vec3f {
    let i = f32(index);
    let n = f32(total);
    let theta = 2.0 * PI * i / GOLDEN_RATIO;
    let phi = acos(1.0 - 2.0 * (i + 0.5) / n);
    return vec3f(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
}

@compute @workgroup_size(${ddgiRaysPerProbe}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let rayIndex = gid.x;
    let probeIndex = i32(wgid.y);
    let totalProbes = ddgi.grid_count.w;

    if (probeIndex >= totalProbes || rayIndex >= DDGI_RAYS_PER_PROBE) {
        return;
    }

    let gridX = ddgi.grid_count.x;
    let gridY = ddgi.grid_count.y;
    let pz = probeIndex / (gridX * gridY);
    let py = (probeIndex % (gridX * gridY)) / gridX;
    let px = probeIndex % gridX;
    let probeWorldPos = ddgiProbePosition(vec3i(px, py, pz), ddgi);

    let baseDir = fibonacciSphereDir(rayIndex, DDGI_RAYS_PER_PROBE);
    let rotatedDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);

    var hitRadiance = vec3f(0.0);
    var hitDist = -1.0; 

    // BVH Software Raycast
    var ray: Ray;
    ray.origin = probeWorldPos;
    ray.direction = rotatedDir;

    let hit = bvhIntersectFirstHit(&bvhNodes, &bvhPositions, &bvhIndices, ray);

    if (hit.didHit) {
        hitDist = hit.dist;
        let matId = hit.indices.w;
        let mat = materials[matId];
        
        let pos = probeWorldPos + rotatedDir * hitDist;
        // Convert to properly outward-facing geometric normal
        let normal = normalize(hit.normal); 
        
        // --- REAL TEXTURE COLOR SAMPLING ---
        let uv0 = bvhUVs[hit.indices.x].xy;
        let uv1 = bvhUVs[hit.indices.y].xy;
        let uv2 = bvhUVs[hit.indices.z].xy;
        let hitUV = uv0 * hit.barycoord.x + uv1 * hit.barycoord.y + uv2 * hit.barycoord.z;
        
        var surfaceColor = mat.baseColor.rgb; 
        
        // Material pad0 stores the texture layer as f32 bits. bitcast to i32.
        let texLayer = bitcast<i32>(mat.pad0);
        if (texLayer >= 0) {
            let texColor = textureSampleLevel(baseColorTexArray, baseColorSampler, hitUV, texLayer, 0.0);
            
            // sRGB -> linear decode happens automatically via texture format if 'rgba8unorm-srgb' 
            // but we created texture_2d_array as 'rgba8unorm' to match shader float bindings without hassle,
            // so we must decode the sRGB texture color to linear properly here:
            let linearTexColor = pow(texColor.rgb, vec3f(2.2));
            surfaceColor = linearTexColor * mat.baseColor.rgb;
        }
        
        var hitLighting = vec3f(0.0);
        
        // Direct Sun evaluation
        if (sunLight.color.a > 0.5) {
            let sunShadow = calculateShadowVSMSimple(vsmPhysAtlas, vsmUniforms, sunLight, pos, normal);
            let sunL = normalize(sunLight.direction.xyz);
            let sunNdotL = max(dot(normal, sunL), 0.0);
            hitLighting += sunLight.color.rgb * sunLight.direction.w * sunNdotL * sunShadow;
        }
        
        // Recursive Indirect DDGI evaluation! (Infinite Bounces)
        // Using evaluateDDGI which correctly biases away from the hit surface
        var indirect = evaluateDDGI(pos, normal, -rotatedDir, ddgi, ddgiIrrAtlas, ddgiVisAtlas, ddgiSampler);
        
        // Filter out NaNs which can propagate through the network and cause black dots
        if (any(indirect != indirect) || any(indirect < vec3f(0.0))) {
            indirect = vec3f(0.0);
        }
        
        hitLighting += indirect;

        // Apply diffuse BRDF
        hitRadiance = surfaceColor * hitLighting / PI; 
    }

    if (hitDist < 0.0) {
        hitRadiance = textureSampleLevel(envMap, envSampler, rotatedDir, 0.0).rgb;
        hitRadiance = min(hitRadiance, vec3f(3.0));
        hitDist = 1000.0; // sky
    }

    let outputIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE + rayIndex;
    rayData[outputIdx] = vec4f(hitRadiance, hitDist);
}
