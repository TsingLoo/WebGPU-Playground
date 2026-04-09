  // CHECKITOUT: code that you add here will be prepended to all shaders

const PI = 3.14159265359;

// Shading Model IDs (stored in G-Buffer specular.a, normalized to [0,1])
// Use f32 values compatible with rgba8unorm (0.0 = 0, 1/255 ≈ 0.004, etc.)
const SHADING_MODEL_PBR: f32      = 0.0;   // Standard PBR Cook-Torrance
const SHADING_MODEL_UNLIT: f32    = 1.0;   // Unlit / emissive only

struct Light {
    pos: vec3f,
    color: vec3f
}

struct LightSet {
    numLights: u32,
    lights: array<Light>
}

struct TileMeta {
    offset: u32,
    count: u32,
};

struct LightIndexList {
    counter: atomic<u32>,
    indices: array<u32>,
};

struct LightIndexListReadOnly {
    counter: u32,
    indices: array<u32>,
};

struct CameraUniforms {
    view_proj_mat: mat4x4f,
    inv_proj_mat: mat4x4f,
    proj_mat: mat4x4f,
    view_mat: mat4x4f,
    near_plane: f32,
    far_plane: f32,
    frame_count: u32,
    _pad1: f32,
    camera_pos: vec4f,
    inv_view_proj_mat: mat4x4f,
}

struct ClusterSet {
    screen_width: u32,
    screen_height: u32,
    num_clusters_X: u32,
    num_clusters_Y: u32,
    num_clusters_Z: u32
}

// Shared cluster lookup — maps screen position + world position to a linear cluster index
fn getClusterIndex(
    screen_pos: vec2f,
    pos_world: vec3f,
    cam: CameraUniforms,
    cs: ClusterSet
) -> u32 {
    let screen_size_cluster_x = f32(cs.screen_width) / f32(cs.num_clusters_X);
    let screen_size_cluster_y = f32(cs.screen_height) / f32(cs.num_clusters_Y);

    let clusterid_x = u32(screen_pos.x / screen_size_cluster_x);
    let clusterid_y_unflipped = u32(screen_pos.y / screen_size_cluster_y);
    let clusterid_y = clamp((cs.num_clusters_Y - 1u) - clusterid_y_unflipped, 0u, cs.num_clusters_Y - 1u);

    let z_view = (cam.view_mat * vec4f(pos_world, 1.0)).z;
    let clamped_Z = clamp(-z_view, cam.near_plane, cam.far_plane);

    let logFN = log(cam.far_plane / cam.near_plane);
    let SCALE = f32(cs.num_clusters_Z) / logFN;
    let BIAS = SCALE * log(cam.near_plane);
    let slice = log(clamped_Z) * SCALE - BIAS;
    let cluster_z = clamp(u32(floor(slice)), 0u, cs.num_clusters_Z - 1u);

    return cluster_z * (cs.num_clusters_X * cs.num_clusters_Y)
         + clusterid_y * cs.num_clusters_X
         + clusterid_x;
}

// ============================
// Attenuation
// ============================
fn rangeAttenuation(distance: f32) -> f32 {
    return clamp(1.f - pow(distance / ${lightRadius}, 4.f), 0.f, 1.f) / (distance * distance);
}

// ============================
// Cook-Torrance PBR BRDF
// ============================

// Trowbridge-Reitz GGX Normal Distribution Function
fn distributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX Geometry function (single direction)
fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method combining geometry for both view and light directions
fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick approximation
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness (for IBL ambient specular)
fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3f, roughness: f32) -> vec3f {
    return F0 + (max(vec3f(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================
// PBR Point Light Contribution (Cook-Torrance)
// ============================
fn calculateLightContribPBR(light: Light, posWorld: vec3f, N: vec3f, V: vec3f, albedo: vec3f, metallic: f32, roughness: f32) -> vec3f {
    let vecToLight = light.pos - posWorld;
    let distToLight = length(vecToLight);
    let L = normalize(vecToLight);
    let H = normalize(V + L);

    let attenuation = rangeAttenuation(distToLight);
    let radiance = light.color * attenuation;

    // F0 for dielectrics is 0.04, for metals it's the albedo
    let F0 = mix(vec3f(0.04), albedo, metallic);

    // Cook-Torrance specular BRDF
    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    // Energy conservation: kS is what's reflected, kD is what's refracted (diffuse)
    let kS = F;
    // Metals have no diffuse
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    let NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// Simple Lambert fallback (for backward compatibility)
fn calculateLightContrib(light: Light, posWorld: vec3f, nor: vec3f) -> vec3f {
    let vecToLight = light.pos - posWorld;
    let distToLight = length(vecToLight);
    let lambert = max(dot(nor, normalize(vecToLight)), 0.f);
    return light.color * lambert * rangeAttenuation(distToLight);
}

// ============================
// Directional Sun Light
// ============================
struct SunLight {
    direction: vec4f,       // xyz = direction TO light (normalized), w = intensity
    color: vec4f,           // rgb = color, a = enabled (0 or 1)
    light_vp: mat4x4f,      // light-space view-projection matrix (unused now, kept for layout compat)
    shadow_params: vec4f,    // x = 1/shadow_map_size, y = bias, z = 0, w = 0
    volumetric_params: vec4f, // x = intensity, y = heightFalloff, z = heightScale, w = maxDist
}

fn calculateSunLightPBR(
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
    V: vec3f,
    albedo: vec3f,
    metallic: f32,
    roughness: f32,
    shadow: f32
) -> vec3f {
    if (sun.color.a < 0.5) { return vec3f(0.0); }

    let L = normalize(sun.direction.xyz);
    let H = normalize(V + L);
    let intensity = sun.direction.w;
    let radiance = sun.color.rgb * intensity;

    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);

    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    let NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL * shadow;
}

// ============================
// Virtual Shadow Map (VSM)
// ============================
struct VSMUniforms {
    clipmap_vp: array<mat4x4f, 8>,   // VP matrix per clipmap level (max 8)
    inv_view_proj: mat4x4f,          // camera inverse view-projection (for mark pass)
    clipmap_count: u32,
    pages_per_axis: u32,             // 128
    phys_atlas_size: u32,            // 4096
    phys_pages_per_axis: u32,        // 32
}

// Select best clipmap level based on world position → light NDC coverage
// Uses an inset margin to prevent level oscillation at boundaries
fn vsmSelectClipmapLevel(
    vsm: VSMUniforms,
    posWorld: vec3f,
) -> u32 {
    for (var level = 0u; level < vsm.clipmap_count; level++) {
        let lightClip = vsm.clipmap_vp[level] * vec4f(posWorld, 1.0);
        let lightNDC = lightClip.xyz / lightClip.w;

        // Inset margin prevents flickering at level boundaries
        let margin = 0.9;
        if (lightNDC.x >= -margin && lightNDC.x <= margin &&
            lightNDC.y >= -margin && lightNDC.y <= margin &&
            lightNDC.z >= 0.0    && lightNDC.z <= 1.0) {
            return level;
        }
    }
    return vsm.clipmap_count; // No valid level
}

// Compute atlas tile offset and size for a clipmap level (square grid layout)
fn vsmTileInfo(vsm: VSMUniforms, level: u32) -> vec3u {
    // Returns (xOffset, yOffset, tileSize)
    let gridCols = u32(ceil(sqrt(f32(vsm.clipmap_count))));
    let tileSize = vsm.phys_atlas_size / gridCols;
    let col = level % gridCols;
    let row = level / gridCols;
    return vec3u(col * tileSize, row * tileSize, tileSize);
}

// Calculate shadow using VSM (Virtual Shadow Map) with clipmap atlas
// Uses textureLoad with bilinear PCF for smooth shadow edges
fn calculateShadowVSM(
    physAtlas: texture_depth_2d,
    vsm: VSMUniforms,
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
) -> f32 {
    // Normal bias to avoid shadow acne
    let bias = sun.shadow_params.y;
    let biasedPos = posWorld + N * bias;

    // Select finest valid clipmap level
    let level = vsmSelectClipmapLevel(vsm, biasedPos);

    // Compute UV and depth for sampling (use level 0 as safe fallback)
    let safeLevel = min(level, vsm.clipmap_count - 1u);
    let lightClip = vsm.clipmap_vp[safeLevel] * vec4f(biasedPos, 1.0);
    let lightNDC = lightClip.xyz / lightClip.w;

    let uv = vec2f(lightNDC.x * 0.5 + 0.5, -lightNDC.y * 0.5 + 0.5);
    let depth = lightNDC.z;

    // Square grid layout: each level gets a square tile
    let tile = vsmTileInfo(vsm, safeLevel);
    let tileX = f32(tile.x);
    let tileY = f32(tile.y);
    let tileSz = f32(tile.z);

    // Sub-texel coordinates for bilinear weighting
    let texCoordX = tileX + uv.x * tileSz - 0.5;
    let texCoordY = tileY + uv.y * tileSz - 0.5;
    let baseX = i32(floor(texCoordX));
    let baseY = i32(floor(texCoordY));
    let fracX = texCoordX - floor(texCoordX);
    let fracY = texCoordY - floor(texCoordY);

    let safeDepth = clamp(depth, 0.0, 1.0);

    // Tile boundary clamps to prevent PCF bleeding into adjacent tiles
    let tileMinX = i32(tile.x);
    let tileMinY = i32(tile.y);
    let tileMaxX = i32(tile.x + tile.z) - 1;
    let tileMaxY = i32(tile.y + tile.z) - 1;

    // Bilinear-interpolated PCF: 5×5 kernel with Gaussian weighting
    var shadow = 0.0;
    var totalWeight = 0.0;

    for (var ky = -2; ky <= 2; ky++) {
        for (var kx = -2; kx <= 2; kx++) {
            let dist = f32(kx * kx + ky * ky);
            let w = exp(-dist * 0.3);

            let ox = baseX + kx;
            let oy = baseY + ky;

            // Clamp to tile boundaries (not whole atlas) to prevent cross-level bleeding
            let s00 = textureLoad(physAtlas, vec2i(clamp(ox,     tileMinX, tileMaxX), clamp(oy,     tileMinY, tileMaxY)), 0);
            let s10 = textureLoad(physAtlas, vec2i(clamp(ox + 1, tileMinX, tileMaxX), clamp(oy,     tileMinY, tileMaxY)), 0);
            let s01 = textureLoad(physAtlas, vec2i(clamp(ox,     tileMinX, tileMaxX), clamp(oy + 1, tileMinY, tileMaxY)), 0);
            let s11 = textureLoad(physAtlas, vec2i(clamp(ox + 1, tileMinX, tileMaxX), clamp(oy + 1, tileMinY, tileMaxY)), 0);

            let c00 = select(0.0, 1.0, safeDepth <= s00);
            let c10 = select(0.0, 1.0, safeDepth <= s10);
            let c01 = select(0.0, 1.0, safeDepth <= s01);
            let c11 = select(0.0, 1.0, safeDepth <= s11);

            let bilinear = mix(mix(c00, c10, fracX), mix(c01, c11, fracX), fracY);

            shadow += bilinear * w;
            totalWeight += w;
        }
    }

    shadow /= totalWeight;

    // If sun is disabled or position is outside all clipmap levels, return fully lit
    let valid = select(0.0, 1.0, sun.color.a >= 0.5 && level < vsm.clipmap_count);
    return mix(1.0, shadow, valid);
}

// Simple VSM shadow for compute shaders (DDGI probes) — no comparison sampler
fn calculateShadowVSMSimple(
    physAtlas: texture_depth_2d,
    vsm: VSMUniforms,
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
) -> f32 {
    if (sun.color.a < 0.5) { return 1.0; }

    let bias = sun.shadow_params.y;
    let biasedPos = posWorld + N * bias;

    let level = vsmSelectClipmapLevel(vsm, biasedPos);
    if (level >= vsm.clipmap_count) { return 1.0; }

    let lightClip = vsm.clipmap_vp[level] * vec4f(biasedPos, 1.0);
    let lightNDC = lightClip.xyz / lightClip.w;
    let uv = vec2f(lightNDC.x * 0.5 + 0.5, -lightNDC.y * 0.5 + 0.5);
    let depth = lightNDC.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth > 1.0) {
        return 1.0;
    }

    // Square grid layout (same as calculateShadowVSM)
    let tile = vsmTileInfo(vsm, level);
    let ssi = vec2i(
        i32(f32(tile.x) + uv.x * f32(tile.z)),
        i32(f32(tile.y) + uv.y * f32(tile.z))
    );
    let shadowDepth = textureLoad(physAtlas, ssi, 0);
    return select(0.0, 1.0, depth <= shadowDepth + 0.005);
}

// ============================
// DDGI
// ============================

// ============================
// DDGI
// ============================

struct DDGIUniforms {
    grid_count: vec4i,       // x, y, z, total
    grid_min: vec4f,         // world-space min corner
    grid_max: vec4f,         // world-space max corner
    grid_spacing: vec4f,     // spacing per axis, w = rays per probe
    irradiance_texel_size: vec4f, // texel_dim, texel_dim_with_border, atlas_width, atlas_height
    visibility_texel_size: vec4f, // texel_dim, texel_dim_with_border, atlas_width, atlas_height
    hysteresis: vec4f,       // irradiance_hysteresis, visibility_hysteresis, normal_bias, view_bias
    ddgi_enabled: vec4f,     // x = enabled (0 or 1), y = debug_mode (0=off,1=irr,2=vis)
}

// Get world-space position of a probe given its 3D grid index
fn ddgiProbePosition(gridIdx: vec3i, ddgi: DDGIUniforms) -> vec3f {
    return ddgi.grid_min.xyz + vec3f(gridIdx) * ddgi.grid_spacing.xyz;
}

// Get the texel coordinate in the irradiance atlas for a probe index and octahedral UV
fn ddgiIrradianceTexelCoord(probeIdx: i32, octUV: vec2f, ddgi: DDGIUniforms) -> vec2f {
    let texelDim = f32(ddgi.irradiance_texel_size.x);      // 8.0
    let texelDimBorder = f32(ddgi.irradiance_texel_size.y); // 10.0
    let atlasWidth = ddgi.irradiance_texel_size.z;
    let atlasHeight = ddgi.irradiance_texel_size.w;

    let probesPerRow = i32(ddgi.grid_count.x);
    let probeRow = probeIdx / probesPerRow;
    let probeCol = probeIdx % probesPerRow;

    // Corner including the 1px border. We add +1.0 because the interior starts at offset 1.0.
    // The octUV * texelDim maps the [0, 1] UV to [0.0, 8.0]. 
    // At exactly 0.0 or 8.0, hardware bilinear filtering will smoothly blend 50% from the border pixel!
    let texelX = f32(probeCol) * texelDimBorder + 1.0 + octUV.x * texelDim;
    let texelY = f32(probeRow) * texelDimBorder + 1.0 + octUV.y * texelDim;

    return vec2f(texelX / atlasWidth, texelY / atlasHeight);
}

// Get the texel coordinate in the visibility atlas for a probe index and octahedral UV
fn ddgiVisibilityTexelCoord(probeIdx: i32, octUV: vec2f, ddgi: DDGIUniforms) -> vec2f {
    let texelDim = f32(ddgi.visibility_texel_size.x);       // 16.0
    let texelDimBorder = f32(ddgi.visibility_texel_size.y);  // 18.0
    let atlasWidth = ddgi.visibility_texel_size.z;
    let atlasHeight = ddgi.visibility_texel_size.w;

    let probesPerRow = i32(ddgi.grid_count.x);
    let probeRow = probeIdx / probesPerRow;
    let probeCol = probeIdx % probesPerRow;

    let texelX = f32(probeCol) * texelDimBorder + 1.0 + octUV.x * texelDim;
    let texelY = f32(probeRow) * texelDimBorder + 1.0 + octUV.y * texelDim;

    return vec2f(texelX / atlasWidth, texelY / atlasHeight);
}

// Flatten 3D grid index to linear probe index
fn ddgiProbeLinearIndex(gridIdx: vec3i, ddgi: DDGIUniforms) -> i32 {
    return gridIdx.z * ddgi.grid_count.x * ddgi.grid_count.y
         + gridIdx.y * ddgi.grid_count.x
         + gridIdx.x;
}

struct RCUniforms {
    grid_count: vec4i,
    grid_min: vec4f,
    grid_max: vec4f,
    grid_spacing: vec4f,
    atlas_dims: vec4f,
    params: vec4f, // x = hysteresis, y = intensity, z = ambient, w = enabled
    debug: vec4f,  // x = debug_mode (0=Off, 1=GI Only, 2=Atlas PIP)
}

// ============================
// NRC (Neural Radiance Caching)
// ============================
struct NRCUniforms {
    scene_min: vec4f,          // xyz = scene AABB min, w = enabled (0/1)
    scene_max: vec4f,          // xyz = scene AABB max, w = debug_mode
    params: vec4f,             // x = learning_rate, y = num_training_samples, z = momentum, w = frame_count
    screen_dims: vec4f,        // x = width, y = height, z = sample_stride_x, w = sample_stride_y
}

// Octahedral encoding: map direction to [0,1]^2
fn octEncode(n: vec3f) -> vec2f {
    let sum = abs(n.x) + abs(n.y) + abs(n.z);
    var oct = vec2f(n.x, n.y) / sum;
    if (n.z < 0.0) {
        let signs = vec2f(
            select(-1.0, 1.0, oct.x >= 0.0),
            select(-1.0, 1.0, oct.y >= 0.0)
        );
        oct = (1.0 - abs(vec2f(oct.y, oct.x))) * signs;
    }
    return oct * 0.5 + 0.5;
}

// Octahedral decoding: map [0,1]^2 to direction
fn octDecode(uv: vec2f) -> vec3f {
    var f = uv * 2.0 - 1.0;
    var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        let signs = vec2f(
            select(-1.0, 1.0, n.x >= 0.0),
            select(-1.0, 1.0, n.y >= 0.0)
        );
        let xy = (1.0 - abs(vec2f(n.y, n.x))) * signs;
        n = vec3f(xy.x, xy.y, n.z);
    }
    return normalize(n);
}

// pt_common.wgsl
// Shared structs, PCG RNG, and utilities for Wavefront Path Tracing.
// Included AFTER common.wgsl and bvh.wgsl by all PT compute shaders.
// NOTE: bvh.wgsl already defines `struct Ray { origin, direction }`.

// ============================================================
// Path Tracing Structs
// ============================================================

// 64 bytes — one per pixel, persists across bounce passes
struct PTRay {
    origin:          vec3f,   // 12
    ior:             f32,     //  4  current medium IOR (1.0=air)
    direction:       vec3f,   // 12
    pixel_id:        u32,     //  4  flat pixel index (y * width + x)
    throughput:      vec3f,   // 12  spectral weight (starts at 1,1,1)
    bounce:          u32,     //  4  current bounce depth
    ray_active:      u32,     //  4  0=inactive, 1=active  (named ray_active to avoid WGSL reserved 'active')
    specular_bounce: u32,     //  4  1 = last event was purely specular
    _pad:            vec2u,   //  8
}; // 64 bytes

// 48 bytes — compact hit; shade reconstructs pos/normal/tangent/uv from BVH
struct HitRecord {
    dist:    f32,    //  4  ray t-parameter
    side:    f32,    //  4  +1.0 = front, -1.0 = back
    bary:    vec2f,  //  8  barycentric (x,y); z = 1-x-y
    idx0:    u32,    //  4  vertex index 0
    idx1:    u32,    //  4  vertex index 1
    idx2:    u32,    //  4  vertex index 2
    mat_id:  u32,    //  4  material id
    did_hit: u32,    //  4  0=miss, 1=hit
    _pad0:   u32,    //  4
    _pad1:   u32,    //  4
    _pad2:   u32,    //  4
}; // 48 bytes

// 48 bytes — NEE shadow ray, one per active pixel
struct ShadowRay {
    origin:       vec3f,  // 12
    max_dist:     f32,    //  4
    direction:    vec3f,  // 12
    pixel_id:     u32,    //  4
    Li:           vec3f,  // 12
    shadow_active: u32,   //  4  (named shadow_active to avoid WGSL reserved 'active')
}; // 48 bytes

// Unpacked material (from global materials buffer, 16 floats = 4 × vec4f per entry)
struct PTMaterial {
    albedo:           vec3f,
    alpha:            f32,
    roughness:        f32,
    metallic:         f32,
    tex_layer:        i32,     // index into baseColor texture array (-1 = none)
    transmission:     f32,     // 0=opaque, 1=fully transmissive
    ior:              f32,
    emissive:         vec3f,
    normal_tex_layer: i32,     // index into normal map texture array (-1 = none)
    mr_tex_layer:     i32,     // index into metallic-roughness texture array (-1 = none)
};

// ============================================================
// NRC Training Struct
// ============================================================
struct NRCWavefrontTrainData {
    features: array<f32, 15>,
    _pad0: f32, // Padding to 64 bytes
    throughput: vec3f,
    _pad1: f32, // Padding to 80 bytes
    primary_radiance: vec3f,
    pixel_id: u32,
    is_active: u32,
    _pad2: vec3u, // Padding to 112 bytes? Wait, let's just make it cleanly aligned
};

// ============================================================
// PT Uniforms (per-frame constants)
// ============================================================
struct PTUniforms {
    width:          u32,
    height:         u32,
    frame_index:    u32,
    sample_count:   u32,
    max_bounces:    u32,
    clamp_radiance: f32,
    pixel_scale:    f32,
    enable_restir:  u32,
};

// ============================================================
// PCG Hash — fast pseudorandom
// ============================================================
fn pcgHash(state: u32) -> u32 {
    let s = state * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn initRNG(pixel_id: u32, frame_index: u32) -> u32 {
    return pcgHash(pixel_id ^ pcgHash(frame_index + 1u));
}

fn randU32(state: ptr<function, u32>) -> u32 {
    *state = pcgHash(*state);
    return *state;
}

fn rand(state: ptr<function, u32>) -> f32 {
    return f32(randU32(state)) / 4294967296.0;
}

fn rand2(state: ptr<function, u32>) -> vec2f {
    return vec2f(rand(state), rand(state));
}

// ============================================================
// Sampling helpers
// ============================================================

fn sampleCosineHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
    let r1 = rand(rng);
    let r2 = rand(rng);
    let cosTheta = sqrt(r1);
    let sinTheta = sqrt(max(0.0, 1.0 - r1));
    let phi = 2.0 * PI * r2;
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.x) > 0.9);
    let tangent   = normalize(cross(n, up));
    let bitangent = cross(n, tangent);
    return normalize(tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + n * cosTheta);
}

fn sampleGGX(n: vec3f, roughness: f32, rng: ptr<function, u32>) -> vec3f {
    let r1  = rand(rng);
    let r2  = rand(rng);
    let a   = roughness * roughness;
    let theta = atan(a * sqrt(r1) / sqrt(max(1.0 - r1, 1e-6)));
    let phi   = 2.0 * PI * r2;
    let lx = sin(theta) * cos(phi);
    let ly = sin(theta) * sin(phi);
    let lz = cos(theta);
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.x) > 0.9);
    let tangent   = normalize(cross(n, up));
    let bitangent = cross(n, tangent);
    return normalize(tangent * lx + bitangent * ly + n * lz);
}

// ============================================================
// Material unpack (4 × vec4f = 16 floats per material entry)
fn unpackPTMaterial(
    mats: ptr<storage, array<vec4f>, read>,
    mat_id: u32
) -> PTMaterial {
    let base = mat_id * 4u;
    let r0 = (*mats)[base + 0u];
    let r1 = (*mats)[base + 1u];
    let r2 = (*mats)[base + 2u];
    let r3 = (*mats)[base + 3u];
    var m: PTMaterial;
    m.albedo           = r0.xyz;
    m.alpha            = r0.w;
    m.roughness        = r1.x;
    m.metallic         = r1.y;
    m.tex_layer        = bitcast<i32>(r1.z);
    m.transmission     = r1.w;
    m.ior              = r2.x;
    m.emissive         = r2.yzw;
    m.normal_tex_layer = bitcast<i32>(r3.x);
    m.mr_tex_layer     = bitcast<i32>(r3.y);
    return m;
}

// ============================================================
// Fresnel & GGX helpers (suffixed _pt to avoid collision)
// ============================================================
fn fresnelSchlickV(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnelDielectric(cosI: f32, etaI: f32, etaT: f32) -> f32 {
    let sinT2 = (etaI / etaT) * (etaI / etaT) * (1.0 - cosI * cosI);
    if (sinT2 >= 1.0) { return 1.0; }
    let cosT = sqrt(max(0.0, 1.0 - sinT2));
    let rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    let rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return 0.5 * (rs * rs + rp * rp);
}

fn distributionGGX_pt(NdotH: f32, a2: f32) -> f32 {
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX_pt(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// ray_gen.cs.wgsl
// Wavefront Path Tracing — Pass 1: Primary Ray Generation

@group(0) @binding(0) var<uniform>            camera:     CameraUniforms;
@group(0) @binding(1) var<uniform>            pt:         PTUniforms;
@group(0) @binding(2) var<storage, read_write> ray_buffer: array<PTRay>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    if (gid.x >= render_width || gid.y >= render_height) { return; }

    let pixel_id = gid.y * render_width + gid.x;

    var rng = initRNG(pixel_id, pt.frame_index);
    let jitter = rand2(&rng) - vec2f(0.5);

    let uv  = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5) + jitter)
            / vec2f(f32(render_width), f32(render_height));
    let ndc = vec2f(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    let clip_near = vec4f(ndc.x, ndc.y, 1.0, 1.0);
    let clip_far  = vec4f(ndc.x, ndc.y, 0.0, 1.0);
    var world_near = camera.inv_view_proj_mat * clip_near;
    var world_far  = camera.inv_view_proj_mat * clip_far;
    world_near /= world_near.w;
    world_far  /= world_far.w;

    let origin    = camera.camera_pos.xyz;
    let direction = normalize(world_far.xyz - world_near.xyz);

    var ray: PTRay;
    ray.origin          = origin;
    ray.ior             = 1.0;
    ray.direction       = direction;
    ray.pixel_id        = pixel_id;
    ray.throughput      = vec3f(1.0);
    ray.bounce          = 0u;
    ray.ray_active      = 1u;
    ray.specular_bounce = 0u;
    ray._pad            = vec2u(0u);

    ray_buffer[pixel_id] = ray;
}
