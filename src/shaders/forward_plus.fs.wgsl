@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read> globalLightIndices: LightIndexListReadOnly;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;
@group(${bindGroup_scene}) @binding(5) var irradianceMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(6) var prefilteredMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(7) var brdfLut: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(8) var iblSampler: sampler;
@group(3) @binding(0) var ddgiIrradianceAtlas: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibilityAtlas: texture_2d<f32>;
@group(3) @binding(2) var<uniform> ddgiParams: DDGIUniforms;
@group(3) @binding(3) var ddgiSampler: sampler;
@group(${bindGroup_scene}) @binding(9) var<uniform> sunLight: SunLight;
// VSM bindings
@group(${bindGroup_scene}) @binding(10) var vsmPhysAtlas: texture_depth_2d;
@group(${bindGroup_scene}) @binding(11) var vsmShadowSampler: sampler_comparison;
@group(${bindGroup_scene}) @binding(12) var<storage, read> vsmPageTable: array<u32>;
@group(${bindGroup_scene}) @binding(13) var<uniform> vsmUniforms: VSMUniforms;
// NRC bindings
@group(${bindGroup_scene}) @binding(14) var nrcInferenceTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(15) var<uniform> nrcParams: NRCUniforms;
@group(${bindGroup_scene}) @binding(16) var gBufferPosition: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(17) var gBufferNormal: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(18) var gBufferAlbedo: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(19) var surfelIrradianceTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(20) var<uniform> surfelParams: vec4f;
@group(${bindGroup_scene}) @binding(21) var ssaoTex: texture_2d<f32>;

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    has_normal_texture: f32,
    base_color_factor: vec4f,
    _reserved: vec4f,
}
@group(${bindGroup_material}) @binding(2) var<uniform> pbrParams: PBRParams;
@group(${bindGroup_material}) @binding(3) var metallicRoughnessTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(4) var metallicRoughnessTexSampler: sampler;
@group(${bindGroup_material}) @binding(5) var normalTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(6) var normalTexSampler: sampler;

struct FragmentInput
{
    @builtin(position) fragcoord: vec4f,
    @location(0) pos_world: vec3f,
    @location(1) nor_world: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent_world: vec4f
}



@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv) * pbrParams.base_color_factor;
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    let albedo = diffuseColor.rgb;

    // Per-pixel metallic/roughness/AO from texture (glTF ORM packing: R = occlusion, G = roughness, B = metallic)
    var metallic = pbrParams.metallic;
    var roughness = pbrParams.roughness;
    var ao = 1.0;
    if (pbrParams.has_mr_texture > 0.5) {
        let mrSample = textureSample(metallicRoughnessTex, metallicRoughnessTexSampler, in.uv);
        // glTF spec: metallicRoughness texture has G=roughness, B=metallic
        // Occlusion is a separate texture (not bound here), so keep ao = 1.0
        roughness = roughness * mrSample.g; // scalar * texture per glTF spec
        metallic = metallic * mrSample.b;
    }
    roughness = max(roughness, 0.04); // clamp to avoid singularity
    
    let ssao_val = textureLoad(ssaoTex, vec2i(in.fragcoord.xy), 0).r;
    ao = ao * ssao_val;

    // Normal mapping: build TBN matrix and sample normal map
    var N = normalize(in.nor_world);
    let vertexNormal = N; // save for debug
    if (pbrParams.has_normal_texture > 0.5) {
        // Re-orthogonalize tangent against normal (Gram-Schmidt)
        let T_raw = in.tangent_world.xyz - N * dot(in.tangent_world.xyz, N);
        let T_len = length(T_raw);
        var T = vec3f(0.0);
        if (T_len > 0.001) {
            T = T_raw / T_len;
        } else {
            // Degenerate tangent - pick one orthogonal to N
            let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
            T = normalize(cross(N, refVec));
        }
        // Handedness: default to 1.0 if tangent.w is zero (missing data)
        let handedness = select(in.tangent_world.w, 1.0, abs(in.tangent_world.w) < 0.5);
        let B = normalize(cross(N, T)) * handedness;
        let tbn = mat3x3f(T, B, N);
        // Sample normal map (stored as [0,1], convert to [-1,1])
        let normalSample = textureSample(normalTex, normalTexSampler, in.uv).rgb;
        let tangentNormal = normalSample * 2.0 - 1.0;
        N = normalize(tbn * tangentNormal);
    }

    let V = normalize(camera.camera_pos.xyz - in.pos_world);

    // ---- Cluster lookup ----
    let screen_width = f32(clusterSet.screen_width);
    let screen_height = f32(clusterSet.screen_height);

    let num_clusters_X = clusterSet.num_clusters_X;
    let num_clusters_Y = clusterSet.num_clusters_Y;
    let num_clusters_Z = clusterSet.num_clusters_Z;

    let screen_size_cluster_x = screen_width / f32(num_clusters_X);
    let screen_size_cluster_y = screen_height / f32(num_clusters_Y);

    let clusterid_x = u32(in.fragcoord.x / screen_size_cluster_x);
    let clusterid_y_unflipped = u32(in.fragcoord.y / screen_size_cluster_y);
    let clusterid_y = clamp((num_clusters_Y - 1u) - clusterid_y_unflipped, 0u, num_clusters_Y - 1u);

    let pos_view = (camera.view_mat * vec4f(in.pos_world, 1.0)).xyz;
    let z_view = pos_view.z;

    let near = camera.near_plane; 
    let far = camera.far_plane;
    let clamped_Z_positive = clamp(-z_view, near, far);

    let logFN = log(far/near);
    let SCALE = f32(num_clusters_Z) / logFN;
    let BIAS = SCALE * log(near);
    let slice = log(clamped_Z_positive) * SCALE - BIAS;
    let cluster_z = clamp(u32(floor(slice)), 0u, num_clusters_Z - 1u);

    let cluster_index = cluster_z * (num_clusters_X * num_clusters_Y) +
                          clusterid_y * num_clusters_X +
                          clusterid_x;

    let lightmeta = tileOffsets[cluster_index];
    let offset = lightmeta.offset;
    let count = lightmeta.count;

    // ---- Direct lighting (PBR Cook-Torrance) ----
    var Lo = vec3f(0.0);
    for (var i = 0u; i < count; i += 1u) {
        let light_idx = globalLightIndices.indices[offset + i];
        let light = lightSet.lights[light_idx];
        Lo += calculateLightContribPBR(light, in.pos_world, N, V, albedo, metallic, roughness);
    }

    // Sun/directional light with VSM shadow
    let shadow = calculateShadowVSM(vsmPhysAtlas, vsmUniforms, sunLight, in.pos_world, N);
    Lo += calculateSunLightPBR(sunLight, in.pos_world, N, V, albedo, metallic, roughness, shadow);

    // ---- IBL Ambient (split-sum approximation) ----
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL from preconvolved irradiance map
    let iblIrradiance = textureSampleLevel(irradianceMap, iblSampler, N, 0.0).rgb;

    // Specular IBL (split-sum)
    let R = reflect(-V, N);
    let maxLod = 4.0; // PREFILTER_MIP_LEVELS - 1
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdf = textureSampleLevel(brdfLut, iblSampler, vec2f(NdotV, roughness), 0.0).rg;
    let specularIBL = prefilteredColor * (F * brdf.x + brdf.y);

    // Build ambient: scale IBL independently from DDGI
    var diffuseAmbient = vec3f(0.0);
    if (ddgiParams.ddgi_enabled.x > 0.5) {
        // Inline DDGI irradiance sampling (trilinear probe interpolation + Chebyshev visibility)
        let ddgi_totalIrr = evaluateDDGI(in.pos_world, N, V, ddgiParams, ddgiIrradianceAtlas, ddgiVisibilityAtlas, ddgiSampler);

        let scr_w = f32(clusterSet.screen_width);
        let scr_h = f32(clusterSet.screen_height);
        diffuseAmbient = evaluateHybridSSGI(
            in.fragcoord.xy, in.pos_world, N, albedo, ddgi_totalIrr, iblIrradiance,
            camera, scr_w, scr_h, gBufferPosition, gBufferAlbedo, ddgiParams.ddgi_enabled.w
        );
    } else if (nrcParams.scene_min.w > 0.5) {
        // NRC mode: sample the neural radiance cache inference texture
        let screenUV = in.fragcoord.xy / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
        let nrcIrradiance = evaluateNRC(screenUV, nrcInferenceTex);
        // NRC provides cached irradiance; apply albedo modulation
        let nrcBounce = nrcIrradiance * albedo;
        let iblFloor2 = iblIrradiance * albedo * 0.15;
        diffuseAmbient = max(nrcBounce, iblFloor2);
    } else if (surfelParams.x > 0.5) {
        // Surfel GI mode
        let screenUV = in.fragcoord.xy / vec2f(f32(clusterSet.screen_width), f32(clusterSet.screen_height));
        let surfelIrradiance = evaluateSurfel(screenUV, surfelIrradianceTex);
        
        if (surfelParams.z > 0.5) {
            // DEBUG MODE: Return irradiance directly
            if (length(surfelIrradiance) > 0.001) {
                return vec4f(surfelIrradiance * 1.5, 1.0);
            } else {
                return vec4f(0.0, 0.0, 0.0, 1.0);
            }
        }
        
        let surfelBounce = surfelIrradiance * albedo * surfelParams.y; // apply intensity
        diffuseAmbient = surfelBounce; // Completely replace IBL ambient to clearly see the real GI
    } else {
        // No DDGI/NRC/Surfel: use IBL irradiance with moderate scaling
        diffuseAmbient = iblIrradiance * albedo * 1.0;
    }

    // ---- DDGI Debug Visualization ----
    let debugMode = i32(ddgiParams.ddgi_enabled.y);
    if (debugMode == 1) {
        // Mode 1: Raw DDGI irradiance (should NOT be black if probes have data)
        if (ddgiParams.ddgi_enabled.x > 0.5) {
            // Re-sample center probe for this fragment to show raw irradiance
            let dbg_spacing = ddgiParams.grid_spacing.xyz;
            let dbg_gridMin = ddgiParams.grid_min.xyz;
            let dbg_fractIdx = (in.pos_world - dbg_gridMin) / dbg_spacing;
            let dbg_baseIdx = clamp(vec3i(floor(dbg_fractIdx)), vec3i(0), ddgiParams.grid_count.xyz - vec3i(1));
            let dbg_probeIdx = ddgiProbeLinearIndex(dbg_baseIdx, ddgiParams);
            let dbg_irrUV = ddgiIrradianceTexelCoord(dbg_probeIdx, octEncode(N), ddgiParams);
            let dbg_raw = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, dbg_irrUV, 0.0).rgb;
            // Show raw atlas value (gamma-encoded) amplified
            return vec4f(dbg_raw * 3.0, 1.0);
        }
        return vec4f(1.0, 0.0, 1.0, 1.0); // Magenta = DDGI disabled
    }
    if (debugMode == 2) {
        // Mode 2: Decoded DDGI irradiance (trilinear sampled, after pow 5)
        if (ddgiParams.ddgi_enabled.x > 0.5) {
            let dbg2_spacing = ddgiParams.grid_spacing.xyz;
            let dbg2_gridMin = ddgiParams.grid_min.xyz;
            let dbg2_fractIdx = (in.pos_world - dbg2_gridMin) / dbg2_spacing;
            let dbg2_baseIdx = clamp(vec3i(floor(dbg2_fractIdx)), vec3i(0), ddgiParams.grid_count.xyz - vec3i(1));
            let dbg2_probeIdx = ddgiProbeLinearIndex(dbg2_baseIdx, ddgiParams);
            let dbg2_irrUV = ddgiIrradianceTexelCoord(dbg2_probeIdx, octEncode(N), ddgiParams);
            let dbg2_encoded = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, dbg2_irrUV, 0.0).rgb;
            let dbg2_decoded = pow(max(dbg2_encoded, vec3f(0.0)), vec3f(5.0));
            let dbg2_mapped = dbg2_decoded / (dbg2_decoded + vec3f(1.0));
            return vec4f(pow(dbg2_mapped, vec3f(1.0/2.2)), 1.0);
        }
        return vec4f(0.0, 1.0, 1.0, 1.0); // Cyan = no DDGI data
    }
    if (debugMode == 3) {
        // Mode 3: IBL irradiance only
        let dbg_ibl = iblIrradiance;
        let dbg_mapped2 = dbg_ibl / (dbg_ibl + vec3f(1.0));
        return vec4f(pow(dbg_mapped2, vec3f(1.0/2.2)), 1.0);
    }
    if (debugMode == 4) {
        // Mode 4: Final mapped world-space normal as RGB
        return vec4f(N * 0.5 + 0.5, 1.0);
    }
    if (debugMode == 5) {
        // Mode 5: Vertex normal (before normal mapping) as RGB
        return vec4f(vertexNormal * 0.5 + 0.5, 1.0);
    }
    if (debugMode == 6) {
        // Mode 6: Tangent vector as RGB
        return vec4f(normalize(in.tangent_world.xyz) * 0.5 + 0.5, 1.0);
    }
    if (debugMode == 7) {
        // Mode 7: NdotL (sun) - bright = facing sun, dark = away
        let sunDir = normalize(sunLight.direction.xyz);
        let ndotl = max(dot(N, sunDir), 0.0);
        return vec4f(vec3f(ndotl), 1.0);
    }
    if (debugMode == 8) {
        // Mode 8: DDGI Probe Grid visualization
        // Show probe positions as colored dots overlaid on the scene
        let spacing = ddgiParams.grid_spacing.xyz;
        let gridMin = ddgiParams.grid_min.xyz;
        let relPos = (in.pos_world - gridMin) / spacing;
        let nearestProbe = round(relPos);
        let probePos = gridMin + nearestProbe * spacing;
        let distToProbe = length(in.pos_world - probePos);
        let probeRadius = min(min(spacing.x, spacing.y), spacing.z) * 0.08;
        if (distToProbe < probeRadius) {
            // Color by grid position
            let gridIdx = vec3i(nearestProbe);
            let col = vec3f(
                f32(gridIdx.x % 2),
                f32(gridIdx.y % 2),
                f32(gridIdx.z % 2)
            ) * 0.5 + 0.5;
            return vec4f(col, 1.0);
        }
        // Show faded version of normal scene + grid lines
        let gridFrac = fract(relPos);
        let gridLine = step(vec3f(0.95), gridFrac) + step(gridFrac, vec3f(0.05));
        let isGrid = max(max(gridLine.x, gridLine.y), gridLine.z);
        if (isGrid > 0.0) {
            return vec4f(0.0, 1.0, 1.0, 1.0); // Cyan grid lines
        }
        // Darken non-grid pixels slightly for contrast
        return vec4f(albedo * 0.3, 1.0);
    }

    // ---- NRC Debug Visualization ----
    if (nrcParams.scene_min.w > 0.5) {
        let nrcDebugMode = i32(nrcParams.scene_max.w);
        if (nrcDebugMode == 1) {
            // Mode 1: Raw Inference (amplified for visibility)
            let screenUV = in.fragcoord.xy / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
            let nrcTexSize = textureDimensions(nrcInferenceTex);
            let nrcCoord = vec2i(i32(screenUV.x * f32(nrcTexSize.x)), i32(screenUV.y * f32(nrcTexSize.y)));
            let nrcIrradiance = textureLoad(nrcInferenceTex, nrcCoord, 0).rgb;
            return vec4f(nrcIrradiance * 3.0, 1.0);
        }
        if (nrcDebugMode == 2) {
            // Mode 2: HDR Mapped
            let screenUV = in.fragcoord.xy / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
            let nrcTexSize = textureDimensions(nrcInferenceTex);
            let nrcCoord = vec2i(i32(screenUV.x * f32(nrcTexSize.x)), i32(screenUV.y * f32(nrcTexSize.y)));
            let nrcIrradiance = textureLoad(nrcInferenceTex, nrcCoord, 0).rgb;
            let mapped = nrcIrradiance / (nrcIrradiance + vec3f(1.0));
            return vec4f(pow(mapped, vec3f(1.0/2.2)), 1.0);
        }
    }

    // Combine: diffuse ambient + specular IBL
    // When DDGI is on, reduce specular IBL since the cubemap doesn't match interior lighting
    // DDGI only provides diffuse indirect; specular IBL from outdoor cubemap
    // creates unrealistic reflections on interior surfaces, so disable it with DDGI
    let specIBLScale = select(0.6, 0.0, ddgiParams.ddgi_enabled.x > 0.5);
    let ambient = (kD * diffuseAmbient + specularIBL * specIBLScale) * ao;

    let finalColor = ambient + Lo;

    // Detect NaN: x != x is true if x is NaN
    if (finalColor.x != finalColor.x || finalColor.y != finalColor.y || finalColor.z != finalColor.z) {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    if (ambient.x != ambient.x || ambient.y != ambient.y || ambient.z != ambient.z) {
        return vec4f(1.0, 1.0, 0.0, 1.0); // Yellow if ambient alone is NaN
    }

    // Tone mapping (Reinhard)
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    let corrected = pow(mapped, vec3f(1.0/2.2));

    return vec4f(corrected, 1.0);
}
