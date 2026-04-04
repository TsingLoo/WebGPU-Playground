@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read> globalLightIndices: LightIndexListReadOnly;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;

@group(${bindGroup_scene}) @binding(5) var albedoTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(6) var normalTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(7) var positionTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(8) var specularTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(9) var depthTex: texture_depth_2d;
@group(${bindGroup_scene}) @binding(10) var outputTex: texture_storage_2d<rgba16float, write>;
@group(${bindGroup_scene}) @binding(11) var irradianceMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(12) var prefilteredMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(13) var brdfLut: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(14) var iblSampler: sampler;
@group(1) @binding(0) var ddgiIrradianceAtlas: texture_2d<f32>;
@group(1) @binding(1) var ddgiVisibilityAtlas: texture_2d<f32>;
@group(1) @binding(2) var<uniform> ddgiParams: DDGIUniforms;
@group(1) @binding(3) var ddgiSampler: sampler;

@group(1) @binding(4) var rcIrradianceAtlas: texture_2d<f32>;
@group(1) @binding(5) var<uniform> rcParams: RCUniforms;
@group(1) @binding(6) var rcSampler: sampler;
@group(1) @binding(7) var<storage, read> ddgiProbeData: array<vec4f>;
@group(${bindGroup_scene}) @binding(15) var<uniform> sunLight: SunLight;
@group(${bindGroup_scene}) @binding(16) var vsmPhysAtlas: texture_depth_2d;
@group(${bindGroup_scene}) @binding(17) var<storage, read> vsmPageTable: array<u32>;
@group(${bindGroup_scene}) @binding(18) var<uniform> vsmUniforms: VSMUniforms;
// NRC bindings
@group(${bindGroup_scene}) @binding(19) var nrcInferenceTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(20) var<uniform> nrcParams: NRCUniforms;

// Surfel GI binding
@group(${bindGroup_scene}) @binding(21) var surfelIrradianceTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(22) var<uniform> surfelParams: vec4f; // .x = enabled, .y = intensity

// SSAO binding
@group(${bindGroup_scene}) @binding(23) var ssaoTex: texture_2d<f32>;

// Shared memory for tile light indices — pre-loaded by thread 0 of each workgroup
const MAX_SHARED_LIGHTS: u32 = ${maxLightsPerCluster}u;
var<workgroup> sharedLightCount: u32;
var<workgroup> sharedLightIndices: array<u32, ${maxLightsPerCluster}>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32
) {
    let fragcoordi = vec2i(global_id.xy);
    let screen_pos_x = f32(global_id.x);
    let screen_pos_y = f32(global_id.y);

    let screen_dims = textureDimensions(albedoTex);
    let in_bounds = global_id.x < screen_dims.x && global_id.y < screen_dims.y;

    // ---- Shared memory light list pre-loading ----
    // Must happen before any early returns so ALL threads reach the barrier.
    // Use center pixel of this workgroup tile for the cluster lookup.
    let tile_center_x = f32(global_id.x - (global_id.x % 8u) + 4u);
    let tile_center_y = f32(global_id.y - (global_id.y % 8u) + 4u);
    // For the cluster Z, we need a world position — thread 0 reads the tile center pixel
    if (local_idx == 0u) {
        let center_coord = vec2i(i32(tile_center_x), i32(tile_center_y));
        let center_pos = textureLoad(positionTex, center_coord, 0).xyz;
        let cluster_index = getClusterIndex(vec2f(tile_center_x, tile_center_y), center_pos, camera, clusterSet);
        let lightmeta = tileOffsets[cluster_index];
        sharedLightCount = min(lightmeta.count, MAX_SHARED_LIGHTS);
        for (var i = 0u; i < sharedLightCount; i++) {
            sharedLightIndices[i] = globalLightIndices.indices[lightmeta.offset + i];
        }
    }
    workgroupBarrier();

    // ---- Early exit for out-of-bounds or transparent pixels ----
    if (!in_bounds) { return; }

    let diffuseColor = textureLoad(albedoTex, fragcoordi, 0);
    if (diffuseColor.a < 0.5f) {
        textureStore(outputTex, fragcoordi, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let albedo = diffuseColor.rgb;
    let pos_world = textureLoad(positionTex, fragcoordi, 0).xyz;
    let nor_world = textureLoad(normalTex, fragcoordi, 0).xyz;
    let specularData = textureLoad(specularTex, fragcoordi, 0);
    
    let roughness = max(specularData.r, 0.04);
    let metallic = specularData.g;
    
    var ao = specularData.b; // ambient occlusion from G-buffer
    let shadingModelId = specularData.a;
    let ssao_val = textureLoad(ssaoTex, fragcoordi, 0).r;
    ao = ao * ssao_val;

    let N = normalize(nor_world);
    let V = normalize(camera.camera_pos.xyz - pos_world);

    // ---- Shading Model Dispatch ----
    if (shadingModelId > 0.5) {
        // SHADING_MODEL_UNLIT: output albedo directly (no lighting)
        let mapped = albedo / (albedo + vec3f(1.0));
        let corrected = pow(mapped, vec3f(1.0 / 2.2));
        textureStore(outputTex, fragcoordi, vec4f(corrected, 1.0));
        return;
    }

    // SHADING_MODEL_PBR: standard Cook-Torrance lighting
    // ---- Direct lighting using shared light list ----
    var Lo = vec3f(0.0);
    let count = sharedLightCount;
    for (var i = 0u; i < count; i += 1u) {
        let light_idx = sharedLightIndices[i];
        let light = lightSet.lights[light_idx];
        Lo += calculateLightContribPBR(light, pos_world, N, V, albedo, metallic, roughness);
    }

    // Sun/directional light with VSM shadow
    let shadow = calculateShadowVSM(vsmPhysAtlas, vsmUniforms, sunLight, pos_world, N);
    Lo += calculateSunLightPBR(sunLight, pos_world, N, V, albedo, metallic, roughness, shadow);

    // ---- IBL Ambient (shared split-sum) ----
    let ibl = computeIBL(N, V, albedo, metallic, roughness);

    // Build ambient: scale IBL independently from RC
    var diffuseAmbient = vec3f(0.0);
    if (ddgiParams.ddgi_enabled.x > 0.5) {
        let ddgi_totalIrr = evaluateDDGI(pos_world, N, V, ddgiParams, ddgiIrradianceAtlas, ddgiVisibilityAtlas, ddgiSampler);
        diffuseAmbient = ddgi_totalIrr * albedo;
    } else if (rcParams.params.w > 0.5) {
        // Evaluate Radiance Cascades
        let rc_totalIrr = evaluateRCProbes(pos_world, N, V, rcParams, rcIrradianceAtlas, rcSampler);

        let rcDebugMode = i32(rcParams.debug.x);
        if (rcDebugMode == 1) {
            textureStore(outputTex, fragcoordi, vec4f(rc_totalIrr * rcParams.params.y, 1.0));
            return;
        } else if (rcDebugMode == 2) {
            let scr_w = f32(clusterSet.screen_width);
            let scr_h = f32(clusterSet.screen_height);
            let uv = vec2f(f32(global_id.x), f32(global_id.y)) / vec2f(scr_w, scr_h);
            if (uv.x > 0.6 && uv.y < 0.4) {
                let atlasUV = vec2f((uv.x - 0.6) / 0.4, uv.y / 0.4);
                let atlasColor = textureSampleLevel(rcIrradianceAtlas, rcSampler, atlasUV, 0.0).rgb;
                textureStore(outputTex, fragcoordi, vec4f(atlasColor * 2.0, 1.0));
                return;
            }
        }

        diffuseAmbient = max(rc_totalIrr * albedo, ibl.iblIrradiance * albedo * 0.25);
    } else if (nrcParams.scene_min.w > 0.5) {
        // NRC mode
        let screenUV = vec2f(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5) / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
        let nrcIrradiance = evaluateNRC(screenUV, nrcInferenceTex);
        let nrcBounce = nrcIrradiance * albedo;
        let iblFloor2 = ibl.iblIrradiance * albedo * 0.15;
        diffuseAmbient = max(nrcBounce, iblFloor2);
    } else if (surfelParams.x > 0.5) {
        // Surfel GI mode
        let screenUV = vec2f(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5) / vec2f(f32(clusterSet.screen_width), f32(clusterSet.screen_height));
        let surfelIrradiance = evaluateSurfel(screenUV, surfelIrradianceTex);
        let surfelBounce = surfelIrradiance * albedo * surfelParams.y;
        let iblFloor3 = ibl.iblIrradiance * albedo * 0.05;
        diffuseAmbient = max(surfelBounce, iblFloor3);
    } else {
        // No DDGI/NRC/Surfel: use IBL irradiance with moderate scaling
        diffuseAmbient = ibl.iblIrradiance * albedo * 1.0;
    }

    // Composite and tone map (shared)
    let isGIActive = ddgiParams.ddgi_enabled.x > 0.5 || rcParams.params.w > 0.5;
    let corrected = compositeAndTonemap(Lo, ibl.kD, diffuseAmbient, ibl.specularIBL, ao, isGIActive);

    textureStore(outputTex, fragcoordi, vec4f(corrected, 1.0));
}