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
@group(${bindGroup_scene}) @binding(10) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(${bindGroup_scene}) @binding(11) var irradianceMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(12) var prefilteredMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(13) var brdfLutTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(14) var iblSampler: sampler;
@group(1) @binding(0) var ddgiIrradianceAtlas: texture_2d<f32>;
@group(1) @binding(1) var ddgiVisibilityAtlas: texture_2d<f32>;
@group(1) @binding(2) var<uniform> ddgiParams: DDGIUniforms;
@group(1) @binding(3) var ddgiSampler: sampler;
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

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let fragcoordi = vec2i(global_id.xy);
    let screen_pos_x = f32(global_id.x);
    let screen_pos_y = f32(global_id.y);

    let screen_dims = textureDimensions(albedoTex);
    if (global_id.x >= screen_dims.x || global_id.y >= screen_dims.y) {
        return;
    }

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
    let ssao_val = textureLoad(ssaoTex, fragcoordi, 0).r;
    ao = ao * ssao_val;

    let N = normalize(nor_world);
    let V = normalize(camera.camera_pos.xyz - pos_world);

    // ---- Cluster lookup ----
    let screen_width = f32(clusterSet.screen_width);
    let screen_height = f32(clusterSet.screen_height);

    let num_clusters_X = clusterSet.num_clusters_X;
    let num_clusters_Y = clusterSet.num_clusters_Y;
    let num_clusters_Z = clusterSet.num_clusters_Z;

    let screen_size_cluster_x = screen_width / f32(num_clusters_X);
    let screen_size_cluster_y = screen_height / f32(num_clusters_Y);

    let clusterid_x = u32(screen_pos_x / screen_size_cluster_x);
    let clusterid_y_unflipped = u32(screen_pos_y / screen_size_cluster_y);
    let clusterid_y = clamp((num_clusters_Y - 1u) - clusterid_y_unflipped, 0u, num_clusters_Y - 1u);

    let pos_view = (camera.view_mat * vec4f(pos_world, 1.0)).xyz;
    let z_view = pos_view.z;

    let near = camera.near_plane;
    let far = camera.far_plane;
    let clamped_Z_positive = clamp(-z_view, near, far);

    let logFN = log(far / near);
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
        Lo += calculateLightContribPBR(light, pos_world, N, V, albedo, metallic, roughness);
    }

    // Sun/directional light with VSM shadow
    let shadow = calculateShadowVSM(vsmPhysAtlas, vsmUniforms, sunLight, pos_world, N);
    Lo += calculateSunLightPBR(sunLight, pos_world, N, V, albedo, metallic, roughness, shadow);

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
    let maxLod = 4.0;
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdfVal = textureSampleLevel(brdfLutTex, iblSampler, vec2f(NdotV, roughness), 0.0).rg;
    let specularIBL = prefilteredColor * (F * brdfVal.x + brdfVal.y);

    // Build ambient: scale IBL independently from DDGI
    var diffuseAmbient = vec3f(0.0);
    if (ddgiParams.ddgi_enabled.x > 0.5) {
        // Inline DDGI irradiance sampling (trilinear probe interpolation + Chebyshev visibility)
        let ddgi_totalIrr = evaluateDDGI(pos_world, N, V, ddgiParams, ddgiIrradianceAtlas, ddgiVisibilityAtlas, ddgiSampler);

        let ddgiBounce = ddgi_totalIrr * albedo;
        let iblFill = iblIrradiance * albedo * 0.3;
        diffuseAmbient = ddgiBounce + iblFill;
    } else if (nrcParams.scene_min.w > 0.5) {
        // NRC mode: sample the neural radiance cache inference texture
        // Map global_id back to screen UV
        let screenUV = vec2f(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5) / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
        let nrcIrradiance = evaluateNRC(screenUV, nrcInferenceTex);
        // NRC provides cached irradiance; apply albedo modulation
        let nrcBounce = nrcIrradiance * albedo;
        let iblFloor2 = iblIrradiance * albedo * 0.15;
        diffuseAmbient = max(nrcBounce, iblFloor2);
    } else if (surfelParams.x > 0.5) {
        // Surfel GI mode
        let screenUV = vec2f(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5) / vec2f(screen_width, screen_height);
        let surfelIrradiance = evaluateSurfel(screenUV, surfelIrradianceTex);
        let surfelBounce = surfelIrradiance * albedo * surfelParams.y; // apply intensity
        let iblFloor3 = iblIrradiance * albedo * 0.05;
        diffuseAmbient = max(surfelBounce, iblFloor3);
    } else {
        // No DDGI/NRC/Surfel: use IBL irradiance with moderate scaling
        diffuseAmbient = iblIrradiance * albedo * 0.7;
    }

    // Combine: DDGI diffuse (unscaled) + specular IBL (scaled down when DDGI is active)
    let specIBLScale = select(0.6, 0.0, ddgiParams.ddgi_enabled.x > 0.5);
    let ambient = (kD * diffuseAmbient + specularIBL * specIBLScale) * ao;
    let finalColor = ambient + Lo;

    // Tone mapping (Reinhard)
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    let corrected = pow(mapped, vec3f(1.0/2.2));

    textureStore(outputTex, fragcoordi, vec4f(corrected, 1.0));
}