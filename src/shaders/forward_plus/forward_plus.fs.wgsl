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

@group(3) @binding(4) var rcIrradianceAtlas: texture_2d<f32>;
@group(3) @binding(5) var<uniform> rcParams: RCUniforms;
@group(3) @binding(6) var rcSampler: sampler;
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
    let surf = evaluateMaterial(in.uv, in.nor_world, in.tangent_world);
    if (surf.alpha < 0.5f) {
        discard;
    }

    let albedo = surf.albedo;
    var roughness = surf.roughness;
    var metallic = surf.metallic;
    
    let ssao_val = textureLoad(ssaoTex, vec2i(in.fragcoord.xy), 0).r;
    var ao = ssao_val;

    var N = surf.N;
    let vertexNormal = normalize(in.nor_world); // save for debug

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

    // Build ambient: scale IBL independently from DDGI/RC
    var diffuseAmbient = vec3f(0.0);
    if (ddgiParams.ddgi_enabled.x > 0.5) {
        let ddgi_totalIrr = evaluateDDGI(in.pos_world, N, V, ddgiParams, ddgiIrradianceAtlas, ddgiVisibilityAtlas, ddgiSampler);
        diffuseAmbient = ddgi_totalIrr * albedo;
    } else if (rcParams.params.w > 0.5) {
        let rc_totalIrr = evaluateRCProbes(in.pos_world, N, V, rcParams, rcIrradianceAtlas, rcSampler);
        diffuseAmbient = rc_totalIrr * albedo;
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

    // DDGI debug visualization removed since Radiance Cascades do not require it

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
    // Distant reflection usually doesn't match interior lighting, scale it down if GI is active
    let isGIActive = ddgiParams.ddgi_enabled.x > 0.5 || rcParams.params.w > 0.5;
    let specIBLScale = select(0.6, 0.0, isGIActive);
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
