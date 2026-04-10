// ==========================================
// Shared Lighting Composite Functions
// ==========================================
// Used by both Forward+ (fragment) and Clustered Deferred (compute) shaders.
// Requires global bindings: irradianceMap, prefilteredMap, brdfLut, iblSampler
// (both shaders must define these with identical names).

struct IBLResult {
    kD: vec3f,
    iblIrradiance: vec3f,
    specularIBL: vec3f,
}

// Compute Image-Based Lighting using split-sum approximation.
// Returns energy-conservation terms and pre-integrated IBL contributions.
fn computeIBL(N: vec3f, V: vec3f, albedo: vec3f, metallic: f32, roughness: f32) -> IBLResult {
    var result: IBLResult;

    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    result.kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL from preconvolved irradiance map
    result.iblIrradiance = textureSampleLevel(irradianceMap, iblSampler, N, 0.0).rgb;

    // Specular IBL (split-sum)
    let R = reflect(-V, N);
    let maxLod = 4.0; // PREFILTER_MIP_LEVELS - 1
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdfVal = textureSampleLevel(brdfLut, iblSampler, vec2f(NdotV, roughness), 0.0).rg;
    result.specularIBL = prefilteredColor * (F * brdfVal.x + brdfVal.y);

    return result;
}

// Compose final color from direct lighting (Lo), GI diffuse ambient, IBL specular, and AO.
// Applies Reinhard tone mapping and gamma correction.
fn compositeAndTonemap(
    Lo: vec3f,
    kD: vec3f,
    diffuseAmbient: vec3f,
    specularIBL: vec3f,
    ao: f32,
    emissive: vec3f,
    isGIActive: bool
) -> vec3f {
    // Scale down specular IBL when GI is active to avoid double-counting
    let specIBLScale = select(0.6, 0.0, isGIActive);
    let ambient = (kD * diffuseAmbient + specularIBL * specIBLScale) * ao;
    let finalColor = ambient + Lo + emissive;

    // Reinhard tone mapping
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    return pow(mapped, vec3f(1.0 / 2.2));
}
