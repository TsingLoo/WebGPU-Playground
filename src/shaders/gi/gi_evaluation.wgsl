// ==========================================
// Global Illumination Evaluation Functions
// ==========================================


fn evaluateDDGI(
    pos_world: vec3f,
    geometricN: vec3f,
    V: vec3f,
    ddgi: DDGIUniforms,
    ddgiIrradianceAtlas: texture_2d<f32>,
    ddgiVisibilityAtlas: texture_2d<f32>,
    ddgiSampler: sampler
) -> vec3f {
    let spacing = ddgi.grid_spacing.xyz;
    let gridMin = ddgi.grid_min.xyz;
    let normalBias = ddgi.hysteresis.z;
    let viewBiasAmount = ddgi.hysteresis.w;

    let biasedPos = pos_world + geometricN * normalBias + V * viewBiasAmount;
    let fractIdx = (biasedPos - gridMin) / spacing;
    let baseIdx = vec3i(floor(fractIdx));
    let alpha = fractIdx - floor(fractIdx);

    var totalIrr = vec3f(0.0);
    var totalW = 0.0;

    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let offset = vec3i(dx, dy, dz);
                let gridIdx = clamp(baseIdx + offset, vec3i(0), ddgi.grid_count.xyz - vec3i(1));
                let p_pos = ddgiProbePosition(gridIdx, ddgi);
                let p_idx = ddgiProbeLinearIndex(gridIdx, ddgi);

                let p_dir = p_pos - biasedPos;
                let p_dist = length(p_dir);
                let p_dirN = select(geometricN, normalize(p_dir), p_dist > 0.001);

                let p_wrap = (dot(p_dirN, geometricN) + 1.0) * 0.5;
                if (p_wrap <= 0.0) { continue; }

                let p_tri = vec3f(
                    select(1.0 - alpha.x, alpha.x, dx == 1),
                    select(1.0 - alpha.y, alpha.y, dy == 1),
                    select(1.0 - alpha.z, alpha.z, dz == 1)
                );
                var p_w = p_tri.x * p_tri.y * p_tri.z;

                let visUV = ddgiVisibilityTexelCoord(p_idx, octEncode(-p_dirN), ddgi);
                let visMoments = textureSampleLevel(ddgiVisibilityAtlas, ddgiSampler, visUV, 0.0).rg;
                let mean = visMoments.x;
                let variance = abs(visMoments.y - mean * mean);

                let d = p_dist;
                
                // ADD MINIMUM VARIANCE TO PREVENT HARSH STEP-FUNCTION BANDING AROUND ISOSURFACES!
                // Since variance naturally decays to 0 in totally static scenes, chebyshev would snap to 0.
                let safeVariance = max(variance, 0.0001); 
                let chebyshev = safeVariance / (safeVariance + (d - mean) * (d - mean));
                
                let vis = select(chebyshev, 1.0, d <= mean);
                
                // Increase exponent to sharpen shadows, but avoid making it totally blocky
                let smoothVis = clamp(vis * vis * vis, 0.0, 1.0);

                p_w *= max(0.001, smoothVis); // Don't let weight go to absolutely 0 to avoid artifacts
                p_w *= p_wrap;

                if (p_w < 0.00001) { continue; }

                let irrUV = ddgiIrradianceTexelCoord(p_idx, octEncode(geometricN), ddgi);
                let p_irr = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, irrUV, 0.0).rgb;
                totalIrr += p_irr * p_w;
                totalW += p_w;
            }
        }
    }

    if (totalW > 0.0) { totalIrr /= totalW; }
    return pow(max(totalIrr, vec3f(0.0)), vec3f(5.0));
}

fn evaluateRCProbes(
    pos_world: vec3f,
    geometricN: vec3f,
    V: vec3f,
    rc: RCUniforms,
    rcAtlas: texture_2d<f32>,
    rcSampler: sampler
) -> vec3f {
    let spacing = rc.grid_spacing.xyz;
    let gridMin = rc.grid_min.xyz;
    let normalBias = rc.params.x; // hysteresis used for blend, but wait... 
    // Actually rcUniformBuffer layout: 
    // rc.params = rcParams (hysteresis, intensity, ambient, enabled)
    // We don't have normal bias explicitly in the uniforms anymore. Let's just hardcode a small bias.
    let normalBiasAmount = 0.25;
    let viewBiasAmount = 0.1;
    
    let biasedPos = pos_world + geometricN * normalBiasAmount + V * viewBiasAmount;
    let fractIdx = (biasedPos - gridMin) / spacing;
    let baseIdx = vec3i(floor(fractIdx));
    let alpha = fractIdx - floor(fractIdx);

    var totalIrr = vec3f(0.0);
    var totalW = 0.0;

    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let offset = vec3i(dx, dy, dz);
                let gridIdx = clamp(baseIdx + offset, vec3i(0), rc.grid_count.xyz - vec3i(1));
                
                // probeWorldPos
                let p_pos = vec3f(
                    gridMin.x + f32(gridIdx.x) * spacing.x,
                    gridMin.y + f32(gridIdx.y) * spacing.y,
                    gridMin.z + f32(gridIdx.z) * spacing.z
                );
                
                let p_idx = gridIdx.z * rc.grid_count.x * rc.grid_count.y + gridIdx.y * rc.grid_count.x + gridIdx.x;

                let p_dir = p_pos - biasedPos;
                let p_dist = length(p_dir);
                let p_dirN = select(geometricN, normalize(p_dir), p_dist > 0.001);

                let p_tri = vec3f(
                    select(1.0 - alpha.x, alpha.x, dx == 1),
                    select(1.0 - alpha.y, alpha.y, dy == 1),
                    select(1.0 - alpha.z, alpha.z, dz == 1)
                );
                
                // Aggressively cull embedded probes using power wrapper (prevents global 50% dimming)
                let spherical_wrap = (dot(p_dirN, geometricN) + 1.0) * 0.5;
                let p_wrap = pow(spherical_wrap, 4.0);
                if (p_wrap <= 0.0001) { continue; }
                
                let p_w = p_tri.x * p_tri.y * p_tri.z * p_wrap;

                // Same texel coord math as DDGI but without the helper functions which we'll inline
                let texelsPerProbe = i32(rc.atlas_dims.y); // TEXELS_WITH_BORDER (10)
                let probesPerRow = 800;
                let probeRow = p_idx / probesPerRow;
                let probeCol = p_idx % probesPerRow;
                
                let probeOriginX = probeCol * texelsPerProbe;
                let probeOriginY = probeRow * texelsPerProbe;
                
                let octUV = octEncode(geometricN);
                
                // Inset UVs by half texel to avoid sampling from uninitialized border
                let interiorSize = f32(rc.atlas_dims.x); // TEXELS (8)
                let inset = 0.5 / interiorSize;
                let safeUV = clamp(octUV, vec2f(inset), vec2f(1.0 - inset));
                
                let mappedUV = safeUV * interiorSize;
                let finalUV = vec2f(
                    (f32(probeOriginX) + 1.0 + mappedUV.x) / f32(rc.atlas_dims.z),
                    (f32(probeOriginY) + 1.0 + mappedUV.y) / f32(rc.atlas_dims.w)
                );

                let p_irr = textureSampleLevel(rcAtlas, rcSampler, finalUV, 0.0).rgb;
                totalIrr += p_irr * p_w;
                totalW += p_w;
            }
        }
    }
    
    if (totalW > 0.0) { 
        totalIrr /= totalW; 
    }
    return totalIrr;
}

fn evaluateNRC(
    screenUV: vec2f,
    nrcInferenceTex: texture_2d<f32>
) -> vec3f {
    let nrcTexSize = textureDimensions(nrcInferenceTex);
    let nrcCoord = vec2i(i32(screenUV.x * f32(nrcTexSize.x)), i32(screenUV.y * f32(nrcTexSize.y)));
    return textureLoad(nrcInferenceTex, nrcCoord, 0).rgb;
}

fn evaluateSurfel(
    screenUV: vec2f,
    surfelIrradianceTex: texture_2d<f32>
) -> vec3f {
    let surfelTexSize = textureDimensions(surfelIrradianceTex);
    let surfelCoord = vec2i(i32(screenUV.x * f32(surfelTexSize.x)), i32(screenUV.y * f32(surfelTexSize.y)));
    var surfelIrradiance = textureLoad(surfelIrradianceTex, surfelCoord, 0).rgb;
    
    // Filter NaNs mathematically
    if (surfelIrradiance.x != surfelIrradiance.x || surfelIrradiance.y != surfelIrradiance.y || surfelIrradiance.z != surfelIrradiance.z || 
        !(surfelIrradiance.x >= 0.0 && surfelIrradiance.x <= 100000.0) ||
        !(surfelIrradiance.y >= 0.0 && surfelIrradiance.y <= 100000.0) ||
        !(surfelIrradiance.z >= 0.0 && surfelIrradiance.z <= 100000.0)) {
        surfelIrradiance = vec3f(0.0);
    }
    return surfelIrradiance;
}
