// ==========================================
// Global Illumination Evaluation Functions
// ==========================================

fn evaluateDDGI(
    pos_world: vec3f,
    geometricN: vec3f,
    V: vec3f,
    ddgiParams: DDGIUniforms,
    ddgiIrradianceAtlas: texture_2d<f32>,
    ddgiVisibilityAtlas: texture_2d<f32>,
    ddgiSampler: sampler
) -> vec3f {
    let ddgi_spacing = ddgiParams.grid_spacing.xyz;
    let ddgi_gridMin = ddgiParams.grid_min.xyz;
    let ddgi_normalBias = ddgiParams.hysteresis.z;
    let ddgi_viewBias = ddgiParams.hysteresis.w;
    
    // Apply normal bias and view bias to avoid self-intersection (DDGI shadow acne)
    let ddgi_biasedPos = pos_world + geometricN * ddgi_normalBias + V * ddgi_viewBias;
    let ddgi_fractIdx = (ddgi_biasedPos - ddgi_gridMin) / ddgi_spacing;
    let ddgi_baseIdx = vec3i(floor(ddgi_fractIdx));
    let ddgi_alpha = ddgi_fractIdx - floor(ddgi_fractIdx);

    var ddgi_totalIrr = vec3f(0.0);
    var ddgi_totalW = 0.0;

    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let p_offset = vec3i(dx, dy, dz);
                let p_gridIdx = clamp(ddgi_baseIdx + p_offset, vec3i(0), ddgiParams.grid_count.xyz - vec3i(1));
                let p_pos = ddgiProbePosition(p_gridIdx, ddgiParams);
                let p_idx = ddgiProbeLinearIndex(p_gridIdx, ddgiParams);

                let p_dir = ddgi_biasedPos - p_pos;
                let p_dist = length(p_dir);
                let p_dirN = select(geometricN, normalize(p_dir), p_dist > 0.001);

                // Use geometric normal for wrap-around test
                let p_wrap = (dot(p_dirN, geometricN) + 1.0) * 0.5;
                if (p_wrap <= 0.0) { continue; }

                let p_tri = vec3f(
                    select(1.0 - ddgi_alpha.x, ddgi_alpha.x, dx == 1),
                    select(1.0 - ddgi_alpha.y, ddgi_alpha.y, dy == 1),
                    select(1.0 - ddgi_alpha.z, ddgi_alpha.z, dz == 1)
                );
                var p_w = p_tri.x * p_tri.y * p_tri.z;

                // Chebyshev visibility
                let p_visUV = ddgiVisibilityTexelCoord(p_idx, octEncode(p_dirN), ddgiParams);
                let p_vis = textureSampleLevel(ddgiVisibilityAtlas, ddgiSampler, p_visUV, 0.0).rg;
                if (p_dist > p_vis.x) {
                    let p_var = max(p_vis.y - p_vis.x * p_vis.x, 0.0001);
                    let p_d = p_dist - p_vis.x;
                    let p_cheb = p_var / (p_var + p_d * p_d);
                    p_w *= max(p_cheb * p_cheb * p_cheb, 0.0);
                }

                p_w *= p_wrap;
                if (p_w < 0.00001) { continue; } // skip probes with negligible weight

                let p_irrUV = ddgiIrradianceTexelCoord(p_idx, octEncode(geometricN), ddgiParams);
                let p_irr_encoded = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, p_irrUV, 0.0).rgb;
                // Clamp to [0,1] before pow decode
                let p_irr = pow(clamp(p_irr_encoded, vec3f(0.0), vec3f(1.0)), vec3f(5.0));
                // Clamp decoded irradiance to prevent HDR fireflies
                let p_irr_clamped = min(p_irr, vec3f(10.0));
                ddgi_totalIrr += p_irr_clamped * p_w;
                ddgi_totalW += p_w;
            }
        }
    }
    
    if (ddgi_totalW > 0.0) { 
        ddgi_totalIrr /= ddgi_totalW; 
    }
    return ddgi_totalIrr;
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
