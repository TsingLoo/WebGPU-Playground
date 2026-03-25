// ==========================================
// Global Illumination Evaluation Functions
// ==========================================

// SSGI Helper: Hash functions for random sampling
fn hash22(p: vec2f) -> vec2f {
    var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// SSGI Helper: Cosine-weighted hemisphere sample
fn getCosHemisphereSample(n: vec3f, u: vec2f) -> vec3f {
    let r = sqrt(u.x);
    let theta = 2.0 * 3.14159265359 * u.y;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u.x));
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return tangent * x + bitangent * y + n * z;
}

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

fn evaluateHybridSSGI(
    fragcoord_xy: vec2f,
    pos_world: vec3f,
    N: vec3f,
    albedo: vec3f,
    ddgi_totalIrr: vec3f,
    iblIrradiance: vec3f,
    camera: CameraUniforms,
    screen_width: f32,
    screen_height: f32,
    gBufferPosition: texture_2d<f32>,
    gBufferAlbedo: texture_2d<f32>,
    ssgi_enabled_flag: f32 // ddgiParams.ddgi_enabled.w
) -> vec3f {
    var ssgiRadiance = vec3f(0.0);
    var ssgiHitCount = 0.0;
    
    let numSSGIRays = select(0, 2, ssgi_enabled_flag > 0.5); 
    for (var i = 0; i < numSSGIRays; i++) {
        let bayer = fract(fragcoord_xy.x * 0.5 + fragcoord_xy.y * 0.25);
        let u = vec2f(fract(bayer + f32(i)*0.5), fract(bayer*1.618 + f32(i)*0.618));
        
        let rayDir = getCosHemisphereSample(N, u);
        let rayOrigin = pos_world + N * 0.05;
        let rayEnd = rayOrigin + rayDir * 10.0; 
        
        let originView = (camera.view_mat * vec4f(rayOrigin, 1.0)).xyz;
        let endView = (camera.view_mat * vec4f(rayEnd, 1.0)).xyz;
        
        if (originView.z < 0.0) { 
            let p0Clip = camera.proj_mat * vec4f(originView, 1.0);
            let p1Clip = camera.proj_mat * vec4f(endView, 1.0);
            let p0NDC = p0Clip.xy / p0Clip.w;
            let p1NDC = p1Clip.xy / p1Clip.w;
            
            let screenDims = vec2f(screen_width, screen_height);
            let p0Screen = (p0NDC * vec2f(0.5, -0.5) + 0.5) * screenDims;
            let p1Screen = (p1NDC * vec2f(0.5, -0.5) + 0.5) * screenDims;
            
            let deltaScreen = p1Screen - p0Screen;
            let steps = min(20.0, max(abs(deltaScreen.x), abs(deltaScreen.y)));
            
            if (steps > 1.0) {
                let stepSize = 1.0 / steps;
                let z0 = originView.z;
                let z1 = min(endView.z, -0.1); 
                let invZ0 = 1.0 / z0;
                let invZ1 = 1.0 / z1;
                
                for (var s = 1; s <= 32; s++) {
                    let t = f32(s) * stepSize;
                    if (t > 1.0) { break; }
                    
                    let ssi = vec2i(mix(p0Screen, p1Screen, t));
                    if (ssi.x < 0 || ssi.y < 0 || ssi.x >= i32(screenDims.x) || ssi.y >= i32(screenDims.y)) { break; }
                    
                    let hitPos = textureLoad(gBufferPosition, ssi, 0).xyz;
                    if (dot(hitPos, hitPos) < 0.1) { continue; }
                    
                    let hitViewZ = (camera.view_mat * vec4f(hitPos, 1.0)).z;
                    let expectedInvZ = mix(invZ0, invZ1, t);
                    let expectedZ = 1.0 / expectedInvZ;
                    
                    let thickness = 1.0; 
                    if (hitViewZ > expectedZ && hitViewZ < expectedZ + thickness) {
                        let hitAlbedo = textureLoad(gBufferAlbedo, ssi, 0).rgb;
                        let hitIrradiance = ddgi_totalIrr * 2.5 + iblIrradiance * 0.5;
                        ssgiRadiance += hitAlbedo * hitIrradiance;
                        
                        ssgiHitCount += 1.0;
                        break;
                    }
                }
            }
        }
    }
    
    let ddgiBounce = ddgi_totalIrr * albedo;
    let avgSSGI = select(vec3f(0.0), ssgiRadiance / max(ssgiHitCount, 1.0), ssgiHitCount > 0.1);
    let hitRatio = ssgiHitCount / f32(max(numSSGIRays, 1));
    
    let directBounces = mix(ddgiBounce, avgSSGI * albedo, hitRatio);
    let iblFloor = iblIrradiance * albedo * 0.25;
    return max(directBounces, iblFloor);
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
