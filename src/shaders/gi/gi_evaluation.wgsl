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
                let chebyshev = variance / (variance + (d - mean) * (d - mean));
                let vis = select(chebyshev, 1.0, d <= mean);
                let smoothVis = clamp(vis * vis * vis, 0.0, 1.0);

                p_w *= max(0.05, smoothVis);
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

fn evaluateHybridSSGI(
    fragcoord_xy: vec2f,
    pos_world: vec3f,
    N: vec3f,
    albedo: vec3f,
    giTotalIrr: vec3f,
    iblIrradiance: vec3f,
    camera: CameraUniforms,
    screen_width: f32,
    screen_height: f32,
    gBufferPosition: texture_2d<f32>,
    gBufferAlbedo: texture_2d<f32>,
    ssgi_enabled_flag: f32 // rcParams.params.w
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
                        let hitIrradiance = giTotalIrr * 2.5 + iblIrradiance * 0.5;
                        ssgiRadiance += hitAlbedo * hitIrradiance;
                        
                        ssgiHitCount += 1.0;
                        break;
                    }
                }
            }
        }
    }
    
    let giBounce = giTotalIrr * albedo;
    let avgSSGI = select(vec3f(0.0), ssgiRadiance / max(ssgiHitCount, 1.0), ssgiHitCount > 0.1);
    let hitRatio = ssgiHitCount / f32(max(numSSGIRays, 1));
    
    let directBounces = mix(giBounce, avgSSGI * albedo, hitRatio);
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
