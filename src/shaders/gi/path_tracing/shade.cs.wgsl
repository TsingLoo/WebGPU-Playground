// shade.cs.wgsl
// Wavefront Path Tracing — Pass 3: Material Shading + NEE
// Full PBR with normal mapping + metallic-roughness texture support.
// Reconstructs pos/uv/normal/tangent from compact HitRecord + BVH buffers.
// Supports spectral rendering mode (hero wavelength, dispersion, spectral throughput).

@group(0) @binding(0)  var<uniform>             camera:             CameraUniforms;
@group(0) @binding(1)  var<uniform>             pt:                 PTUniforms;
@group(0) @binding(2)  var<storage, read_write>  ray_buffer:         array<PTRay>;
@group(0) @binding(3)  var<storage, read_write>  hit_buffer:         array<HitRecord>;
@group(0) @binding(4)  var<storage, read_write>  shadow_buffer:      array<ShadowRay>;
@group(0) @binding(5)  var<storage, read_write>  accum_buffer:       array<vec4f>;
@group(0) @binding(6)  var<storage, read>        materials:          array<vec4f>;
@group(0) @binding(7)  var<uniform>              sun_light:          SunLight;
@group(0) @binding(8)  var                       base_color_tex:     texture_2d_array<f32>;
@group(0) @binding(9)  var                       tex_sampler:        sampler;
@group(0) @binding(10) var                       normal_map_tex:     texture_2d_array<f32>;
@group(0) @binding(11) var                       mr_tex:             texture_2d_array<f32>;
// BVH vertex data for reconstructing hit attributes
@group(0) @binding(12) var<storage, read>        bvh_uvs:            array<vec4f>;
@group(0) @binding(13) var<storage, read>        bvh_normals:        array<vec4f>;
@group(0) @binding(14) var<storage, read>        bvh_tangents:       array<vec4f>;
@group(0) @binding(15) var                       emissive_tex:       texture_2d_array<f32>;

// NRC Group
@group(1) @binding(0)  var<uniform>              nrc:                NRCUniforms;
@group(1) @binding(1)  var<storage, read>        nrc_weights:        array<f32>;
@group(1) @binding(2)  var<storage, read_write>  sampleCounter:      atomic<u32>;
@group(1) @binding(3)  var<storage, read_write>  pt_nrc_train_data:  array<NRCWavefrontTrainData>;

// Spectral Group
@group(2) @binding(0) var<storage, read>         spectral_wavelengths:    array<vec4f>;
@group(2) @binding(1) var<storage, read>         spectral_pdfs:           array<vec4f>;
@group(2) @binding(2) var<storage, read_write>   spectral_throughput_buf: array<vec4f>;
@group(2) @binding(3) var<storage, read_write>   spectral_accum_buf:      array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = pt.width;
    let render_height = pt.height;
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    // Default: deactivate shadow ray
    var shadow: ShadowRay;
    shadow.shadow_active = 0u;
    shadow_buffer[pixel_id] = shadow;

    var ray = ray_buffer[pixel_id];
    if (ray.ray_active == 0u) { return; }

    let hit = hit_buffer[pixel_id];

    // Miss: leave ray state for the miss shader to handle
    if (hit.did_hit == 0u) {
        if (ray.ray_active == 2u) {
            // Training ray missed, we still need to clear it so it doesn't loop forever
            ray.ray_active = 0u;
        }
        ray_buffer[pixel_id] = ray;
        return;
    }

    // ============================================================
    // Spectral state (loaded once, used throughout)
    // ============================================================
    let is_spectral = (pt.spectral_enabled == 1u);
    var lambdas = vec4f(550.0);
    var spec_throughput = vec4f(1.0);
    if (is_spectral) {
        lambdas         = spectral_wavelengths[pixel_id];
        spec_throughput = spectral_throughput_buf[pixel_id];
    }

    // ============================================================
    // Reconstruct hit attributes from BVH + compact HitRecord
    // ============================================================
    let bw = vec3f(hit.bary.x, hit.bary.y, 1.0 - hit.bary.x - hit.bary.y);

    // Position: from ray parametric
    let hit_pos = ray.origin + ray.direction * hit.dist;

    // UV interpolation
    let uv0 = bvh_uvs[hit.idx0].xy;
    let uv1 = bvh_uvs[hit.idx1].xy;
    let uv2 = bvh_uvs[hit.idx2].xy;
    let hit_uv = bw.x * uv0 + bw.y * uv1 + bw.z * uv2;

    // Smooth normal interpolation
    let n0 = bvh_normals[hit.idx0].xyz;
    let n1 = bvh_normals[hit.idx1].xyz;
    let n2 = bvh_normals[hit.idx2].xyz;
    var smooth_N = normalize(bw.x * n0 + bw.y * n1 + bw.z * n2);
    // Ensure normal faces the incoming ray
    if (dot(smooth_N, ray.direction) > 0.0) {
        smooth_N = -smooth_N;
    }

    // Tangent interpolation
    let t0 = bvh_tangents[hit.idx0];
    let t1 = bvh_tangents[hit.idx1];
    let t2 = bvh_tangents[hit.idx2];
    let interp_tangent_dir = normalize(bw.x * t0.xyz + bw.y * t1.xyz + bw.z * t2.xyz);
    let handedness = t0.w; // same for all verts of a triangle

    // ============================================================
    // Material evaluation
    // ============================================================
    let mat = unpackPTMaterial(&materials, hit.mat_id);

    var albedo = mat.albedo;
    var final_alpha = mat.alpha;
    if (mat.tex_layer >= 0) {
        let tex_col = textureSampleLevel(base_color_tex, tex_sampler, hit_uv, mat.tex_layer, 0.0);
        // Base color textures are sRGB-encoded; decode to linear for PBR math
        albedo *= srgbToLinear(tex_col.xyz);
        final_alpha *= tex_col.w;
    }

    // Alpha test: if the surface is transparent (leaf cutouts, etc.), pass through
    if (mat.alpha_mode != 0u && final_alpha < mat.alpha_cutoff) {
        // Advance the ray slightly past the hit point and continue tracing
        ray.origin    = hit_pos + ray.direction * 0.001;
        // Don't change direction, throughput, or bounce — the ray just passes through
        ray_buffer[pixel_id] = ray;
        return;
    }

    // --- Metallic-Roughness map sampling ---
    var roughness = mat.roughness;
    var metallic  = mat.metallic;
    if (mat.mr_tex_layer >= 0) {
        let mr_sample = textureSampleLevel(mr_tex, tex_sampler, hit_uv, mat.mr_tex_layer, 0.0);
        // glTF spec: G = roughness, B = metallic
        roughness *= mr_sample.g;
        metallic  *= mr_sample.b;
    }
    roughness = max(roughness, 0.04);

    // --- Normal mapping ---
    var N = smooth_N;
    if (mat.normal_tex_layer >= 0) {
        let T_raw = interp_tangent_dir - N * dot(interp_tangent_dir, N);
        let T_len = length(T_raw);
        var T = vec3f(0.0);
        if (T_len > 0.001) {
            T = T_raw / T_len;
        } else {
            let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
            T = normalize(cross(N, refVec));
        }
        let h = select(handedness, 1.0, abs(handedness) < 0.5);
        let B = normalize(cross(N, T)) * h;
        let tbn = mat3x3f(T, B, N);

        let normal_sample = textureSampleLevel(normal_map_tex, tex_sampler, hit_uv, mat.normal_tex_layer, 0.0).rgb;
        let tangent_normal = normal_sample * 2.0 - 1.0;
        N = normalize(tbn * tangent_normal);
    }

    // =================================================================
    // USER DEBUG VISUALIZER
    // Set to 1=Normals, 2=Barycentric, 3=Albedo, 4=Alpha Debug
    // =================================================================
    const DEBUG_VIEW_MODE: u32 = 0u; // Change this to debug

    if (DEBUG_VIEW_MODE != 0u && ray.bounce == 0u && ray.ray_active != 2u) {
        if (DEBUG_VIEW_MODE == 1u) {
            accum_buffer[pixel_id] = vec4f(abs(N), 1.0);
        } else if (DEBUG_VIEW_MODE == 2u) {
            accum_buffer[pixel_id] = vec4f(bw, 1.0);
        } else if (DEBUG_VIEW_MODE == 3u) {
            accum_buffer[pixel_id] = vec4f(albedo, 1.0);
        } else if (DEBUG_VIEW_MODE == 4u) {
            if (mat.alpha_mode != 0u && final_alpha < mat.alpha_cutoff) {
                accum_buffer[pixel_id] = vec4f(1.0, 0.0, 0.0, 1.0); // Red = skipped transparency
            } else {
                accum_buffer[pixel_id] = vec4f(final_alpha, final_alpha, final_alpha, 1.0); // Grayscale = opaque alpha
            }
        }
        ray.ray_active = 0u;
        ray_buffer[pixel_id] = ray;
        return;
    }

    var rng = initRNG(pixel_id ^ (ray.bounce * 1009u), pt.frame_index);

    let V     = -ray.direction;
    let NdotV = max(dot(N, V), 0.0001);

    // ============================================================
    // Spectral: convert material properties to spectral domain
    // ============================================================
    var spec_albedo = vec4f(0.0);
    var spec_F0     = vec4f(0.04);
    if (is_spectral) {
        spec_albedo = rgbToSpectrum(albedo, lambdas);
        // F0 for dielectrics in spectral: 0.04 uniform, for metals: spectral albedo
        let dielectric_F0 = rgbToSpectrum(vec3f(0.04), lambdas);
        let metal_F0      = spec_albedo;
        spec_F0 = mix(dielectric_F0, metal_F0, metallic);
    }

    // Emissive
    var final_emission = mat.emissive;
    if (mat.emissive_tex_layer >= 0) {
        let em_sample = textureSampleLevel(emissive_tex, tex_sampler, hit_uv, mat.emissive_tex_layer, 0.0).rgb;
        final_emission *= em_sample;
    }
    
    if (any(final_emission > vec3f(0.001))) {
        if (is_spectral) {
            // Convert emissive to spectral, multiply by spectral throughput, accumulate
            let spec_emission = rgbToSpectrum(final_emission, lambdas);
            let spec_contrib = spec_throughput * spec_emission;
            let clamped_spec = min(spec_contrib, vec4f(pt.clamp_radiance));
            let prev = spectral_accum_buf[pixel_id];
            spectral_accum_buf[pixel_id] = prev + clamped_spec;
        } else {
            let prev = accum_buffer[pixel_id];
            let em   = clamp(ray.throughput * final_emission, vec3f(0.0), vec3f(pt.clamp_radiance));
            accum_buffer[pixel_id] = vec4f(prev.xyz + em, prev.w);
        }
    }

    // =================================================================
    // Glass / Transmissive
    // =================================================================
    if (mat.transmission > 0.5) {
        if (is_spectral) {
            // Spectral dispersion path
            let eta_base = mat.ior;
            let spec_ior = spectralIOR(eta_base, lambdas);
            let air_ior  = vec4f(1.0);

            let eta_in  = select(spec_ior, air_ior, hit.side > 0.0);
            let eta_out = select(air_ior, spec_ior, hit.side > 0.0);

            let cos_i = max(dot(N, V), 0.0001);
            let F_spec = fresnelDielectricSpectral(cos_i, eta_in, eta_out);

            // Use hero wavelength (index 0) for path decision
            let F_hero = F_spec[0];
            let eta_hero_in  = eta_in[0];
            let eta_hero_out = eta_out[0];

            var new_dir: vec3f;
            if (rand(&rng) < F_hero) {
                new_dir = reflect(-V, N);
            } else {
                let eta = eta_hero_in / eta_hero_out;
                let refracted = refract(-V, N * hit.side, eta);
                if (all(refracted == vec3f(0.0))) {
                    new_dir = reflect(-V, N);
                } else {
                    new_dir = normalize(refracted);
                    // Terminate secondary wavelengths that would go in different directions
                    // (pbrt v4 TerminateSecondary approach)
                    // For now, we weight the non-hero channels by their Fresnel ratio
                    for (var i = 1u; i < N_SPECTRAL_SAMPLES; i++) {
                        let eta_i = eta_in[i] / eta_out[i];
                        let sin2_t = eta_i * eta_i * (1.0 - cos_i * cos_i);
                        if (sin2_t >= 1.0) {
                            // TIR for this wavelength — zero its contribution
                            spec_throughput[i] = 0.0;
                        }
                    }
                }
            }

            // Spectral throughput update: albedo tint
            spec_throughput *= rgbToSpectrum(albedo, lambdas);
            spectral_throughput_buf[pixel_id] = spec_throughput;

            ray.throughput       = vec3f(1.0); // placeholder, unused in spectral mode
            ray.origin           = hit_pos + new_dir * 0.001;
            ray.direction        = new_dir;
            ray.ior              = eta_hero_out;
            ray.bounce          += 1u;
            ray.ray_active       = select(0u, 1u, ray.bounce < pt.max_bounces);
            ray.specular_bounce  = 1u;
            ray_buffer[pixel_id] = ray;
            return;
        } else {
            // Original RGB glass path
            let eta_in  = select(mat.ior, 1.0, hit.side > 0.0);
            let eta_out = select(1.0, mat.ior, hit.side > 0.0);
            let cos_i   = max(dot(N, V), 0.0001);
            let F       = fresnelDielectric(cos_i, eta_in, eta_out);

            var new_dir: vec3f;
            var new_ior: f32;
            if (rand(&rng) < F) {
                new_dir = reflect(-V, N);
                new_ior = ray.ior;
            } else {
                let eta       = eta_in / eta_out;
                let refracted = refract(-V, N * hit.side, eta);
                if (all(refracted == vec3f(0.0))) {
                    new_dir = reflect(-V, N);
                    new_ior = ray.ior;
                } else {
                    new_dir = normalize(refracted);
                    new_ior = eta_out;
                }
            }
            ray.throughput      *= albedo;
            ray.origin           = hit_pos + new_dir * 0.001;
            ray.direction        = new_dir;
            ray.ior              = new_ior;
            ray.bounce          += 1u;
            ray.ray_active       = select(0u, 1u, ray.bounce < pt.max_bounces);
            ray.specular_bounce  = 1u;
            ray_buffer[pixel_id] = ray;
            return;
        }
    }

    // =================================================================
    // Opaque PBR
    // =================================================================
    let a  = roughness * roughness;
    let a2 = a * a;
    let F0 = mix(vec3f(0.04), albedo, metallic);

    // Russian Roulette (only after bounce 2)
    // We let training rays (ray_active == 2u) run without RR to stabilize target radiance, or we can use RR.
    // Following the paper, we leave RR enabled for unbiased targets, but we must ignore ray_active == 2u 
    // termination if we want to ensure N=5. Let's just disable RR for training paths (ray_active == 2) for better variance.
    var rr_max: f32;
    if (is_spectral) {
        // sRGB response basis can produce negative spectral values; use abs for RR
        rr_max = max(abs(spec_throughput[0]), max(abs(spec_throughput[1]), max(abs(spec_throughput[2]), abs(spec_throughput[3]))));
    } else {
        rr_max = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));
    }
    let rr_survive = clamp(rr_max, 0.05, 1.0);
    if (ray.bounce >= 2u && ray.ray_active != 2u && rand(&rng) > rr_survive) {
        ray.ray_active = 0u;
        ray_buffer[pixel_id] = ray;
        return;
    }
    let rr_weight = select(1.0 / rr_survive, 1.0, ray.bounce < 2u || ray.ray_active == 2u);

    // Stochastic lobe selection: specular vs diffuse
    let F_avg = fresnelSchlickV(NdotV, F0);
    let spec_weight = max(F_avg.x, max(F_avg.y, F_avg.z));
    let spec_prob   = clamp(spec_weight, 0.1, 0.9);
    let use_specular = rand(&rng) < spec_prob;

    var new_dir: vec3f;
    var new_throughput: vec3f;
    var is_specular: bool = false;

    // Spectral throughput update variables
    var new_spec_throughput = spec_throughput;

    if (use_specular) {
        // Retry GGX sampling a few times to find a valid above-hemisphere direction.
        // For rough surfaces, a large fraction of GGX samples produce NdotL <= 0.
        var found_valid = false;
        var L = vec3f(0.0);
        var NdotH_s = 0.0;
        var VdotH_s = 0.0;
        var NdotL_s = 0.0;
        for (var attempt = 0u; attempt < 4u; attempt++) {
            let H      = sampleGGX(N, roughness, &rng);
            L          = normalize(reflect(-V, H));
            NdotL_s    = dot(N, L);
            if (NdotL_s > 0.0) {
                NdotH_s = max(dot(N, H), 0.0001);
                VdotH_s = max(dot(V, H), 0.0001);
                found_valid = true;
                break;
            }
        }
        if (!found_valid) {
            // All GGX attempts failed — fall back to cosine hemisphere
            L = sampleCosineHemisphere(N, &rng);
            NdotL_s = max(dot(N, L), 0.0001);
            let H_fb = normalize(V + L);
            NdotH_s = max(dot(N, H_fb), 0.0001);
            VdotH_s = max(dot(V, H_fb), 0.0001);
        }

        let Fspec  = fresnelSchlickV(VdotH_s, F0);
        let G      = geometrySchlickGGX_pt(NdotV, roughness)
                   * geometrySchlickGGX_pt(max(NdotL_s, 0.0001), roughness);

        let spec_brdf_weight = Fspec * G * VdotH_s / max(NdotH_s * NdotV, 0.0001);
        new_dir       = L;
        new_throughput = ray.throughput * spec_brdf_weight * (rr_weight / spec_prob);
        is_specular   = (roughness < 0.05);

        if (is_spectral) {
            // Spectral BRDF weight: use spectral F0 for Fresnel
            let Fspec_s  = fresnelSchlickSpectral(VdotH_s, spec_F0);
            let spec_brdf_w = Fspec_s * G * VdotH_s / max(NdotH_s * NdotV, 0.0001);
            new_spec_throughput = spec_throughput * spec_brdf_w * (rr_weight / spec_prob);
        }
    } else {
        let L       = sampleCosineHemisphere(N, &rng);
        let kD      = (vec3f(1.0) - fresnelSchlickV(NdotV, F0)) * (1.0 - metallic);
        new_dir       = L;
        new_throughput = ray.throughput * albedo * kD * (rr_weight / (1.0 - spec_prob));

        if (is_spectral) {
            let kD_spec = (vec4f(1.0) - fresnelSchlickSpectral(NdotV, spec_F0)) * (1.0 - metallic);
            new_spec_throughput = spec_throughput * spec_albedo * kD_spec * (rr_weight / (1.0 - spec_prob));
        }
    }

    // Clamp throughput to prevent fireflies
    if (is_spectral) {
        let tp_max = max(abs(new_spec_throughput[0]), max(abs(new_spec_throughput[1]), max(abs(new_spec_throughput[2]), abs(new_spec_throughput[3]))));
        if (tp_max > 20.0) {
            new_spec_throughput *= 20.0 / tp_max;
        }
        spectral_throughput_buf[pixel_id] = new_spec_throughput;
    } else {
        let tp_max = max(new_throughput.x, max(new_throughput.y, new_throughput.z));
        if (tp_max > 20.0) {
            new_throughput *= 20.0 / tp_max;
        }
    }

    // NEE — direct sun light (skipped for bounce 0 when ReSTIR is active)
    let skip_nee = (pt.restir_enabled == 1u) && (ray.bounce == 0u);
    if (sun_light.color.a > 0.5 && !is_specular && !skip_nee) {
        let base_sun_dir = normalize(sun_light.direction.xyz);
        let up_vec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(base_sun_dir.x) > 0.9);
        let sun_tangent = normalize(cross(base_sun_dir, up_vec));
        let sun_bitangent = cross(base_sun_dir, sun_tangent);
        let r2 = rand2(&rng);
        let r = sqrt(r2.x) * 0.02;
        let theta = 2.0 * PI * r2.y;
        let sun_dir = normalize(base_sun_dir + sun_tangent * r * cos(theta) + sun_bitangent * r * sin(theta));

        let NdotL_s  = max(dot(N, sun_dir), 0.0);
        if (NdotL_s > 0.0) {
            let H_s     = normalize(V + sun_dir);
            let NdotH_s = max(dot(N, H_s), 0.0);
            let VdotH_s = max(dot(V, H_s), 0.0);
            let F_s     = fresnelSchlickV(VdotH_s, F0);
            let G_s     = geometrySchlickGGX_pt(NdotV, roughness)
                        * geometrySchlickGGX_pt(NdotL_s, roughness);
            let D_s     = distributionGGX_pt(NdotH_s, a2);
            let spec_s  = F_s * D_s * G_s / max(4.0 * NdotV * NdotL_s, 0.0001);
            let diff_s  = (vec3f(1.0) - F_s) * (1.0 - metallic) * albedo / PI;
            let sun_rad = sun_light.color.rgb * sun_light.direction.w;
            let Lo_sun  = (diff_s + spec_s) * sun_rad * NdotL_s;

            if (is_spectral) {
                // For NEE in spectral mode: we store the RGB-based Li and also
                // separately accumulate spectral NEE contribution.
                // We convert the sun radiance to spectral, evaluate spectral BRDF,
                // and store the spectral contribution directly.
                let spec_sun_rad = rgbToSpectrum(sun_rad, lambdas);
                let F_s_spec     = fresnelSchlickSpectral(VdotH_s, spec_F0);
                let spec_s_val   = F_s_spec * D_s * G_s / max(4.0 * NdotV * NdotL_s, 0.0001);
                let diff_s_spec  = (vec4f(1.0) - F_s_spec) * (1.0 - metallic) * spec_albedo / PI;
                let Lo_sun_spec  = (diff_s_spec + spec_s_val) * spec_sun_rad * NdotL_s;
                let spec_Li      = min(spec_throughput * Lo_sun_spec, vec4f(pt.clamp_radiance));

                // Store in shadow ray Li field as RGB (hero-wavelength approximation for shadow test)
                // The full spectral contribution goes through the spectral accum buffer
                shadow.origin       = hit_pos + smooth_N * 0.002;
                shadow.max_dist     = 1000.0;
                shadow.direction    = sun_dir;
                shadow.pixel_id     = pixel_id;
                // Pack luminance from spectral into Li for shadow test visibility
                shadow.Li           = clamp(ray.throughput * Lo_sun, vec3f(0.0), vec3f(pt.clamp_radiance));
                shadow.shadow_active = 1u;

                // We'll store the spectral NEE Li in a temporary slot in the accum_buffer's .w channel
                // Actually, we need the shadow test to know both RGB and spectral contributions.
                // Simplification: shadow test only does visibility. If unoccluded, shade already
                // computed the spectral contribution. We'll handle this by pre-accumulating
                // the spectral NEE contribution pending shadow test validation.
                // Store spectral NEE in the _pad fields of ShadowRay? No — struct is fixed.
                // Use a separate spectral shadow buffer? That's expensive.
                // Simplest: store spectral NEE contribution in spectral_accum_buf now,
                //   and if shadow test finds occlusion, we subtract it back.
                //   BUT shadow_test adds Li to accum on un-occluded, doesn't subtract.
                //
                // Final approach: Don't add spectral here. Let shadow_test handle it.
                // shadow.Li stores RGB-space value. After shadow test adds shadow.Li to accum,
                // accumulate pass will just use the RGB accum. For spectral correctness:
                // we convert the final RGB accum to spectral in the accumulate pass.
                //
                // ACTUALLY the cleanest approach: store spectral Li in spectral_accum_buf
                // conditionally, and shadow_test for spectral mode does its own spectral add.
                // But shadow_test doesn't have spectral bindings. So:
                //
                // We pre-store the spectral NEE in shadow ray Li (truncated to RGB, hero-wl weighted)
                // and accept a small spectral inaccuracy for direct lighting NEE.
                // The primary spectral correctness comes from indirect paths (throughput).
                // This is a reasonable trade-off for implementation simplicity.
            } else {
                shadow.origin       = hit_pos + smooth_N * 0.002;
                shadow.max_dist     = 1000.0;
                shadow.direction    = sun_dir;
                shadow.pixel_id     = pixel_id;
                shadow.Li           = clamp(ray.throughput * Lo_sun, vec3f(0.0), vec3f(pt.clamp_radiance));
                shadow.shadow_active = 1u;
            }
        }
    }
    shadow_buffer[pixel_id] = shadow;

    // Spawn next bounce (or evaluate NRC)
    ray.throughput      = new_throughput;
    ray.origin          = hit_pos + smooth_N * 0.001;
    ray.direction       = new_dir;
    ray.bounce         += 1u;
    
    // Default active state
    if (ray.ray_active != 2u) {
        ray.ray_active = select(0u, 1u, ray.bounce < pt.max_bounces);
    } else {
        ray.ray_active = select(0u, 2u, ray.bounce < pt.max_bounces);
    }
    ray.specular_bounce = select(0u, 1u, is_specular);

    if (is_spectral) {
        if (max(abs(new_spec_throughput[0]), max(abs(new_spec_throughput[1]), max(abs(new_spec_throughput[2]), abs(new_spec_throughput[3])))) < 0.001) {
            ray.ray_active = 0u;
        }
    } else {
        if (max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z)) < 0.001) {
            ray.ray_active = 0u;
        }
    }
    
    let is_inference_enabled = nrc.scene_min.w > 0.5; // w holds enabled flag

    // =================================================================
    // NRC Inference & Training Logic (Always at 2nd Vertex: bounce == 1)
    // NRC is disabled in spectral mode — it operates in RGB space only.
    // =================================================================
    if (ray.bounce == 1u && is_inference_enabled && ray.ray_active != 0u && !is_spectral) {
        // We are at the second vertex! Should we train or infer?
        // Randomly select a subset to train (approx bounding to MAX_TRAINING_SAMPLES)
        let total_pixels_f = f32(total_pixels);
        let train_fraction = f32(nrc.params.y) / total_pixels_f; // nrc.params.y holds the requested max samples
        
        let is_training_ray = rand(&rng) < train_fraction;
        
        var became_training = false;
        if (is_training_ray) {
            let slot = atomicAdd(&sampleCounter, 1u);
            if (slot < u32(nrc.params.y)) {
                // We got a slot! Save training features
                var train_data: NRCWavefrontTrainData;
                let features = nrcEncodeInput(hit_pos, smooth_N, -ray.direction, nrc.scene_min.xyz, nrc.scene_max.xyz);
                for (var i=0u; i<15u; i++) { train_data.features[i] = features[i]; }
                
                train_data.throughput = ray.throughput; // this is throughput AT the 2nd vertex
                train_data.primary_radiance = accum_buffer[pixel_id].xyz; // captures direct light from bounce 0
                train_data.pixel_id = pixel_id;
                train_data.is_active = 1u;
                pt_nrc_train_data[slot] = train_data;
                
                // Mark this ray to continue tracing up to pt.max_bounces
                ray.ray_active = 2u;
                became_training = true;
            }
        }
        
        if (!became_training) {
            // Path terminates here! Query NRC for outgoing radiance from this vertex
            let features = nrcEncodeInput(hit_pos, smooth_N, -ray.direction, nrc.scene_min.xyz, nrc.scene_max.xyz);
            let pred = nrcForward(features, &nrc_weights);
            
            // Tone map recovery, clamp and guard
            var clampedPred = clamp(pred, vec3f(0.0), vec3f(0.95));
            if (clampedPred.x != clampedPred.x || clampedPred.y != clampedPred.y || clampedPred.z != clampedPred.z) {
                clampedPred = vec3f(0.0);
            }
            let hdrPred = clampedPred / max(vec3f(1.0) - clampedPred, vec3f(0.001));
            
            // Add NRC contribution multiplied by the throughput up to this vertex
            let incoming_radiance = ray.throughput * hdrPred;
            
            let prev = accum_buffer[pixel_id];
            accum_buffer[pixel_id] = vec4f(prev.xyz + clamp(incoming_radiance, vec3f(0.0), vec3f(pt.clamp_radiance)), prev.w);
            
            // Terminate inference ray
            ray.ray_active = 0u;
        }
    }
    
    // =================================================================
    // Terminal Vertex for Training Rays
    // =================================================================
    if (ray.bounce == pt.max_bounces - 1u && ray.ray_active == 2u && is_inference_enabled && !is_spectral) {
        // Query NRC for the tail of the training ray
        let features = nrcEncodeInput(hit_pos, smooth_N, -ray.direction, nrc.scene_min.xyz, nrc.scene_max.xyz);
        let pred = nrcForward(features, &nrc_weights);
        
        var clampedPred = clamp(pred, vec3f(0.0), vec3f(0.95));
        if (clampedPred.x != clampedPred.x || clampedPred.y != clampedPred.y || clampedPred.z != clampedPred.z) {
            clampedPred = vec3f(0.0);
        }
        let hdrPred = clampedPred / max(vec3f(1.0) - clampedPred, vec3f(0.001));
        
        let incoming_radiance = ray.throughput * hdrPred;
        let prev = accum_buffer[pixel_id];
        accum_buffer[pixel_id] = vec4f(prev.xyz + clamp(incoming_radiance, vec3f(0.0), vec3f(pt.clamp_radiance)), prev.w);
        
        // Terminate naturally
    }

    ray_buffer[pixel_id] = ray;
}
