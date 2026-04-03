@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var hizTexture: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(2) var normalTex: texture_2d<f32>;

struct SSAOUniforms {
    radius: f32,
    bias: f32,
    power: f32,
    enabled: f32,
    kernel: array<vec4f, 64>,
}
@group(${bindGroup_scene}) @binding(3) var<uniform> ssaoParams: SSAOUniforms;

fn rand(co: vec2f) -> f32 {
    return fract(sin(dot(co ,vec2f(12.9898,78.233))) * 43758.5453);
}

fn unprojectUV(uv: vec2f, ndc_z: f32) -> vec3f {
    let ndcPos = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, ndc_z, 1.0);
    var viewPos = camera.inv_proj_mat * ndcPos;
    return viewPos.xyz / viewPos.w;
}

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) f32 {
    if (ssaoParams.enabled < 0.5) {
        return 1.0;
    }
    
    let fragCoord_i2 = vec2i(fragCoord.xy);
    let dims = textureDimensions(hizTexture, 0);
    let uv = fragCoord.xy / vec2f(dims);
    
    // Sample exact depth
    let center_ndc_z = textureLoad(hizTexture, fragCoord_i2, 0).g;
    
    // Background clear check (Reverse Z: sky is 0.0)
    if (center_ndc_z <= 0.000001) {
        return 1.0;
    }

    let view_pos = unprojectUV(uv, center_ndc_z);
    
    let normalColor = textureLoad(normalTex, fragCoord_i2, 0);
    let normal_world = normalize(normalColor.xyz);
    let view_normal = normalize((camera.view_mat * vec4f(normal_world, 0.0)).xyz);

    var noiseVec = normalize(vec3f(
        rand(fragCoord.xy * 1.32) * 2.0 - 1.0,
        rand(fragCoord.xy * 5.12) * 2.0 - 1.0,
        0.0
    ));

    var tangent = noiseVec - view_normal * dot(noiseVec, view_normal);
    if (length(tangent) < 0.001) {
        tangent = vec3f(1.0, 0.0, 0.0);
        tangent = tangent - view_normal * dot(tangent, view_normal);
    }
    tangent = normalize(tangent);
    
    let bitangent = cross(view_normal, tangent);
    let TBN = mat3x3f(tangent, bitangent, view_normal);

    let max_mip = u32(textureNumLevels(hizTexture)) - 1u;
    var occlusion = 0.0;
    
    for (var i = 0u; i < 64u; i += 1u) {
        let sample_view = TBN * ssaoParams.kernel[i].xyz;
        let sample_pos_view = view_pos + sample_view * ssaoParams.radius;

        var offset = vec4f(sample_pos_view, 1.0);
        offset = camera.proj_mat * offset;
        offset = offset / offset.w; 
        
        let sample_uv = vec2f(offset.x * 0.5 + 0.5, -offset.y * 0.5 + 0.5);
        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            
            // --- Hi-Z SAO Optimization ---
            let offset_pixels = distance(uv * vec2f(dims), sample_uv * vec2f(dims));
            
            // Fetch dynamically matching mip map level based on distance from center pixel
            let mipLevel = u32(clamp(log2(max(offset_pixels, 1.0) * 0.5), 0.0, f32(max_mip)));
            
            let mip_dims = textureDimensions(hizTexture, mipLevel);
            let sample_coord = vec2i(sample_uv * vec2f(mip_dims));
            
            // Read foreground (near) bounds in this mip
            let sample_ndc_z = textureLoad(hizTexture, clamp(sample_coord, vec2i(0), vec2i(mip_dims)-1), mipLevel).g;
            
            let sampleDepthView = unprojectUV(sample_uv, sample_ndc_z);
            
            let rangeCheck = smoothstep(0.0, 1.0, ssaoParams.radius / abs(view_pos.z - sampleDepthView.z));
            if (sampleDepthView.z >= sample_pos_view.z + ssaoParams.bias) {
                occlusion += rangeCheck;
            }
        }
    }

    occlusion = 1.0 - (occlusion / 64.0);
    occlusion = pow(occlusion, ssaoParams.power);

    return occlusion;
}
