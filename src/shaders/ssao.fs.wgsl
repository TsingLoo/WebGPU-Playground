@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var positionTex: texture_2d<f32>;
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

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) f32 {
    if (ssaoParams.enabled < 0.5) {
        return 1.0;
    }
    
    let fragCoord_i2 = vec2i(fragCoord.xy);
    let dims = textureDimensions(positionTex);
    
    let posColor = textureLoad(positionTex, fragCoord_i2, 0);
    let normalColor = textureLoad(normalTex, fragCoord_i2, 0);
    
    if (posColor.w < 0.5 && length(posColor.xyz) < 0.001) {
        return 1.0;
    }

    let pos_world = posColor.xyz;
    let normal_world = normalize(normalColor.xyz);

    let view_pos = (camera.view_mat * vec4f(pos_world, 1.0)).xyz;
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

    var occlusion = 0.0;
    for (var i = 0u; i < 64u; i += 1u) {
        let sample_view = TBN * ssaoParams.kernel[i].xyz;
        let sample_pos_view = view_pos + sample_view * ssaoParams.radius;

        var offset = vec4f(sample_pos_view, 1.0);
        offset = camera.proj_mat * offset;
        offset = offset / offset.w; 
        
        let uv = vec2f(offset.x * 0.5 + 0.5, -offset.y * 0.5 + 0.5);
        let sample_coord = vec2i(uv * vec2f(dims));
        
        if (sample_coord.x >= 0 && sample_coord.y >= 0 && sample_coord.x < i32(dims.x) && sample_coord.y < i32(dims.y)) {
            let sampleDepthWorld = textureLoad(positionTex, sample_coord, 0).xyz;
            if (length(sampleDepthWorld) > 0.001) {
                let sampleDepthView = (camera.view_mat * vec4f(sampleDepthWorld, 1.0)).xyz;
                let rangeCheck = smoothstep(0.0, 1.0, ssaoParams.radius / abs(view_pos.z - sampleDepthView.z));
                if (sampleDepthView.z >= sample_pos_view.z + ssaoParams.bias) {
                    occlusion += rangeCheck;
                }
            }
        }
    }

    occlusion = 1.0 - (occlusion / 64.0);
    occlusion = pow(occlusion, ssaoParams.power);

    return occlusion;
}
