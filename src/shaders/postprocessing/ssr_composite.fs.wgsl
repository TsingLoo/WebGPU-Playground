@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var sceneColorTexture: texture_2d<f32>;
@group(1) @binding(1) var ssrHitTexture: texture_2d<f32>;
@group(1) @binding(2) var albedoTexture: texture_2d<f32>;
@group(1) @binding(3) var specularTexture: texture_2d<f32>;
@group(1) @binding(4) var normalTexture: texture_2d<f32>;
@group(1) @binding(5) var depthTexture: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
    let uvCoords = vec2i(in.position.xy);
    let baseColor = textureLoad(sceneColorTexture, uvCoords, 0);
    
    let ssrHit = textureLoad(ssrHitTexture, uvCoords, 0);
    
    // If debug mode is enabled, visually mask out missed pixels vs hit pixels
    if (ssrHit.b > 0.5) {
        if (ssrHit.a <= 0.0) {
            return vec4f(0.0, 0.0, 0.0, 1.0); // Show black for ray misses to contrast against hits
        }
        return vec4f(ssrHit.x, ssrHit.y, 0.0, 1.0);
    }

    if (ssrHit.a <= 0.0) {
        return baseColor;
    }
    
    let depth = textureLoad(depthTexture, uvCoords, 0);
    if (depth <= 0.0) {
        return baseColor; // Sky
    }

    let albedo = textureLoad(albedoTexture, uvCoords, 0).rgb;
    let pbrParams = textureLoad(specularTexture, uvCoords, 0);
    let roughness = pbrParams.r;
    let metallic = pbrParams.g;
    
    let normal = normalize(textureLoad(normalTexture, uvCoords, 0).xyz);
    
    // Compute UV from framebuffer coords (y=0 at top) — NOT in.uv (y=0 at bottom)
    let screenDims = vec2f(textureDimensions(depthTexture));
    let uv = vec2f(uvCoords) / screenDims;

    let clipPos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let worldPosV = camera.inv_view_proj_mat * clipPos;
    let worldPos = worldPosV.xyz / worldPosV.w;
    
    let viewDir = normalize(camera.camera_pos.xyz - worldPos);
    
    // Approximate F0
    var F0 = vec3f(0.04);
    F0 = mix(F0, albedo, metallic);
    let cosTheta = max(dot(normal, viewDir), 0.0);
    let F = fresnelSchlick(cosTheta, F0);
    
    // Simple environment BRDF using Roughness
    // Approximate reflection factor: metallic materials reflect more colored light, dielectrics reflect 4% white
    let reflectionWeight = F * ssrHit.a * (1.0 - roughness);
    
    // Sample the scene color at the hit UV
    let hitUV = ssrHit.xy;
    let hitCoords = vec2i(hitUV * vec2f(textureDimensions(sceneColorTexture)));
    let reflectedColor = textureLoad(sceneColorTexture, hitCoords, 0).rgb;
    
    // Output composite
    let finalColor = baseColor.rgb + reflectedColor * reflectionWeight;
    return vec4f(finalColor, 1.0);
}
