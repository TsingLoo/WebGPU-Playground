struct SSRUniforms {
    enabled: f32,
    maxDistance: f32,
    resolutionScale: f32,
    maxSteps: f32,
    thickness: f32,
    debugMode: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var hizTexture: texture_2d<f32>;
@group(1) @binding(1) var normalTexture: texture_2d<f32>;
@group(1) @binding(2) var specularTexture: texture_2d<f32>; // R=roughness, G=metallic
@group(1) @binding(3) var depthTexture: texture_depth_2d;
@group(1) @binding(4) var<uniform> ssrUniforms: SSRUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

fn reconstructPosition(uv: vec2f, rawDepth: f32) -> vec3f {
    let ndc = vec4f(uv * 2.0 - 1.0, rawDepth, 1.0);
    let wp = camera.inv_view_proj_mat * vec4f(ndc.x, -ndc.y, ndc.z, ndc.w);
    return wp.xyz / wp.w;
}

fn getViewPosition(uv: vec2f, rawDepth: f32) -> vec3f {
    let clipPos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, rawDepth, 1.0);
    let viewPos = camera.inv_proj_mat * clipPos;
    return viewPos.xyz / viewPos.w;
}

// Project view space point to texture space (0..1)
fn projectViewToScreen(viewPos: vec3f) -> vec3f {
    let clipPos = camera.proj_mat * vec4f(viewPos, 1.0);
    let ndc = clipPos.xyz / clipPos.w;
    // return uv and depth
    return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
    if (ssrUniforms.enabled == 0.0) {
        return vec4f(0.0);
    }

    let depth = textureLoad(depthTexture, vec2i(in.position.xy), 0);
    if (depth <= 0.0) { // ReverseZ sky
        return vec4f(0.0);
    }

    let pbrParams = textureLoad(specularTexture, vec2i(in.position.xy), 0);
    let roughness = pbrParams.r;
    let metallic = pbrParams.g;

    // Fast reject highly rough surfaces
    if (roughness > 0.6) {
        return vec4f(0.0);
    }

    // Compute UV from framebuffer coordinates (y=0 at top, like textureLoad convention)
    // NOT from in.uv which has y=0 at bottom (NDC convention from the blit vertex shader)
    let screenSize = vec2f(textureDimensions(depthTexture));
    let uv = in.position.xy / screenSize;

    let normal = normalize(textureLoad(normalTexture, vec2i(in.position.xy), 0).xyz);
    let viewPos = getViewPosition(uv, depth);
    let viewDir = normalize(viewPos);
    // View space normal
    let viewNormal = normalize((camera.view_mat * vec4f(normal, 0.0)).xyz);
    
    // Reflection vector in view space
    let reflDirView = reflect(viewDir, viewNormal);
    
    if (reflDirView.z > 0.0) {
        // Reflecting towards camera (mostly out of bounds)
        // return vec4f(0.0);
    }

    // Ray marching in View Space / Screen Space
    let maxDistance = ssrUniforms.maxDistance;
    var endViewPos = viewPos + reflDirView * maxDistance;
    
    // Prevent ray from going behind the near plane.
    // In WebGPU right-handed view space, camera looks down -Z.
    // If endViewPos.z > -0.1, it crossed the near plane, which would cause clipPos.w <= 0 and flip the projected NDC.
    let nearPlaneZ = -0.1;
    if (endViewPos.z > nearPlaneZ) {
        if (reflDirView.z > 0.0001) {
            let t = (nearPlaneZ - viewPos.z) / reflDirView.z;
            endViewPos = viewPos + reflDirView * t;
        } else {
            return vec4f(0.0);
        }
    }
    
    // Bias ray origin along view-space normal to prevent self-intersection.
    // Without this, the ray immediately hits the surface it started from
    // because the interpolated screen-space Z diverges from the actual surface depth.
    let biasedViewPos = viewPos + viewNormal * 0.1;
    let startFrag = projectViewToScreen(biasedViewPos);
    let endFrag = projectViewToScreen(endViewPos);
    
    let delta = endFrag - startFrag;
    
    // Convert to pixel space
    let texSize = vec2f(textureDimensions(depthTexture));
    let texDelta = delta.xy * texSize;
    if (abs(texDelta.x) < 0.001 && abs(texDelta.y) < 0.001) {
        return vec4f(0.0);
    }
    
    let useX = abs(texDelta.x) >= abs(texDelta.y);
    var deltaStep: f32;
    if (useX) {
        deltaStep = abs(1.0 / texDelta.x);
    } else {
        deltaStep = abs(1.0 / texDelta.y);
    }
    
    var rayP = startFrag;
    var advance = deltaStep;
    var hit = false;
    var hitUV = vec2f(0.0);
    
    // Basic linear raymarching as fallback if Hi-Z is not fully structured
    // A true Hi-Z uses quadtrees, here we do a fast linear search through the top mip
    // with fixed thickness check.
    
    // Skip a few initial steps to further avoid self-intersection on large flat surfaces
    var currentT = advance * 3.0;
    let maxSteps = i32(ssrUniforms.maxSteps);
    let thickness = ssrUniforms.thickness;
    
    for (var i = 0; i < maxSteps; i++) {
        if (currentT > 1.0) {
            break;
        }
        
        rayP = startFrag + delta * currentT;
        if (rayP.x < 0.0 || rayP.x > 1.0 || rayP.y < 0.0 || rayP.y > 1.0 || rayP.z < 0.0) {
             break;
        }
        
        let sampleUV = rayP.xy;
        let hizSize = vec2f(textureDimensions(hizTexture, 0));
        let sampleDepth = textureLoad(hizTexture, vec2i(sampleUV * hizSize), 0).r; // .r is far depth
        
        // Reverse-Z: greater value means closer to camera
        // So ray depth (rayP.z) should be < sampleDepth if ray is BEHIND the geometry
        if (rayP.z < sampleDepth) {
            let depthDiff = sampleDepth - rayP.z;
            // Check thickness 
            let viewSamplePos = getViewPosition(sampleUV, sampleDepth);
            let viewRayPos = getViewPosition(sampleUV, rayP.z);
            let distDiff = abs(viewSamplePos.z - viewRayPos.z);
            
            if (distDiff < thickness) {
                hit = true;
                hitUV = sampleUV;
                break;
            }
        }
        currentT += advance;
    }

    if (hit) {
        // Return Hit UV in R and G, debugMode in B, and a valid mask in A
        let edgeFade = smoothstep(0.0, 0.1, hitUV.x) * smoothstep(1.0, 0.9, hitUV.x) *
                       smoothstep(0.0, 0.1, hitUV.y) * smoothstep(1.0, 0.9, hitUV.y);
        
        return vec4f(hitUV.x, hitUV.y, ssrUniforms.debugMode, edgeFade);
    }

    return vec4f(0.0, 0.0, ssrUniforms.debugMode, 0.0);
}
