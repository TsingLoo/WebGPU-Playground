@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct BoxUniforms {
    minPos: vec4f,
    maxPos: vec4f,
    color: vec4f,
};
@group(1) @binding(0) var<uniform> box: BoxUniforms;

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> @builtin(position) vec4f {
    var pos = array<vec3f, 24>(
        // Bottom loop
        vec3f(box.minPos.x, box.minPos.y, box.minPos.z), vec3f(box.maxPos.x, box.minPos.y, box.minPos.z),
        vec3f(box.maxPos.x, box.minPos.y, box.minPos.z), vec3f(box.maxPos.x, box.minPos.y, box.maxPos.z),
        vec3f(box.maxPos.x, box.minPos.y, box.maxPos.z), vec3f(box.minPos.x, box.minPos.y, box.maxPos.z),
        vec3f(box.minPos.x, box.minPos.y, box.maxPos.z), vec3f(box.minPos.x, box.minPos.y, box.minPos.z),
        
        // Top loop
        vec3f(box.minPos.x, box.maxPos.y, box.minPos.z), vec3f(box.maxPos.x, box.maxPos.y, box.minPos.z),
        vec3f(box.maxPos.x, box.maxPos.y, box.minPos.z), vec3f(box.maxPos.x, box.maxPos.y, box.maxPos.z),
        vec3f(box.maxPos.x, box.maxPos.y, box.maxPos.z), vec3f(box.minPos.x, box.maxPos.y, box.maxPos.z),
        vec3f(box.minPos.x, box.maxPos.y, box.maxPos.z), vec3f(box.minPos.x, box.maxPos.y, box.minPos.z),
        
        // Vertical pillars
        vec3f(box.minPos.x, box.minPos.y, box.minPos.z), vec3f(box.minPos.x, box.maxPos.y, box.minPos.z),
        vec3f(box.maxPos.x, box.minPos.y, box.minPos.z), vec3f(box.maxPos.x, box.maxPos.y, box.minPos.z),
        vec3f(box.maxPos.x, box.minPos.y, box.maxPos.z), vec3f(box.maxPos.x, box.maxPos.y, box.maxPos.z),
        vec3f(box.minPos.x, box.minPos.y, box.maxPos.z), vec3f(box.minPos.x, box.maxPos.y, box.maxPos.z)
    );
    
    let p = pos[v_idx];
    return camera.view_proj_mat * vec4f(p, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return box.color;
}
