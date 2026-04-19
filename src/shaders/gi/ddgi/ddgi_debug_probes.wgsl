@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<uniform> ddgi: DDGIUniforms;
@group(1) @binding(1) var<storage, read> probeData: array<vec4f>;
@group(1) @binding(2) var irradianceAtlas: texture_2d<f32>;
@group(1) @binding(3) var ddgiSampler: sampler;
@group(1) @binding(4) var visibilityAtlas: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) normal: vec3f,
    @location(1) @interpolate(flat) probe_index: u32,
};

@vertex
fn vs_main(@builtin(instance_index) inst_idx: u32, @builtin(vertex_index) v_idx: u32) -> VertexOutput {
    var pos = array<vec3f, 24>(
        // Top pyramid
        vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0),
        vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(-1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0),
        vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(1.0, 0.0, 0.0),
        // Bottom pyramid
        vec3f(0.0, -1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, -1.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0),
        vec3f(0.0, -1.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(-1.0, 0.0, 0.0),
        vec3f(0.0, -1.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0)
    );
    
    let local_pos = pos[v_idx];
    let local_normal = normalize(local_pos);
    
    // Probe index
    let probe_idx = inst_idx;
    let grid_z = probe_idx / (u32(ddgi.grid_count.x) * u32(ddgi.grid_count.y));
    let grid_y = (probe_idx / u32(ddgi.grid_count.x)) % u32(ddgi.grid_count.y);
    let grid_x = probe_idx % u32(ddgi.grid_count.x);
    let gridIdx = vec3i(i32(grid_x), i32(grid_y), i32(grid_z));
    
    // Probe center
    let base_center = ddgiProbePosition(gridIdx, ddgi);
    let offset = probeData[probe_idx].xyz;
    let center = base_center + offset;
    
    // State: w == 1 if asleep, 0 if active. (0.0 offset usually means active)
    let state = probeData[probe_idx].w;
    
    // Size based on spacing to make it not too large
    let radius = min(ddgi.grid_spacing.x, min(ddgi.grid_spacing.y, ddgi.grid_spacing.z)) * 0.1;
    let world_pos = center + local_pos * radius * select(1.0, 0.3, state > 0.0); // Make sleeping probes smaller
    
    var out: VertexOutput;
    out.clip_pos = camera.view_proj_mat * vec4f(world_pos, 1.0);
    out.normal = local_normal;
    out.probe_index = probe_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let mode = i32(ddgi.ddgi_enabled.y + 0.5); // 1 = irradiance, 2 = visibility
    let octUV = octEncode(normalize(in.normal));
    
    if (mode == 2) {
        let texelCoord = ddgiVisibilityTexelCoord(i32(in.probe_index), octUV, ddgi);
        let dist = textureSampleLevel(visibilityAtlas, ddgiSampler, texelCoord, 0.0).rg;
        
        let max_dist = max(ddgi.grid_spacing.x, max(ddgi.grid_spacing.y, ddgi.grid_spacing.z)) * 4.0;
        let visVal = dist.x / max_dist;
        
        // Show mean distance in red, and mean square distance in green
        let color = vec3f(visVal, dist.y / (max_dist * max_dist), 0.0);
        return vec4f(color, 1.0);
    } else {
        let texelCoord = ddgiIrradianceTexelCoord(i32(in.probe_index), octUV, ddgi);
        let irradiance = textureSampleLevel(irradianceAtlas, ddgiSampler, texelCoord, 0.0).rgb;
        
        // Simple tonemapping for debug + Gamma
        let color = irradiance / (irradiance + vec3f(1.0));
        let srgbColor = pow(max(color, vec3f(0.0)), vec3f(1.0 / 2.2));
        
        return vec4f(srgbColor, 1.0);
    }
}
