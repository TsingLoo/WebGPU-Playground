// miss.cs.wgsl
// Wavefront Path Tracing — Pass 5: Environment Miss
// For PTRays that escaped the scene: sample HDR cubemap and accumulate.

@group(0) @binding(0) var<uniform>             pt:           PTUniforms;
@group(0) @binding(1) var<storage, read_write>  ray_buffer:   array<PTRay>;
@group(0) @binding(2) var<storage, read_write>  hit_buffer:   array<HitRecord>;
@group(0) @binding(3) var<storage, read_write>  accum_buffer: array<vec4f>;
@group(0) @binding(4) var                       env_cubemap:  texture_cube<f32>;
@group(0) @binding(5) var                       env_sampler:  sampler;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let ray = ray_buffer[pixel_id];
    let hit = hit_buffer[pixel_id];

    // Skip: terminated by Russian Roulette/bounce limit, not a miss
    if (ray.ray_active == 0u && hit.did_hit != 0u) { return; }
    // Skip: still has a pending hit (processed in next bounce)
    if (ray.ray_active == 1u && hit.did_hit == 1u) { return; }

    let env_color    = textureSampleLevel(env_cubemap, env_sampler, ray.direction, 0.0).xyz;
    let contribution = clamp(ray.throughput * env_color, vec3f(0.0), vec3f(pt.clamp_radiance));

    let prev = accum_buffer[pixel_id];
    accum_buffer[pixel_id] = vec4f(prev.xyz + contribution, prev.w);

    ray_buffer[pixel_id].ray_active = 0u;
}
