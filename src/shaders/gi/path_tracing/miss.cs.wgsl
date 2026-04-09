// miss.cs.wgsl
// Wavefront Path Tracing — Pass 4: Environment Miss
// For PTRays that missed geometry at this bounce: sample HDR cubemap and accumulate.
// Runs INSIDE the bounce loop, after shade + shadow_test.

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

    // We want to add environment light for rays that MISSED geometry at this bounce.
    // After the shade pass:
    //  - If the ray missed: shade set ray_active=0, hit.did_hit=0
    //  - If the ray hit and was shaded: shade spawned a new bounce ray, ray_active=1, hit.did_hit=1
    //  - If the ray was terminated (RR/bounce limit): ray_active=0, hit.did_hit=1
    //  - If the ray was already inactive from a previous bounce: ray_active=0

    // Only process rays that missed geometry (did_hit == 0) and have non-zero throughput
    if (hit.did_hit != 0u) { return; }

    // If throughput is zero, no point sampling env
    if (max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z)) < 0.001) { return; }

    let env_color    = textureSampleLevel(env_cubemap, env_sampler, ray.direction, 0.0).xyz;
    let contribution = clamp(ray.throughput * env_color, vec3f(0.0), vec3f(pt.clamp_radiance));

    let prev = accum_buffer[pixel_id];
    accum_buffer[pixel_id] = vec4f(prev.xyz + contribution, prev.w);

    // Mark ray as done
    ray_buffer[pixel_id].ray_active = 0u;
    // Mark hit as processed so we don't re-add env on the next bounce iteration
    hit_buffer[pixel_id].did_hit = 1u;
}
