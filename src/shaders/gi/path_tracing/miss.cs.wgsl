// miss.cs.wgsl
// Wavefront Path Tracing — Pass 4: Environment Miss
// For PTRays that missed geometry at this bounce: sample HDR cubemap and accumulate.
// Runs INSIDE the bounce loop, after shade + shadow_test.
// Supports spectral rendering mode.

@group(0) @binding(0) var<uniform>             pt:                      PTUniforms;
@group(0) @binding(1) var<storage, read_write>  ray_buffer:              array<PTRay>;
@group(0) @binding(2) var<storage, read_write>  hit_buffer:              array<HitRecord>;
@group(0) @binding(3) var<storage, read_write>  accum_buffer:            array<vec4f>;
@group(0) @binding(4) var                       env_cubemap:             texture_cube<f32>;
@group(0) @binding(5) var                       env_sampler:             sampler;
@group(0) @binding(6) var<storage, read>        spectral_wavelengths:    array<vec4f>;
@group(0) @binding(7) var<storage, read>        spectral_throughput_buf: array<vec4f>;
@group(0) @binding(8) var<storage, read_write>  spectral_accum_buf:      array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = pt.width;
    let render_height = pt.height;
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let ray = ray_buffer[pixel_id];
    let hit = hit_buffer[pixel_id];

    // A ray "missed" when:
    //  - ray_active == 1 (was active going into this bounce)
    //  - did_hit == 0 (intersect found no geometry)
    // All other cases:
    //  - ray_active==0: already terminated (RR, bounce limit, or previously processed miss)
    //  - did_hit==1: hit geometry, shade pass processed it

    if (ray.ray_active == 0u || hit.did_hit != 0u) { return; }

    let env_color = textureSampleLevel(env_cubemap, env_sampler, ray.direction, 0.0).xyz;

    if (pt.spectral_enabled == 1u) {
        // Spectral mode: convert env color to spectral, multiply by spectral throughput
        let lambdas         = spectral_wavelengths[pixel_id];
        let spec_throughput = spectral_throughput_buf[pixel_id];
        let spec_env        = rgbToIlluminantSpectrum(env_color, lambdas);
        let contribution    = min(spec_throughput * spec_env, vec4f(pt.clamp_radiance));

        let prev = spectral_accum_buf[pixel_id];
        spectral_accum_buf[pixel_id] = prev + contribution;
    } else {
        // RGB mode: original behavior
        let contribution = clamp(ray.throughput * env_color, vec3f(0.0), vec3f(pt.clamp_radiance));

        let prev = accum_buffer[pixel_id];
        accum_buffer[pixel_id] = vec4f(prev.xyz + contribution, prev.w);
    }

    // Terminate ray
    ray_buffer[pixel_id].ray_active = 0u;
}
