// ray_gen.cs.wgsl
// Wavefront Path Tracing — Pass 1: Primary Ray Generation
// Supports spectral rendering: samples hero wavelengths and initializes spectral buffers.

@group(0) @binding(0) var<uniform>            camera:                  CameraUniforms;
@group(0) @binding(1) var<uniform>            pt:                      PTUniforms;
@group(0) @binding(2) var<storage, read_write> ray_buffer:              array<PTRay>;
@group(0) @binding(3) var<storage, read_write> spectral_wavelengths:    array<vec4f>;  // per-pixel wavelengths (nm)
@group(0) @binding(4) var<storage, read_write> spectral_pdfs:           array<vec4f>;  // per-pixel wavelength PDFs
@group(0) @binding(5) var<storage, read_write> spectral_throughput_buf: array<vec4f>;  // per-pixel spectral throughput

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = pt.width;
    let render_height = pt.height;
    if (gid.x >= render_width || gid.y >= render_height) { return; }

    let pixel_id = gid.y * render_width + gid.x;

    var rng = initRNG(pixel_id, pt.frame_index);
    let jitter = rand2(&rng) - vec2f(0.5);

    let uv  = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5) + jitter)
            / vec2f(f32(render_width), f32(render_height));
    let ndc = vec2f(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    let clip_near = vec4f(ndc.x, ndc.y, 1.0, 1.0);
    let clip_far  = vec4f(ndc.x, ndc.y, 0.0, 1.0);
    var world_near = camera.inv_view_proj_mat * clip_near;
    var world_far  = camera.inv_view_proj_mat * clip_far;
    world_near /= world_near.w;
    world_far  /= world_far.w;

    let origin    = camera.camera_pos.xyz;
    let direction = normalize(world_far.xyz - world_near.xyz);

    var ray: PTRay;
    ray.origin          = origin;
    ray.ior             = 1.0;
    ray.direction       = direction;
    ray.pixel_id        = pixel_id;
    ray.throughput      = vec3f(1.0);
    ray.bounce          = 0u;
    ray.ray_active      = 1u;
    ray.specular_bounce = 0u;
    ray._pad            = 0u;

    // ============================================================
    // Spectral Rendering: sample hero wavelengths
    // ============================================================
    if (pt.spectral_enabled == 1u) {
        let u_wavelength = rand(&rng);
        let lambdas = sampleHeroWavelengths(u_wavelength);
        let pdfs = wavelengthPDF();

        spectral_wavelengths[pixel_id]    = lambdas;
        spectral_pdfs[pixel_id]           = pdfs;
        spectral_throughput_buf[pixel_id] = vec4f(1.0);  // initialize to 1
    }

    ray.rng_state        = rng;
    ray_buffer[pixel_id] = ray;
}
