// pt_tonemap.wgsl
// Wavefront Path Tracing — Fullscreen Blit with Temporal Accumulation + Tonemapping
// Reads the persistent sample sum buffer, divides by sample_count, tonemaps, outputs to canvas.

// ---- Vertex Shader ----
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );
    return vec4f(positions[vi], 0.0, 1.0);
}

// ---- Fragment Shader ----
@group(0) @binding(0) var<uniform>            pt:             PTUniforms;
@group(0) @binding(1) var<storage, read>       sample_sum_buf: array<vec4f>;  // sum of all PT samples

// ACES filmic tone mapping
fn ACESFilm(x: vec3f) -> vec3f {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let render_width  = pt.width;
    let render_height = pt.height;

    // Recover canvas dimensions: pt.width/height are the render resolution (already scaled)
    let canvas_width  = f32(pt.width)  / pt.pixel_scale;
    let canvas_height = f32(pt.height) / pt.pixel_scale;

    // Map canvas pixel to render-resolution pixel
    let uv = frag_pos.xy / vec2f(canvas_width, canvas_height);
    let render_px = vec2u(uv * vec2f(f32(render_width), f32(render_height)));
    let safe_u = clamp(render_px.x, 0u, render_width - 1u);
    let safe_v = clamp(render_px.y, 0u, render_height - 1u);
    let pixel_id = safe_v * render_width + safe_u;

    // Read accumulated sum and normalize by sample count
    let radiance_sum = sample_sum_buf[pixel_id].xyz;
    let n = f32(max(pt.sample_count, 1u));
    let avg_radiance = radiance_sum / n;

    // ACES tonemapping + gamma
    let mapped = ACESFilm(avg_radiance);
    let gamma_corrected = pow(max(mapped, vec3f(0.0)), vec3f(1.0 / 2.2));

    return vec4f(gamma_corrected, 1.0);
}
