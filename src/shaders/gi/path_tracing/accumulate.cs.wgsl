// accumulate.cs.wgsl
// Wavefront Path Tracing — Final Compute Pass: Frame Accumulation
// Adds this frame's radiance to the running sum stored in accum_texture.
// The accum_texture stores the SUM of all samples (not average).
// The tonemap divides by sample_count to get the mean.
//
// accum_buffer: temporary scratch for the current frame's radiance (set by shade/shadow)
// accum_texture: PERSISTENT sum across all samples (read-write, write-only storage not ideal)
//
// IMPORTANT: accum_texture is CLEARED by resetAccumulation() which zeros the accum_buffer,
// and sample_count is reset to 0.

@group(0) @binding(0) var<uniform>             pt:              PTUniforms;
@group(0) @binding(1) var<storage, read_write>  accum_buffer:    array<vec4f>;
@group(0) @binding(2) var<storage, read_write>  sample_sum_buf:  array<vec4f>;  // persistent sum across frames

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = pt.width;
    let render_height = pt.height;
    if (gid.x >= render_width || gid.y >= render_height) { return; }

    let pixel_id = gid.y * render_width + gid.x;

    // Add this frame's sample to the running sum
    let current_sample  = accum_buffer[pixel_id].xyz;
    let running_sum     = sample_sum_buf[pixel_id].xyz;
    let new_sum         = running_sum + current_sample;

    sample_sum_buf[pixel_id] = vec4f(new_sum, 0.0);

    // Clear per-frame scratch buffer
    accum_buffer[pixel_id] = vec4f(0.0);
}
