// accumulate.cs.wgsl
// Wavefront Path Tracing — Final Compute Pass: Frame Accumulation
// Adds this frame's radiance to the running sum stored in sample_sum_buf.
// The sample_sum_buf stores the SUM of all samples (not average).
// The tonemap divides by sample_count to get the mean.
//
// accum_buffer: temporary scratch for the current frame's radiance (set by shade/shadow)
// sample_sum_buf: PERSISTENT sum across all samples (read-write)
//
// In spectral mode, reads from spectral_accum_buf, converts to linear sRGB
// via spectrumToRGB (Gram-corrected MC estimator), and adds to sample_sum_buf.
// This way tonemap is unchanged.
//
// IMPORTANT: sample_sum_buf is CLEARED by resetAccumulation() which zeros the buffer,
// and sample_count is reset to 0.

@group(0) @binding(0) var<uniform>             pt:                   PTUniforms;
@group(0) @binding(1) var<storage, read_write>  accum_buffer:         array<vec4f>;
@group(0) @binding(2) var<storage, read_write>  sample_sum_buf:       array<vec4f>;  // persistent sum across frames
@group(0) @binding(3) var<storage, read>        spectral_wavelengths: array<vec4f>;
@group(0) @binding(4) var<storage, read>        spectral_pdfs:        array<vec4f>;
@group(0) @binding(5) var<storage, read_write>  spectral_accum_buf:   array<vec4f>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = pt.width;
    let render_height = pt.height;
    if (gid.x >= render_width || gid.y >= render_height) { return; }

    let pixel_id = gid.y * render_width + gid.x;

    if (pt.spectral_enabled == 1u) {
        // ============================================================
        // Spectral mode: spectral_accum_buf has 4 spectral samples.
        // Convert directly to sRGB using Gram-corrected MC estimator.
        // Also add any RGB contributions from shadow_test (NEE Li).
        // ============================================================
        let spectral_sample = spectral_accum_buf[pixel_id];
        let lambdas         = spectral_wavelengths[pixel_id];

        // Direct spectral→RGB with Gram matrix correction (guaranteed round-trip)
        var rgb = spectrumToRGB(spectral_sample, lambdas);

        // Clamp negative values (can happen from out-of-gamut spectral conversions)
        rgb = max(rgb, vec3f(0.0));

        // Also add any RGB-based NEE contribution from shadow_test
        let rgb_nee = accum_buffer[pixel_id].xyz;
        rgb += rgb_nee;

        // Guard against NaN/Inf
        if (any(rgb != rgb) || any(rgb > vec3f(1e6))) {
            rgb = vec3f(0.0);
        }

        // Add to persistent sum
        let running_sum = sample_sum_buf[pixel_id].xyz;
        sample_sum_buf[pixel_id] = vec4f(running_sum + rgb, 0.0);

        // Clear per-frame scratch buffers
        spectral_accum_buf[pixel_id] = vec4f(0.0);
        accum_buffer[pixel_id] = vec4f(0.0);
    } else {
        // ============================================================
        // RGB mode: original behavior
        // ============================================================
        let current_sample  = accum_buffer[pixel_id].xyz;
        let running_sum     = sample_sum_buf[pixel_id].xyz;
        let new_sum         = running_sum + current_sample;

        sample_sum_buf[pixel_id] = vec4f(new_sum, 0.0);

        // Clear per-frame scratch buffer
        accum_buffer[pixel_id] = vec4f(0.0);
    }
}
