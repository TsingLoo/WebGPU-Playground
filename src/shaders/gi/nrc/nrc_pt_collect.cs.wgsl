// ============================
// NRC Path Tracer Collect Training Data
// ============================
// Runs after the 5-bounce path tracing is complete.
// Reads the full accumulated radiance of the training paths, computes the 
// target incoming radiance from the second vertex, and stores the (features, target)
// pair into the NRC training buffer.

@group(0) @binding(0) var<uniform> nrc: NRCUniforms;
@group(0) @binding(1) var<storage, read> pt_nrc_train_data: array<NRCWavefrontTrainData>;
@group(0) @binding(2) var<storage, read> accum_buffer: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> sampleCounter: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> trainingSamples: array<f32>;

const MAX_TRAINING_SAMPLES: u32 = ${nrcMaxTrainingSamples}u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    
    // We only process up to the number of allocated training paths this frame
    let numSamples = atomicLoad(&sampleCounter);
    if (index >= numSamples || index >= MAX_TRAINING_SAMPLES) {
        return;
    }

    let data = pt_nrc_train_data[index];
    
    // Safety check: skip disabled or invalid data entries
    if (data.is_active == 0u) {
        return;
    }

    // `accum_buffer` contains the full integrated radiance for this pixel up to max_bounces.
    let final_radiance = accum_buffer[data.pixel_id].xyz;
    
    // The target radiance is the radiance from the 2nd vertex onwards.
    // It's calculated by subtracting the radiance accumulated BEFORE the 2nd vertex (primary radiance + bounce 0 direct light).
    // And divided by the path throughput from the camera to the 2nd vertex.
    let tail_radiance = final_radiance - data.primary_radiance;
    
    var target_radiance = tail_radiance / max(data.throughput, vec3f(1e-6));
    
    // Prevent bad samples (NaNs / negatives) from poisoning the network
    if (target_radiance.x != target_radiance.x || target_radiance.y != target_radiance.y || target_radiance.z != target_radiance.z || any(target_radiance < vec3f(0.0))) {
        target_radiance = vec3f(0.0);
    }
    
    // Tone map target to [0,1] so sigmoid output can represent it
    // Using simple Reinhard: x / (x + 1)
    target_radiance = target_radiance / (target_radiance + vec3f(1.0));

    // ---- Write to training buffer ----
    let baseOffset = index * NRC_SAMPLE_STRIDE;

    // Input features [0..14]
    for (var i = 0u; i < 15u; i++) {
        trainingSamples[baseOffset + i] = data.features[i];
    }

    // Target radiance [15..17]
    trainingSamples[baseOffset + 15u] = target_radiance.x;
    trainingSamples[baseOffset + 16u] = target_radiance.y;
    trainingSamples[baseOffset + 17u] = target_radiance.z;

    // Sample weight [18]
    trainingSamples[baseOffset + 18u] = 1.0;

    // Pad [19]
    trainingSamples[baseOffset + 19u] = 0.0;
}
