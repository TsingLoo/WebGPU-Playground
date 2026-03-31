// Radiance Cascades Border Copy
// Copies interior edge texels to the 1px border for seamless bilinear sampling
// in the octahedral atlas.

@group(0) @binding(0) var sourceAtlas: texture_2d<f32>;
@group(0) @binding(1) var destAtlas: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: vec4u; // (texelDim, texelDimWithBorder, probesPerRow, totalProbes)

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let texelDim = params.x;
    let texelDimBorder = params.y;
    let probesPerRow = params.z;
    let totalProbes = params.w;

    let borderTexelsPerProbe = 4u * texelDim + 4u;
    let totalBorderTexels = borderTexelsPerProbe * totalProbes;

    let globalIdx = gid.x;
    if (globalIdx >= totalBorderTexels) { return; }

    let probeIdx = globalIdx / borderTexelsPerProbe;
    let borderIdx = globalIdx % borderTexelsPerProbe;

    let probeRow = probeIdx / probesPerRow;
    let probeCol = probeIdx % probesPerRow;

    let probeOriginX = i32(probeCol * texelDimBorder);
    let probeOriginY = i32(probeRow * texelDimBorder);

    var borderX: i32;
    var borderY: i32;
    var srcX: i32;
    var srcY: i32;

    let td = i32(texelDim);
    let tdb = i32(texelDimBorder);

    if (borderIdx < texelDim + 2u) {
        borderX = i32(borderIdx);
        borderY = 0;
        srcX = clamp(tdb - 1 - borderX, 1, td);
        srcY = 1;
    } else if (borderIdx < 2u * (texelDim + 2u)) {
        let localIdx = borderIdx - (texelDim + 2u);
        borderX = i32(localIdx);
        borderY = td + 1;
        srcX = clamp(tdb - 1 - borderX, 1, td);
        srcY = td;
    } else if (borderIdx < 2u * (texelDim + 2u) + texelDim) {
        let localIdx = borderIdx - 2u * (texelDim + 2u);
        borderX = 0;
        borderY = i32(localIdx) + 1;
        srcX = 1;
        srcY = clamp(tdb - 1 - borderY, 1, td);
    } else {
        let localIdx = borderIdx - 2u * (texelDim + 2u) - texelDim;
        borderX = td + 1;
        borderY = i32(localIdx) + 1;
        srcX = td;
        srcY = clamp(tdb - 1 - borderY, 1, td);
    }

    let dstCoord = vec2i(probeOriginX + borderX, probeOriginY + borderY);
    let srcCoord = vec2i(probeOriginX + srcX, probeOriginY + srcY);

    let srcColor = textureLoad(sourceAtlas, srcCoord, 0);
    textureStore(destAtlas, dstCoord, srcColor);
}
