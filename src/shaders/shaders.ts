// Shader loader — preprocesses all shaders with common code and constants

import commonRaw from './core/common.wgsl?raw';
import standardMaterialRaw from './materials/standard_material.wgsl?raw';
import unlitMaterialRaw from './materials/unlit_material.wgsl?raw';
import giEvaluationRaw from './gi/gi_evaluation.wgsl?raw';
import lightingCompositeRaw from './core/lighting_composite.wgsl?raw';

import standardVertRaw from './core/standard.vs.wgsl?raw';

import geometryFragRaw from './core/geometry.fs.wgsl?raw';

import forwardPlusFragRaw from './rendering_path/forward_plus/forward_plus.fs.wgsl?raw';

import fullscreenBlitVertRaw from './core/blit/fullscreen_blit.vs.wgsl?raw';
import fullscreenBlitFragRaw from './core/blit/fullscreen_blit.fs.wgsl?raw';

import clusteredDeferredComputeSrcRaw from './rendering_path/clustered_deferred/clusteredDeferred.cs.wgsl?raw';

import moveLightsComputeRaw from './core/clustering/move_lights.cs.wgsl?raw';
import clusteringComputeRaw from './core/clustering/clustering.cs.wgsl?raw';

import zPrepassFragRaw from './core/zPrepass.fs.wgsl?raw';
import debugBoxRaw from './core/debug_box.wgsl?raw';

// IBL shaders
import generateCubemapRaw from './ibl/generate_cubemap.cs.wgsl?raw';
import irradianceConvolutionRaw from './ibl/irradiance_convolution.cs.wgsl?raw';
import prefilterEnvmapRaw from './ibl/prefilter_envmap.cs.wgsl?raw';
import brdfLutRaw from './ibl/brdf_lut.cs.wgsl?raw';
import equirectangularToCubemapRaw from './ibl/equirectangular_to_cubemap.cs.wgsl?raw';

// DDGI
import ddgiProbeTraceRaw from './gi/ddgi/ddgi_probe_trace.cs.wgsl?raw';
import ddgiIrradianceUpdateRaw from './gi/ddgi/ddgi_irradiance_update.cs.wgsl?raw';
import ddgiVisibilityUpdateRaw from './gi/ddgi/ddgi_visibility_update.cs.wgsl?raw';
import ddgiBorderUpdateRaw from './gi/ddgi/ddgi_border_update.cs.wgsl?raw';

// Radiance Cascades
import rcTraceRaw from './gi/radiance_cascades/rc_trace.cs.wgsl?raw';
import rcBorderRaw from './gi/radiance_cascades/rc_border.cs.wgsl?raw';

// NRC shaders
import nrcCommonRaw from './gi/nrc/nrc_common.wgsl?raw';
import nrcScatterTrainingRaw from './gi/nrc/nrc_scatter_training.cs.wgsl?raw';
import nrcTrainRaw from './gi/nrc/nrc_train.cs.wgsl?raw';
import nrcInferenceRaw from './gi/nrc/nrc_inference.cs.wgsl?raw';

// Surfel shaders
import surfelCommonRaw from './gi/surfel/surfel_common.wgsl?raw';
import bvhRaw from './gi/surfel/bvh.wgsl?raw';
import surfelLifecycleRaw from './gi/surfel/surfel_lifecycle.cs.wgsl?raw';
import surfelGridRaw from './gi/surfel/surfel_grid.cs.wgsl?raw';
import surfelIntegratorRaw from './gi/surfel/surfel_integrator.cs.wgsl?raw';
import surfelResolveRaw from './gi/surfel/surfel_resolve.cs.wgsl?raw';

// Shadow shaders
import shadowVertRaw from './shadows/shadow.vs.wgsl?raw';
import shadowFragRaw from './shadows/shadow.fs.wgsl?raw';

// VSM shaders
import vsmClearRaw from './shadows/vsm_clear.cs.wgsl?raw';
import vsmMarkPagesRaw from './shadows/vsm_mark_pages.cs.wgsl?raw';
import vsmAllocatePagesRaw from './shadows/vsm_allocate_pages.cs.wgsl?raw';

// Skybox shaders
import skyboxVertRaw from './environment/skybox.vs.wgsl?raw';
import skyboxFragRaw from './environment/skybox.fs.wgsl?raw';

// SSAO shaders
import ssaoFragRaw from './postprocessing/ssao.fs.wgsl?raw';
import ssaoBlurFragRaw from './postprocessing/ssao_blur.fs.wgsl?raw';

// Volumetric Lighting shaders
import volumetricLightingVertRaw from './environment/volumetric_lighting.vs.wgsl?raw';
import volumetricLightingFragRaw from './environment/volumetric_lighting.fs.wgsl?raw';
import volumetricCompositeFragRaw from './environment/volumetric_composite.fs.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

const numClustersXConfig = 16;
const numClustersYConfig = 16;
const numClusterZConfig = 16;
const numTotalClustersConfig = numClustersXConfig * numClustersYConfig * numClusterZConfig;

export const constants = {
    numClustersX: numClustersXConfig,
    numClustersY: numClustersYConfig,
    numClustersZ: numClusterZConfig,
    numTotalClustersConfig: numTotalClustersConfig,

    averageLightsPerCluster: 1024,
    maxLightsPerCluster: 1024,

    ambientR: 0.05,
    ambientG: 0.05,
    ambientB: 0.05,

    bindGroup_scene: 0,
    bindGroup_model: 1,
    bindGroup_material: 2,

    moveLightsWorkgroupSize: 128,

    lightRadius: 2,

    // Radiance Cascades
    rcProbeGridX: 64, // 28m / 64 = 0.43m
    rcProbeGridY: 48, // 20m / 48 = 0.41m
    rcProbeGridZ: 32, // 14m / 32 = 0.43m
    rcIrradianceTexels: 8,

    // DDGI
    ddgiProbeGridX: 24,
    ddgiProbeGridY: 16,
    ddgiProbeGridZ: 24,
    ddgiRaysPerProbe: 256,
    ddgiIrradianceTexels: 8,
    ddgiVisibilityTexels: 16,

    // VSM
    vsmPageSize: 128,
    vsmPhysAtlasSize: 4096,
    vsmPhysPagesPerAxis: 32,
    vsmNumClipmapLevels: 6,
    vsmPagesPerLevelAxis: 128,

    // NRC
    nrcMaxTrainingSamples: 4096,
};

// =================================

function evalShaderRaw(raw: string) {
    return raw
    .replace(/\$\{bindGroup_scene\}/g, constants.bindGroup_scene.toString())
    .replace(/\$\{bindGroup_model\}/g, constants.bindGroup_model.toString())
    .replace(/\$\{bindGroup_material\}/g, constants.bindGroup_material.toString())
    .replace(/\$\{moveLightsWorkgroupSize\}/g, constants.moveLightsWorkgroupSize.toString())

    .replace(/\$\{averageLightsPerCluster\}/g, constants.averageLightsPerCluster.toString())
    .replace(/\$\{maxLightsPerCluster\}/g, constants.maxLightsPerCluster.toString())
    
    .replace(/\$\{ambientR\}/g, constants.ambientR.toString())
    .replace(/\$\{ambientG\}/g, constants.ambientG.toString())
    .replace(/\$\{ambientB\}/g, constants.ambientB.toString())

    .replace(/\$\{lightRadius\}/g, constants.lightRadius.toString())

    .replace(/\$\{ddgiRaysPerProbe\}/g, constants.ddgiRaysPerProbe.toString())
    .replace(/\$\{ddgiIrradianceTexels\}/g, constants.ddgiIrradianceTexels.toString())
    .replace(/\$\{ddgiVisibilityTexels\}/g, constants.ddgiVisibilityTexels.toString())

    .replace(/\$\{rcIrradianceTexels\}/g, constants.rcIrradianceTexels.toString())
    .replace(/\$\{nrcMaxTrainingSamples\}/g, constants.nrcMaxTrainingSamples.toString());
}

const commonSrc: string = evalShaderRaw(commonRaw);
const giEvaluationSrc: string = evalShaderRaw(giEvaluationRaw);
const lightingCompositeSrc: string = evalShaderRaw(lightingCompositeRaw);
const standardMaterialSrc: string = evalShaderRaw(standardMaterialRaw);
const unlitMaterialSrc: string = evalShaderRaw(unlitMaterialRaw);

export interface MaterialShaderDesc {
    vertexShaderRaw?: string;
    materialEval: string;
}

const materials: Record<string, MaterialShaderDesc> = {
    'standard': { materialEval: standardMaterialSrc },
    'unlit': { materialEval: unlitMaterialSrc }
};

function processShaderRaw(raw: string) {
    return commonSrc + giEvaluationSrc + evalShaderRaw(raw);
}

export const standardVertSrc: string = processShaderRaw(standardVertRaw);

export function buildVertexShader(materialType: string): string {
    const desc = materials[materialType];
    if (desc && desc.vertexShaderRaw) {
        return processShaderRaw(desc.vertexShaderRaw);
    }
    return standardVertSrc;
}

const discardRegex = /if\s*\(surf\.alpha\s*<\s*0\.5f?\)\s*\{\s*discard;\s*\}/g;

export function buildGeometryShader(materialType: string, isOpaque: boolean): string {
    const matDesc = materials[materialType] || materials['standard'];
    const matSrc = matDesc.materialEval;
    let raw = evalShaderRaw(geometryFragRaw);
    if (isOpaque) raw = raw.replace(discardRegex, '');
    return commonSrc + matSrc + raw; 
}

export function buildForwardPlusShader(materialType: string, isOpaque: boolean): string {
    const matDesc = materials[materialType] || materials['standard'];
    const matSrc = matDesc.materialEval;
    let raw = evalShaderRaw(forwardPlusFragRaw);
    if (isOpaque) raw = raw.replace(discardRegex, '');
    return commonSrc + matSrc + lightingCompositeSrc + giEvaluationSrc + raw; 
}

export const fullscreenBlitVertSrc: string = processShaderRaw(fullscreenBlitVertRaw);
export const fullscreenBlitFragSrc: string = processShaderRaw(fullscreenBlitFragRaw);

export const clusteredDeferredComputeSrc: string = commonSrc + lightingCompositeSrc + giEvaluationSrc + evalShaderRaw(clusteredDeferredComputeSrcRaw);

export const moveLightsComputeSrc: string = processShaderRaw(moveLightsComputeRaw);
export const clusteringComputeSrc: string = processShaderRaw(clusteringComputeRaw);

export function buildZPrepassShader(materialType: string): string {
    const matDesc = materials[materialType] || materials['standard'];
    const matSrc = matDesc.materialEval;
    return commonSrc + matSrc + evalShaderRaw(zPrepassFragRaw);
}

export const debugBoxSrc: string = processShaderRaw(debugBoxRaw);

// IBL shaders (standalone, not prepended with common)
export const generateCubemapSrc = generateCubemapRaw;
export const irradianceConvolutionSrc = irradianceConvolutionRaw;
export const prefilterEnvmapSrc = prefilterEnvmapRaw;
export const brdfLutSrc = brdfLutRaw;
export const equirectangularToCubemapSrc = equirectangularToCubemapRaw;

// Skybox shaders (need common for CameraUniforms)
export const skyboxVertSrc: string = processShaderRaw(skyboxVertRaw);
export const skyboxFragSrc: string = processShaderRaw(skyboxFragRaw);

// SSAO shaders (need common)
export const ssaoFragSrc: string = processShaderRaw(ssaoFragRaw);
export const ssaoBlurFragSrc: string = processShaderRaw(ssaoBlurFragRaw);

// Volumetric shaders (need common)
export const volumetricLightingVertSrc: string = processShaderRaw(volumetricLightingVertRaw);
export const volumetricLightingFragSrc: string = processShaderRaw(volumetricLightingFragRaw);
export const volumetricCompositeFragSrc: string = processShaderRaw(volumetricCompositeFragRaw);

const bvhSrc: string = evalShaderRaw(bvhRaw);

// DDGI shaders (need common + bvh)
export const ddgiProbeTraceSrc: string = processShaderRaw(bvhSrc + ddgiProbeTraceRaw);
export const ddgiIrradianceUpdateSrc: string = processShaderRaw(ddgiIrradianceUpdateRaw);
export const ddgiVisibilityUpdateSrc: string = processShaderRaw(ddgiVisibilityUpdateRaw);
export const ddgiBorderUpdateSrc: string = ddgiBorderUpdateRaw;

// Radiance Cascades shaders
export const rcTraceSrc: string = processShaderRaw(rcTraceRaw);
export const rcBorderSrc: string = rcBorderRaw;

// Shadow shaders (standalone)
export const shadowVertSrc: string = shadowVertRaw;
export const shadowFragSrc: string = shadowFragRaw;

// VSM shaders (need common for VSMUniforms struct)
export const vsmClearSrc: string = processShaderRaw(vsmClearRaw);
export const vsmMarkPagesSrc: string = processShaderRaw(vsmMarkPagesRaw);
export const vsmAllocatePagesSrc: string = processShaderRaw(vsmAllocatePagesRaw);

// NRC shaders (need common + nrc_common for structs/utilities)
const nrcCommonSrc: string = evalShaderRaw(nrcCommonRaw);
function processNrcShaderRaw(raw: string) {
    return commonSrc + nrcCommonSrc + evalShaderRaw(raw);
}
export const nrcScatterTrainingSrc: string = processNrcShaderRaw(nrcScatterTrainingRaw);
export const nrcTrainSrc: string = processNrcShaderRaw(nrcTrainRaw);
export const nrcInferenceSrc: string = processNrcShaderRaw(nrcInferenceRaw);

// Surfel shaders (need common + surfel_common + bvh for structs/utilities)
const surfelCommonSrc: string = evalShaderRaw(surfelCommonRaw);
function processSurfelShaderRaw(raw: string) {
    return commonSrc + surfelCommonSrc + bvhSrc + evalShaderRaw(raw);
}
export const surfelLifecycleSrc: string = processSurfelShaderRaw(surfelLifecycleRaw);
export const surfelGridSrc: string = processSurfelShaderRaw(surfelGridRaw);
export const surfelIntegratorSrc: string = processSurfelShaderRaw(surfelIntegratorRaw);
export const surfelResolveSrc: string = processSurfelShaderRaw(surfelResolveRaw);