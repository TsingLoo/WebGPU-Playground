// CHECKITOUT: this file loads all the shaders and preprocesses them with some common code

import commonRaw from './core/common.wgsl?raw';
import standardMaterialRaw from './materials/standard_material.wgsl?raw';
import unlitMaterialRaw from './materials/unlit_material.wgsl?raw';
import giEvaluationRaw from './core/gi_evaluation.wgsl?raw';

import naiveVertRaw from './core/naive.vs.wgsl?raw';
import naiveFragRaw from './core/naive.fs.wgsl?raw';

import geometryFragRaw from './core/geometry.fs.wgsl?raw';

import forwardPlusFragRaw from './forward_plus/forward_plus.fs.wgsl?raw';

import clusteredDeferredFragRaw from './clustered_deferred/clustered_deferred.fs.wgsl?raw';
import clusteredDeferredFullscreenVertRaw from './clustered_deferred/clustered_deferred_fullscreen.vs.wgsl?raw';
import clusteredDeferredFullscreenFragRaw from './clustered_deferred/clustered_deferred_fullscreen.fs.wgsl?raw';

import clusteredDeferredComputeSrcRaw from './clustered_deferred/clusteredDeferred.cs.wgsl?raw';

import moveLightsComputeRaw from './forward_plus/move_lights.cs.wgsl?raw';
import clusteringComputeRaw from './forward_plus/clustering.cs.wgsl?raw';

import zPrepassFragRaw from './core/zPrepass.fs.wgsl?raw';

// IBL shaders
import generateCubemapRaw from './ibl/generate_cubemap.cs.wgsl?raw';
import irradianceConvolutionRaw from './ibl/irradiance_convolution.cs.wgsl?raw';
import prefilterEnvmapRaw from './ibl/prefilter_envmap.cs.wgsl?raw';
import brdfLutRaw from './ibl/brdf_lut.cs.wgsl?raw';
import equirectangularToCubemapRaw from './ibl/equirectangular_to_cubemap.cs.wgsl?raw';

// DDGI
import ddgiProbeTraceRaw from './ddgi/ddgi_probe_trace.cs.wgsl?raw';
import ddgiIrradianceUpdateRaw from './ddgi/ddgi_irradiance_update.cs.wgsl?raw';
import ddgiVisibilityUpdateRaw from './ddgi/ddgi_visibility_update.cs.wgsl?raw';
import ddgiBorderUpdateRaw from './ddgi/ddgi_border_update.cs.wgsl?raw';

// Radiance Cascades
import rcTraceRaw from './radiance_cascades/rc_trace.cs.wgsl?raw';
import rcBorderRaw from './radiance_cascades/rc_border.cs.wgsl?raw';

// NRC shaders
import nrcCommonRaw from './nrc/nrc_common.wgsl?raw';
import nrcScatterTrainingRaw from './nrc/nrc_scatter_training.cs.wgsl?raw';
import nrcTrainRaw from './nrc/nrc_train.cs.wgsl?raw';
import nrcInferenceRaw from './nrc/nrc_inference.cs.wgsl?raw';

// Surfel shaders
import surfelCommonRaw from './surfel/surfel_common.wgsl?raw';
import bvhRaw from './surfel/bvh.wgsl?raw';
import surfelLifecycleRaw from './surfel/surfel_lifecycle.cs.wgsl?raw';
import surfelGridRaw from './surfel/surfel_grid.cs.wgsl?raw';
import surfelIntegratorRaw from './surfel/surfel_integrator.cs.wgsl?raw';
import surfelResolveRaw from './surfel/surfel_resolve.cs.wgsl?raw';

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
    rcProbeGridX: 8,
    rcProbeGridY: 8,
    rcProbeGridZ: 8,
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
const standardMaterialSrc: string = evalShaderRaw(standardMaterialRaw);
const unlitMaterialSrc: string = evalShaderRaw(unlitMaterialRaw);

const materials: Record<string, string> = {
    'standard': standardMaterialSrc,
    'unlit': unlitMaterialSrc
};

function processShaderRaw(raw: string) {
    return commonSrc + giEvaluationSrc + evalShaderRaw(raw);
}

export const naiveVertSrc: string = processShaderRaw(naiveVertRaw);
export const naiveFragSrc: string = processShaderRaw(naiveFragRaw);

const discardRegex = /if\s*\(surf\.alpha\s*<\s*0\.5f?\)\s*\{\s*discard;\s*\}/g;

export function buildGeometryShader(materialType: string, isOpaque: boolean): string {
    const matSrc = materials[materialType] || materials['standard'];
    let raw = evalShaderRaw(geometryFragRaw);
    if (isOpaque) raw = raw.replace(discardRegex, '');
    return commonSrc + matSrc + raw; 
}

export function buildForwardPlusShader(materialType: string, isOpaque: boolean): string {
    const matSrc = materials[materialType] || materials['standard'];
    let raw = evalShaderRaw(forwardPlusFragRaw);
    if (isOpaque) raw = raw.replace(discardRegex, '');
    return commonSrc + matSrc + giEvaluationSrc + raw; 
}

export const clusteredDeferredFragSrc: string = processShaderRaw(clusteredDeferredFragRaw);
export const clusteredDeferredFullscreenVertSrc: string = processShaderRaw(clusteredDeferredFullscreenVertRaw);
export const clusteredDeferredFullscreenFragSrc: string = processShaderRaw(clusteredDeferredFullscreenFragRaw);

export const clusteredDeferredComputeSrc: string = processShaderRaw(clusteredDeferredComputeSrcRaw);

export const moveLightsComputeSrc: string = processShaderRaw(moveLightsComputeRaw);
export const clusteringComputeSrc: string = processShaderRaw(clusteringComputeRaw);

export function buildZPrepassShader(materialType: string): string {
    const matSrc = materials[materialType] || materials['standard'];
    return commonSrc + matSrc + evalShaderRaw(zPrepassFragRaw);
}

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

// DDGI shaders (need common)
export const ddgiProbeTraceSrc: string = processShaderRaw(ddgiProbeTraceRaw);
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
const bvhSrc: string = evalShaderRaw(bvhRaw);
function processSurfelShaderRaw(raw: string) {
    return commonSrc + surfelCommonSrc + bvhSrc + evalShaderRaw(raw);
}
export const surfelLifecycleSrc: string = processSurfelShaderRaw(surfelLifecycleRaw);
export const surfelGridSrc: string = processSurfelShaderRaw(surfelGridRaw);
export const surfelIntegratorSrc: string = processSurfelShaderRaw(surfelIntegratorRaw);
export const surfelResolveSrc: string = processSurfelShaderRaw(surfelResolveRaw);