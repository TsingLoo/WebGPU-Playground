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

import hizDownsampleRaw from './hiz/hiz_downsample.cs.wgsl?raw';

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
import ddgiProbeRelocateRaw from './gi/ddgi/ddgi_probe_relocate.cs.wgsl?raw';
import ddgiDebugProbesRaw from './gi/ddgi/ddgi_debug_probes.wgsl?raw';

// Radiance Cascades
import rcTraceRaw from './gi/radiance_cascades/rc_trace.cs.wgsl?raw';
import rcBorderRaw from './gi/radiance_cascades/rc_border.cs.wgsl?raw';

// NRC shaders
import nrcCommonRaw from './gi/nrc/nrc_common.wgsl?raw';
import nrcScatterTrainingRaw from './gi/nrc/nrc_scatter_training.cs.wgsl?raw';
import nrcTrainRaw from './gi/nrc/nrc_train.cs.wgsl?raw';
import nrcInferenceRaw from './gi/nrc/nrc_inference.cs.wgsl?raw';
import nrcPtCollectRaw from './gi/nrc/nrc_pt_collect.cs.wgsl?raw';

// Surfel shaders
import surfelCommonRaw from './gi/surfel/surfel_common.wgsl?raw';
import bvhRaw from './gi/surfel/bvh.wgsl?raw';
import surfelLifecycleRaw from './gi/surfel/surfel_lifecycle.cs.wgsl?raw';
import surfelGridRaw from './gi/surfel/surfel_grid.cs.wgsl?raw';
import surfelIntegratorRaw from './gi/surfel/surfel_integrator.cs.wgsl?raw';
import surfelResolveRaw from './gi/surfel/surfel_resolve.cs.wgsl?raw';

// Path Tracing (Wavefront) shaders
import ptCommonRaw from './gi/path_tracing/pt_common.wgsl?raw';
import ptRayGenRaw from './gi/path_tracing/ray_gen.cs.wgsl?raw';
import ptIntersectRaw from './gi/path_tracing/intersect.cs.wgsl?raw';
import ptShadeRaw from './gi/path_tracing/shade.cs.wgsl?raw';
import ptShadowTestRaw from './gi/path_tracing/shadow_test.cs.wgsl?raw';
import ptMissRaw from './gi/path_tracing/miss.cs.wgsl?raw';
import ptAccumulateRaw from './gi/path_tracing/accumulate.cs.wgsl?raw';
import ptTonemapRaw from './gi/path_tracing/pt_tonemap.wgsl?raw';
import spectralCommonRaw from './gi/path_tracing/spectral_common.wgsl?raw';

// ReSTIR DI shaders
import restirCommonRaw from './gi/path_tracing/restir_common.wgsl?raw';
import restirInitialRaw from './gi/path_tracing/restir_initial.cs.wgsl?raw';
import restirTemporalRaw from './gi/path_tracing/restir_temporal.cs.wgsl?raw';
import restirSpatialRaw from './gi/path_tracing/restir_spatial.cs.wgsl?raw';
import restirShadeRaw from './gi/path_tracing/restir_shade.cs.wgsl?raw';

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

// SSR shaders
import ssrFragRaw from './postprocessing/ssr.fs.wgsl?raw';
import ssrCompositeFragRaw from './postprocessing/ssr_composite.fs.wgsl?raw';

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

function injectMocks(rawCode: string, composed: string): string {
    if (!rawCode.includes("ddgiProbeData")) {
        return "var<private> ddgiProbeData: array<vec4f, 1>;\n" + composed;
    }
    return composed;
}

function processShaderRaw(raw: string) {
    let evaled = evalShaderRaw(raw);
    let gi = injectMocks(evaled, giEvaluationSrc + evaled).replace(evaled, ""); // Just get the injected part
    return commonSrc + gi + evaled;
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
    let gi = injectMocks(raw, giEvaluationSrc + raw).replace(raw, "");
    return commonSrc + matSrc + lightingCompositeSrc + gi + raw; 
}

export const fullscreenBlitVertSrc: string = processShaderRaw(fullscreenBlitVertRaw);
export const fullscreenBlitFragSrc: string = processShaderRaw(fullscreenBlitFragRaw);

let clusteredRaw = evalShaderRaw(clusteredDeferredComputeSrcRaw);
export const clusteredDeferredComputeSrc: string = commonSrc + lightingCompositeSrc + injectMocks(clusteredRaw, giEvaluationSrc + clusteredRaw).replace(clusteredRaw, "") + clusteredRaw;

export const moveLightsComputeSrc: string = processShaderRaw(moveLightsComputeRaw);
export const clusteringComputeSrc: string = processShaderRaw(clusteringComputeRaw);

export const hizDownsampleSrc: string = processShaderRaw(hizDownsampleRaw);

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

// SSR shaders (need common)
export const ssrFragSrc: string = processShaderRaw(ssrFragRaw);
export const ssrCompositeFragSrc: string = processShaderRaw(ssrCompositeFragRaw);

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
export const ddgiProbeRelocateSrc: string = processShaderRaw(ddgiProbeRelocateRaw);
export const ddgiDebugProbesSrc: string = processShaderRaw(ddgiDebugProbesRaw);

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

// ===========================================================================
// Path Tracing (Wavefront) shaders
// ===========================================================================
const ptCommonSrc: string = evalShaderRaw(ptCommonRaw);
const spectralCommonSrc: string = evalShaderRaw(spectralCommonRaw);

// Builds a complete PT compute shader: common + nrc_common + bvh + pt_common + spectral_common + shader_body
function processPTShaderRaw(raw: string): string {
    return commonSrc + nrcCommonSrc + bvhSrc + ptCommonSrc + spectralCommonSrc + evalShaderRaw(raw);
}

// RayGen needs camera + pt_common + spectral_common (no BVH)
export const ptRayGenSrc: string = commonSrc + ptCommonSrc + spectralCommonSrc + evalShaderRaw(ptRayGenRaw);

// Intersection needs BVH
export const ptIntersectSrc: string = processPTShaderRaw(ptIntersectRaw);

// Shade needs pt_common (for material/RNG helpers) + sun light structs from common
export const ptShadeSrc: string = processPTShaderRaw(ptShadeRaw);

// Shadow test needs BVH
export const ptShadowTestSrc: string = processPTShaderRaw(ptShadowTestRaw);

// Miss needs pt_common + spectral_common (no BVH)
export const ptMissSrc: string = commonSrc + ptCommonSrc + spectralCommonSrc + evalShaderRaw(ptMissRaw);

// Accumulate: no BVH, no RNG, needs spectral_common for conversion
export const ptAccumulateSrc: string = commonSrc + ptCommonSrc + spectralCommonSrc + evalShaderRaw(ptAccumulateRaw);

// Tonemap: standalone (vertex + fragment in one file, split by entry points)
export const ptTonemapSrc: string = commonSrc + ptCommonSrc + evalShaderRaw(ptTonemapRaw);

// NRC Collect Training requires full PT structs and NRC structs
export const nrcPtCollectSrc: string = processPTShaderRaw(nrcPtCollectRaw);

// ===========================================================================
// ReSTIR DI shaders
// ===========================================================================
const restirCommonSrc: string = evalShaderRaw(restirCommonRaw);

// Builds a complete ReSTIR compute shader: common + nrc_common + bvh + pt_common + restir_common + shader_body
function processReSTIRShaderRaw(raw: string): string {
    return commonSrc + nrcCommonSrc + bvhSrc + ptCommonSrc + restirCommonSrc + evalShaderRaw(raw);
}

export const ptRestirInitialSrc: string = processReSTIRShaderRaw(restirInitialRaw);
export const ptRestirTemporalSrc: string = processReSTIRShaderRaw(restirTemporalRaw);
export const ptRestirSpatialSrc: string = processReSTIRShaderRaw(restirSpatialRaw);
export const ptRestirShadeSrc: string = processReSTIRShaderRaw(restirShadeRaw);