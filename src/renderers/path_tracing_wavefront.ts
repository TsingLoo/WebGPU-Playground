/**
 * path_tracing_wavefront.ts
 *
 * Wavefront Path Tracing Renderer for WebGPU.
 * With full ReSTIR DI (Reservoir-based Spatiotemporal Importance Resampling).
 *
 * Architecture:
 *   Each frame = N bounce iterations of:
 *     1. ray_gen (bounce 0 only)   → rayBuffer[]
 *     2. intersect                 → hitBuffer[]
 *     3. shade + NEE               → new rayBuffer[], shadowBuffer[], accumBuffer[]
 *        (NEE skipped for bounce 0 when ReSTIR is enabled)
 *     4. [ReSTIR, bounce 0 only]:
 *        a. restir_initial        → reservoirA[]
 *        b. restir_temporal       → reservoirA[] (merged with prev frame)
 *        c. restir_spatial        → reservoirB[] (merged with neighbors)
 *        d. restir_shade          → shadowBuffer[] (from reservoir)
 *     5. shadow_test               → accumBuffer[] (direct light contribution)
 *     6. miss (env)                → accumBuffer[]
 *   Then:
 *     7. accumulate  → adds accumBuffer to sampleSumBuffer (persistent), clears accumBuffer
 *     8. tonemap blit → reads sampleSumBuffer / sample_count → canvas
 */

import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { Renderer } from '../renderer';
import { pipelineCache } from '../engine/PipelineCache';
import { NRC } from '../stage/nrc';

// ============================================================
// PTUniforms layout (matches pt_common.wgsl PTUniforms struct)
// 8 × 4 = 32 bytes
// ============================================================
interface PTConfig {
    maxBounces: number;
    clampRadiance: number;
    pixelScale: number;
    // ReSTIR DI
    restirEnabled: boolean;
    restirCandidateCount: number;   // M = initial candidates per pixel
    restirSpatialRadius: number;    // R = pixel radius for spatial resampling
    restirSpatialCount: number;     // K = number of spatial neighbors
    restirTemporalMaxM: number;     // temporal M clamp multiplier
}

export class WavefrontPathTracingRenderer extends Renderer {

    // ------------ Configuration ------------
    config: PTConfig = {
        maxBounces: 4,
        clampRadiance: 10.0,
        pixelScale: 1.0,
        // ReSTIR defaults
        restirEnabled: true,
        restirCandidateCount: 32,
        restirSpatialRadius: 30,
        restirSpatialCount: 5,
        restirTemporalMaxM: 20,
    };

    // ------------ Accumulation State ------------
    sampleCount: number = 0;
    private lastCameraPos: [number, number, number] = [0, 0, 0];
    private lastCameraFront: [number, number, number] = [0, 0, -1];

    // ------------ GPU Buffers ------------
    private rayBuffer!: GPUBuffer;
    private hitBuffer!: GPUBuffer;
    private shadowBuffer!: GPUBuffer;
    private accumWorkBuffer!: GPUBuffer;  // Per-pixel radiance sum (cleared each frame)
    private sampleSumBuffer!: GPUBuffer;  // Persistent sum across all samples
    private ptUniformBuffer!: GPUBuffer;  // PTUniforms (32 bytes)
    private ptNRCTrainDataBuffer!: GPUBuffer; // Array of NRCWavefrontTrainData

    // ------------ ReSTIR GPU Buffers ------------
    private reservoirBufferA!: GPUBuffer; // Current frame reservoir
    private reservoirBufferB!: GPUBuffer; // Ping-pong / spatial output
    private prevReservoirBuffer!: GPUBuffer; // Previous frame reservoir
    private restirUniformBuffer!: GPUBuffer; // ReSTIRUniforms (32 bytes)
    private prevCameraBuffer!: GPUBuffer;    // Previous frame camera uniforms
    private pixelDataBuffer!: GPUBuffer;     // Current frame normal+depth per pixel (vec4f)
    private prevPixelDataBuffer!: GPUBuffer; // Previous frame normal+depth per pixel

    private renderWidth!: number;
    private renderHeight!: number;
    private totalPixels!: number;

    // ------------ Pipelines ------------
    private rayGenPipeline!: GPUComputePipeline;
    private intersectPipeline!: GPUComputePipeline;
    private shadePipeline!: GPUComputePipeline;
    private shadowTestPipeline!: GPUComputePipeline;
    private missPipeline!: GPUComputePipeline;
    private accumulatePipeline!: GPUComputePipeline;
    private nrcCollectPipeline!: GPUComputePipeline;
    private tonemapPipeline!: GPURenderPipeline;

    // ReSTIR Pipelines
    private restirInitialPipeline!: GPUComputePipeline;
    private restirTemporalPipeline!: GPUComputePipeline;
    private restirSpatialPipeline!: GPUComputePipeline;
    private restirShadePipeline!: GPUComputePipeline;

    // ------------ Bind Group Layouts ------------
    private rayGenLayout!: GPUBindGroupLayout;
    private intersectLayout!: GPUBindGroupLayout;
    private shadeLayout!: GPUBindGroupLayout;
    private shadeNRCLayout!: GPUBindGroupLayout;
    private shadowTestLayout!: GPUBindGroupLayout;
    private missLayout!: GPUBindGroupLayout;
    private accumulateLayout!: GPUBindGroupLayout;
    private nrcCollectLayout!: GPUBindGroupLayout;
    private tonemapLayout!: GPUBindGroupLayout;

    // ReSTIR Layouts
    private restirInitialLayout!: GPUBindGroupLayout;
    private restirTemporalLayout!: GPUBindGroupLayout;
    private restirSpatialLayout!: GPUBindGroupLayout;
    private restirShadeLayout!: GPUBindGroupLayout;

    // ------------ ReSTIR Frame Counter ------------
    private restirFrameIndex: number = 0;

    constructor(stage: Stage) {
        super(stage);
        console.log('[WPT] Starting initialization...');
        this.initGPUResources();
        console.log('[WPT] GPU resources allocated');
        this.initPipelines();
        console.log('[WPT] Wavefront Path Tracing Renderer initialized (ReSTIR enabled:', this.config.restirEnabled, ')');
    }

    // ============================================================
    // Initialization
    // ============================================================
    private initGPUResources() {
        const dev = renderer.device;
        const W = renderer.canvas.width;
        const H = renderer.canvas.height;
        this.renderWidth  = Math.max(1, Math.floor(W * this.config.pixelScale));
        this.renderHeight = Math.max(1, Math.floor(H * this.config.pixelScale));
        this.totalPixels  = this.renderWidth * this.renderHeight;

        console.log(`[WPT] Render: ${this.renderWidth}×${this.renderHeight}, ${this.totalPixels} pixels`);

        const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

        // Ray buffer: 64 bytes per pixel
        this.rayBuffer = dev.createBuffer({ label: 'PT Ray Buffer',    size: this.totalPixels * 64, usage: storageUsage });
        // Hit buffer: 48 bytes per pixel (compact: bary + vertex indices)
        this.hitBuffer = dev.createBuffer({ label: 'PT Hit Buffer',    size: this.totalPixels * 48, usage: storageUsage });
        // Shadow ray buffer: 48 bytes per pixel
        this.shadowBuffer = dev.createBuffer({ label: 'PT Shadow Buf', size: this.totalPixels * 48, usage: storageUsage });
        // Per-frame radiance accumulator (cleared after each frame)
        this.accumWorkBuffer = dev.createBuffer({ label: 'PT Accum Work', size: this.totalPixels * 16, usage: storageUsage });
        // Persistent sum across all frames (NOT cleared unless resetAccumulation() is called)
        this.sampleSumBuffer = dev.createBuffer({ label: 'PT Sample Sum',  size: this.totalPixels * 16, usage: storageUsage });

        // PT Uniforms buffer: 8 × u32/f32 = 32 bytes
        this.ptUniformBuffer = dev.createBuffer({
            label: 'PT Uniforms',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ptNRCTrainDataBuffer: array of NRCWavefrontTrainData (96 bytes each)
        this.ptNRCTrainDataBuffer = dev.createBuffer({
            label: 'PT NRC Train Data',
            size: NRC.MAX_TRAINING_SAMPLES * 96,
            usage: GPUBufferUsage.STORAGE,
        });

        // ============================================================
        // ReSTIR Buffers
        // ============================================================
        const reservoirSize = this.totalPixels * 64; // Reservoir = 64 bytes per pixel
        console.log(`[ReSTIR] Allocating reservoir buffers: ${(reservoirSize / (1024*1024)).toFixed(2)} MB each`);

        this.reservoirBufferA = dev.createBuffer({
            label: 'ReSTIR Reservoir A',
            size: reservoirSize,
            usage: storageUsage,
        });
        this.reservoirBufferB = dev.createBuffer({
            label: 'ReSTIR Reservoir B',
            size: reservoirSize,
            usage: storageUsage,
        });
        this.prevReservoirBuffer = dev.createBuffer({
            label: 'ReSTIR Prev Reservoir',
            size: reservoirSize,
            usage: storageUsage,
        });
        console.log('[ReSTIR] Reservoir buffers created (A, B, Prev)');

        // ReSTIR uniforms: 32 bytes (8 × u32)
        this.restirUniformBuffer = dev.createBuffer({
            label: 'ReSTIR Uniforms',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        console.log('[ReSTIR] Uniform buffer created');

        // Previous camera buffer (same size as camera uniforms)
        this.prevCameraBuffer = dev.createBuffer({
            label: 'ReSTIR Prev Camera',
            size: this.camera.uniforms.buffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        console.log('[ReSTIR] Previous camera buffer created');

        // Per-pixel data: vec4f(normal.xyz, linear_depth) — 16 bytes per pixel
        const pixelDataSize = this.totalPixels * 16;
        this.pixelDataBuffer = dev.createBuffer({
            label: 'ReSTIR Pixel Data',
            size: pixelDataSize,
            usage: storageUsage,
        });
        this.prevPixelDataBuffer = dev.createBuffer({
            label: 'ReSTIR Prev Pixel Data',
            size: pixelDataSize,
            usage: storageUsage,
        });
        console.log(`[ReSTIR] Pixel data buffers created: ${(pixelDataSize / (1024*1024)).toFixed(2)} MB each`);

        this.uploadPTUniforms();
        this.uploadReSTIRUniforms();
    }

    private initPipelines() {
        const dev = renderer.device;

        // ---- Bind Group Layouts (existing passes) ----

        this.rayGenLayout = dev.createBindGroupLayout({
            label: 'PT RayGen Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },   // camera
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },   // pt uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // ray_buffer
            ]
        });

        this.intersectLayout = dev.createBindGroupLayout({
            label: 'PT Intersect Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // ray_buffer
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // hit_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh nodes
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh pos
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh indices
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // materials
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // base_color_tex
                { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: {} },                            // tex_sampler
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_uvs
            ]
        });

        this.shadeLayout = dev.createBindGroupLayout({
            label: 'PT Shade Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // camera
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // ray_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // hit_buffer
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // shadow_buffer
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // accum_buffer
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // materials
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // sun_light
                { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // base_color
                { binding: 9, visibility: GPUShaderStage.COMPUTE, sampler: {} },                            // tex_sampler (shared)
                { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // normal_map
                { binding: 11, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // mr_tex
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_uvs
                { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_normals
                { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_tangents
            ]
        });

        this.shadeNRCLayout = dev.createBindGroupLayout({
            label: 'PT Shade NRC Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },   // NRCUniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // weights
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sampleCounter
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // ptNRCTrainData
            ]
        });

        this.shadowTestLayout = dev.createBindGroupLayout({
            label: 'PT ShadowTest Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // shadow_buffer
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // accum_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh nodes
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh pos
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh indices
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // materials
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // base_color_tex
                { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: {} },                            // tex_sampler
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_uvs
            ]
        });

        this.missLayout = dev.createBindGroupLayout({
            label: 'PT Miss Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // ray_buffer
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // hit_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // accum_buffer
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: 'cube' } }, // env
                { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: {} },                  // env sampler
            ]
        });

        this.accumulateLayout = dev.createBindGroupLayout({
            label: 'PT Accumulate Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // accum_buffer (work, cleared each frame)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // sample_sum_buf (persistent)
            ]
        });

        this.nrcCollectLayout = dev.createBindGroupLayout({
            label: 'PT NRC Collect Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // NRCUniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // ptNRCTrainData
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // accum_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // sampleCounter
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // trainingSamples
            ]
        });

        this.tonemapLayout = dev.createBindGroupLayout({
            label: 'PT Tonemap Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },          // pt
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // sample_sum_buf
            ]
        });

        // ============================================================
        // ReSTIR Bind Group Layouts
        // ============================================================
        console.log('[ReSTIR] Creating bind group layouts...');

        this.restirInitialLayout = dev.createBindGroupLayout({
            label: 'ReSTIR Initial Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // restir uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // ray_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // hit_buffer
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // materials
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // sun_light
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // reservoir_buffer
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // emissive_indices
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // bvh_pos
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // bvh_normals
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_uvs
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // pixel_data_out
            ]
        });
        console.log('[ReSTIR] Initial layout created');

        this.restirTemporalLayout = dev.createBindGroupLayout({
            label: 'ReSTIR Temporal Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // restir uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // camera
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // prev_camera
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // reservoir_buffer (r/w)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // prev_reservoir
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // ray_buffer
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // hit_buffer
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // bvh_normals
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // prev_pixel_data
            ]
        });
        console.log('[ReSTIR] Temporal layout created');

        this.restirSpatialLayout = dev.createBindGroupLayout({
            label: 'ReSTIR Spatial Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // restir uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // reservoir_in
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // reservoir_out
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // ray_buffer
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // hit_buffer
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // bvh_normals
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // pixel_data
            ]
        });
        console.log('[ReSTIR] Spatial layout created');

        this.restirShadeLayout = dev.createBindGroupLayout({
            label: 'ReSTIR Shade Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // restir uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // camera
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // ray_buffer
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // hit_buffer
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // reservoir_buffer
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // shadow_buffer
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // accum_buffer
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // materials
                { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // base_color_tex
                { binding: 10, visibility: GPUShaderStage.COMPUTE, sampler: {} },                            // tex_sampler
                { binding: 11, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // mr_tex
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_uvs
                { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_normals
                { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh_tangents
                { binding: 15, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } }, // normal_map_tex
                { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // pixel_data_out
            ]
        });
        console.log('[ReSTIR] Shade layout created');

        // ---- Compute Pipelines ----
        const mkCompute = (label: string, code: string, layout: GPUBindGroupLayout) =>
            pipelineCache.getComputePipeline(dev, {
                label,
                layout: dev.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: dev.createShaderModule({ label, code }), entryPoint: 'main' }
            }, label);

        const mkComputeEx = (label: string, code: string, layouts: GPUBindGroupLayout[]) =>
            pipelineCache.getComputePipeline(dev, {
                label,
                layout: dev.createPipelineLayout({ bindGroupLayouts: layouts }),
                compute: { module: dev.createShaderModule({ label, code }), entryPoint: 'main' }
            }, label);

        this.rayGenPipeline     = mkCompute('WPT RayGen',     shaders.ptRayGenSrc,     this.rayGenLayout);
        console.log('[WPT] RayGen pipeline created');
        this.intersectPipeline  = mkCompute('WPT Intersect',  shaders.ptIntersectSrc,  this.intersectLayout);
        console.log('[WPT] Intersect pipeline created');
        this.shadePipeline      = mkComputeEx('WPT Shade',    shaders.ptShadeSrc,      [this.shadeLayout, this.shadeNRCLayout]);
        console.log('[WPT] Shade pipeline created');
        this.shadowTestPipeline = mkCompute('WPT ShadowTest', shaders.ptShadowTestSrc, this.shadowTestLayout);
        console.log('[WPT] ShadowTest pipeline created');
        this.missPipeline       = mkCompute('WPT Miss',       shaders.ptMissSrc,       this.missLayout);
        console.log('[WPT] Miss pipeline created');
        this.accumulatePipeline = mkCompute('WPT Accumulate', shaders.ptAccumulateSrc, this.accumulateLayout);
        console.log('[WPT] Accumulate pipeline created');
        this.nrcCollectPipeline = mkCompute('WPT NRC Collect', shaders.nrcPtCollectSrc, this.nrcCollectLayout);
        console.log('[WPT] NRC Collect pipeline created');

        // ---- ReSTIR Pipelines ----
        console.log('[ReSTIR] Creating compute pipelines...');
        this.restirInitialPipeline  = mkCompute('ReSTIR Initial',  shaders.ptRestirInitialSrc,  this.restirInitialLayout);
        console.log('[ReSTIR] Initial pipeline created');
        this.restirTemporalPipeline = mkCompute('ReSTIR Temporal', shaders.ptRestirTemporalSrc, this.restirTemporalLayout);
        console.log('[ReSTIR] Temporal pipeline created');
        this.restirSpatialPipeline  = mkCompute('ReSTIR Spatial',  shaders.ptRestirSpatialSrc,  this.restirSpatialLayout);
        console.log('[ReSTIR] Spatial pipeline created');
        this.restirShadePipeline    = mkCompute('ReSTIR Shade',    shaders.ptRestirShadeSrc,    this.restirShadeLayout);
        console.log('[ReSTIR] Shade pipeline created');

        // ---- Tonemap Render Pipeline ----
        const tonemapModule = dev.createShaderModule({ label: 'WPT Tonemap', code: shaders.ptTonemapSrc });
        this.tonemapPipeline = dev.createRenderPipeline({
            label: 'WPT Tonemap Pipeline',
            layout: dev.createPipelineLayout({ bindGroupLayouts: [this.tonemapLayout] }),
            vertex:   { module: tonemapModule, entryPoint: 'vs_main' },
            fragment: {
                module: tonemapModule,
                entryPoint: 'fs_main',
                targets: [{ format: renderer.canvasFormat }]
            },
            primitive: { topology: 'triangle-list' }
        });

        console.log('[WPT] All pipelines created');
    }

    // ============================================================
    // Camera Change Detection → Reset Accumulation
    // ============================================================
    private checkCameraChanged(): boolean {
        const pos   = this.camera.cameraPos as any;
        const front = this.camera.cameraFront as any;
        const eps = 0.0001;
        const changed = (
            Math.abs(pos[0] - this.lastCameraPos[0]) > eps ||
            Math.abs(pos[1] - this.lastCameraPos[1]) > eps ||
            Math.abs(pos[2] - this.lastCameraPos[2]) > eps ||
            Math.abs(front[0] - this.lastCameraFront[0]) > eps ||
            Math.abs(front[1] - this.lastCameraFront[1]) > eps ||
            Math.abs(front[2] - this.lastCameraFront[2]) > eps
        );
        this.lastCameraPos   = [pos[0],   pos[1],   pos[2]];
        this.lastCameraFront = [front[0], front[1], front[2]];
        return changed;
    }

    resetAccumulation() {
        this.sampleCount = 0;
        const zeros = new Float32Array(this.totalPixels * 4);
        renderer.device.queue.writeBuffer(this.accumWorkBuffer,  0, zeros);
        renderer.device.queue.writeBuffer(this.sampleSumBuffer, 0, zeros);
        console.log('[WPT] Accumulation reset');
    }

    // ============================================================
    // PTUniforms upload
    // ============================================================
    private uploadPTUniforms() {
        const buf = new ArrayBuffer(32);
        const u32 = new Uint32Array(buf);
        const f32 = new Float32Array(buf);
        u32[0] = this.renderWidth;
        u32[1] = this.renderHeight;
        u32[2] = this.camera.frameCount;
        u32[3] = this.sampleCount;
        u32[4] = this.config.maxBounces;
        f32[5] = this.config.clampRadiance;
        f32[6] = this.config.pixelScale;
        u32[7] = this.config.restirEnabled ? 1 : 0;
        renderer.device.queue.writeBuffer(this.ptUniformBuffer, 0, buf);
    }

    // ============================================================
    // ReSTIR Uniforms upload
    // ============================================================
    private uploadReSTIRUniforms() {
        const buf = new ArrayBuffer(32);
        const u32 = new Uint32Array(buf);
        const bvhData = this.stage.scene.bvhData;
        u32[0] = this.config.restirCandidateCount;
        u32[1] = this.config.restirSpatialRadius;
        u32[2] = this.config.restirSpatialCount;
        u32[3] = this.config.restirTemporalMaxM;
        u32[4] = this.config.restirEnabled ? 1 : 0;
        u32[5] = this.restirFrameIndex;
        u32[6] = bvhData ? bvhData.emissiveTriCount : 0;
        u32[7] = 0; // pad
        renderer.device.queue.writeBuffer(this.restirUniformBuffer, 0, buf);
    }

    // ============================================================
    // Main Draw Loop
    // ============================================================
    protected override draw(): void {
        const dev   = renderer.device;
        const scene = this.stage.scene;

        // Flush sun light changes to GPU buffer
        this.stage.updateSunLight();

        if (this.checkCameraChanged()) {
            this.resetAccumulation();
        }

        this.sampleCount++;
        this.restirFrameIndex++;
        this.uploadPTUniforms();
        this.uploadReSTIRUniforms();

        const bvhData = scene.bvhData;
        const encoder = dev.createCommandEncoder({ label: 'WPT Frame' });

        const rW  = Math.ceil(this.renderWidth  / 8);
        const rH  = Math.ceil(this.renderHeight / 8);
        const r1D = Math.ceil(this.totalPixels  / 64);

        const baseColorSampler = dev.createSampler({
            magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'repeat', addressModeV: 'repeat'
        });

        // Initialize NRC states
        if (this.stage.nrc.enabled) {
            this.stage.nrc.updateUniforms(); // ensures params are ready
            // Reset the sample counter
            encoder.copyBufferToBuffer(
                this.stage.nrc.sampleCounterZeroBuffer, 0,
                this.stage.nrc.sampleCounterBuffer, 0,
                4
            );
        }

        // ---- Pass 0: Primary Ray Generation ----
        {
            const bg = dev.createBindGroup({
                layout: this.rayGenLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                    { binding: 1, resource: { buffer: this.ptUniformBuffer } },
                    { binding: 2, resource: { buffer: this.rayBuffer } },
                ]
            });
            const pass = encoder.beginComputePass({ label: 'WPT RayGen' });
            pass.setPipeline(this.rayGenPipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(rW, rH, 1);
            pass.end();
        }

        // ---- Bounce Loop ----
        for (let bounce = 0; bounce < this.config.maxBounces; bounce++) {

            // Pass 1: Intersect (with alpha-cutout transparency)
            {
                const bg = dev.createBindGroup({
                    layout: this.intersectLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                        { binding: 1, resource: { buffer: this.rayBuffer } },
                        { binding: 2, resource: { buffer: this.hitBuffer } },
                        { binding: 3, resource: { buffer: bvhData.nodeBuffer } },
                        { binding: 4, resource: { buffer: bvhData.positionBuffer } },
                        { binding: 5, resource: { buffer: bvhData.indexBuffer } },
                        { binding: 6, resource: { buffer: scene.globalMaterialBuffer } },
                        { binding: 7, resource: scene.baseColorTexArrayView },
                        { binding: 8, resource: baseColorSampler },
                        { binding: 9, resource: { buffer: bvhData.uvBuffer } },
                    ]
                });
                const pass = encoder.beginComputePass({ label: `WPT Intersect b${bounce}` });
                pass.setPipeline(this.intersectPipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(r1D, 1, 1);
                pass.end();
            }

            // Pass 2: Shade + NEE (NEE skipped for bounce 0 when ReSTIR active)
            {
                const bg = dev.createBindGroup({
                    layout: this.shadeLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                        { binding: 1, resource: { buffer: this.ptUniformBuffer } },
                        { binding: 2, resource: { buffer: this.rayBuffer } },
                        { binding: 3, resource: { buffer: this.hitBuffer } },
                        { binding: 4, resource: { buffer: this.shadowBuffer } },
                        { binding: 5, resource: { buffer: this.accumWorkBuffer } },
                        { binding: 6, resource: { buffer: scene.globalMaterialBuffer } },
                        { binding: 7, resource: { buffer: this.stage.sunLightBuffer } },
                        { binding: 8, resource: scene.baseColorTexArrayView },
                        { binding: 9, resource: baseColorSampler },
                        { binding: 10, resource: scene.normalMapTexArrayView },
                        { binding: 11, resource: scene.mrTexArrayView },
                        { binding: 12, resource: { buffer: bvhData.uvBuffer } },
                        { binding: 13, resource: { buffer: bvhData.normalBuffer } },
                        { binding: 14, resource: { buffer: bvhData.tangentBuffer } },
                    ]
                });
                
                const nrcBg = dev.createBindGroup({
                    layout: this.shadeNRCLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.stage.nrc.nrcUniformBuffer } },
                        { binding: 1, resource: { buffer: this.stage.nrc.weightsBuffer } },
                        { binding: 2, resource: { buffer: this.stage.nrc.sampleCounterBuffer } },
                        { binding: 3, resource: { buffer: this.ptNRCTrainDataBuffer } },
                    ]
                });

                const pass = encoder.beginComputePass({ label: `WPT Shade b${bounce}` });
                pass.setPipeline(this.shadePipeline);
                pass.setBindGroup(0, bg);
                pass.setBindGroup(1, nrcBg);
                pass.dispatchWorkgroups(r1D, 1, 1);
                pass.end();
            }

            // ============================================================
            // ReSTIR DI — only at bounce 0
            // ============================================================
            if (bounce === 0 && this.config.restirEnabled) {
                // Pass A: Initial Candidate Generation (RIS)
                {
                    const bg = dev.createBindGroup({
                        layout: this.restirInitialLayout,
                        entries: [
                            { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                            { binding: 1, resource: { buffer: this.restirUniformBuffer } },
                            { binding: 2, resource: { buffer: this.rayBuffer } },
                            { binding: 3, resource: { buffer: this.hitBuffer } },
                            { binding: 4, resource: { buffer: scene.globalMaterialBuffer } },
                            { binding: 5, resource: { buffer: this.stage.sunLightBuffer } },
                            { binding: 6, resource: { buffer: this.reservoirBufferA } },
                            { binding: 7, resource: { buffer: bvhData.emissiveIndexBuffer } },
                            { binding: 8, resource: { buffer: bvhData.positionBuffer } },
                            { binding: 9, resource: { buffer: bvhData.normalBuffer } },
                            { binding: 10, resource: { buffer: bvhData.uvBuffer } },
                            { binding: 11, resource: { buffer: this.pixelDataBuffer } },
                        ]
                    });
                    const pass = encoder.beginComputePass({ label: 'ReSTIR Initial' });
                    pass.setPipeline(this.restirInitialPipeline);
                    pass.setBindGroup(0, bg);
                    pass.dispatchWorkgroups(r1D, 1, 1);
                    pass.end();
                }

                // Pass B: Temporal Resampling
                {
                    const bg = dev.createBindGroup({
                        layout: this.restirTemporalLayout,
                        entries: [
                            { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                            { binding: 1, resource: { buffer: this.restirUniformBuffer } },
                            { binding: 2, resource: { buffer: this.camera.uniformsBuffer } },
                            { binding: 3, resource: { buffer: this.prevCameraBuffer } },
                            { binding: 4, resource: { buffer: this.reservoirBufferA } },
                            { binding: 5, resource: { buffer: this.prevReservoirBuffer } },
                            { binding: 6, resource: { buffer: this.rayBuffer } },
                            { binding: 7, resource: { buffer: this.hitBuffer } },
                            { binding: 8, resource: { buffer: bvhData.normalBuffer } },
                            { binding: 9, resource: { buffer: this.prevPixelDataBuffer } },
                        ]
                    });
                    const pass = encoder.beginComputePass({ label: 'ReSTIR Temporal' });
                    pass.setPipeline(this.restirTemporalPipeline);
                    pass.setBindGroup(0, bg);
                    pass.dispatchWorkgroups(r1D, 1, 1);
                    pass.end();
                }

                // Pass C: Spatial Resampling (A → B)
                {
                    const bg = dev.createBindGroup({
                        layout: this.restirSpatialLayout,
                        entries: [
                            { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                            { binding: 1, resource: { buffer: this.restirUniformBuffer } },
                            { binding: 2, resource: { buffer: this.reservoirBufferA } },   // in
                            { binding: 3, resource: { buffer: this.reservoirBufferB } },   // out
                            { binding: 4, resource: { buffer: this.rayBuffer } },
                            { binding: 5, resource: { buffer: this.hitBuffer } },
                            { binding: 6, resource: { buffer: bvhData.normalBuffer } },
                            { binding: 7, resource: { buffer: this.pixelDataBuffer } },
                        ]
                    });
                    const pass = encoder.beginComputePass({ label: 'ReSTIR Spatial' });
                    pass.setPipeline(this.restirSpatialPipeline);
                    pass.setBindGroup(0, bg);
                    pass.dispatchWorkgroups(r1D, 1, 1);
                    pass.end();
                }

                // Pass D: ReSTIR Shade → Shadow Ray (reads from B)
                {
                    const bg = dev.createBindGroup({
                        layout: this.restirShadeLayout,
                        entries: [
                            { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                            { binding: 1, resource: { buffer: this.restirUniformBuffer } },
                            { binding: 2, resource: { buffer: this.camera.uniformsBuffer } },
                            { binding: 3, resource: { buffer: this.rayBuffer } },
                            { binding: 4, resource: { buffer: this.hitBuffer } },
                            { binding: 5, resource: { buffer: this.reservoirBufferB } },
                            { binding: 6, resource: { buffer: this.shadowBuffer } },
                            { binding: 7, resource: { buffer: this.accumWorkBuffer } },
                            { binding: 8, resource: { buffer: scene.globalMaterialBuffer } },
                            { binding: 9, resource: scene.baseColorTexArrayView },
                            { binding: 10, resource: baseColorSampler },
                            { binding: 11, resource: scene.mrTexArrayView },
                            { binding: 12, resource: { buffer: bvhData.uvBuffer } },
                            { binding: 13, resource: { buffer: bvhData.normalBuffer } },
                            { binding: 14, resource: { buffer: bvhData.tangentBuffer } },
                            { binding: 15, resource: scene.normalMapTexArrayView },
                            { binding: 16, resource: { buffer: this.pixelDataBuffer } },
                        ]
                    });
                    const pass = encoder.beginComputePass({ label: 'ReSTIR Shade' });
                    pass.setPipeline(this.restirShadePipeline);
                    pass.setBindGroup(0, bg);
                    pass.dispatchWorkgroups(r1D, 1, 1);
                    pass.end();
                }
            }

            // Pass 3: Shadow Test (with alpha-cutout transparency)
            {
                const bg = dev.createBindGroup({
                    layout: this.shadowTestLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                        { binding: 1, resource: { buffer: this.shadowBuffer } },
                        { binding: 2, resource: { buffer: this.accumWorkBuffer } },
                        { binding: 3, resource: { buffer: bvhData.nodeBuffer } },
                        { binding: 4, resource: { buffer: bvhData.positionBuffer } },
                        { binding: 5, resource: { buffer: bvhData.indexBuffer } },
                        { binding: 6, resource: { buffer: scene.globalMaterialBuffer } },
                        { binding: 7, resource: scene.baseColorTexArrayView },
                        { binding: 8, resource: baseColorSampler },
                        { binding: 9, resource: { buffer: bvhData.uvBuffer } },
                    ]
                });
                const pass = encoder.beginComputePass({ label: `WPT Shadow b${bounce}` });
                pass.setPipeline(this.shadowTestPipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(r1D, 1, 1);
                pass.end();
            }

            // Pass 4: Miss (env sampling for rays that escaped this bounce)
            {
                const bg = dev.createBindGroup({
                    layout: this.missLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                        { binding: 1, resource: { buffer: this.rayBuffer } },
                        { binding: 2, resource: { buffer: this.hitBuffer } },
                        { binding: 3, resource: { buffer: this.accumWorkBuffer } },
                        { binding: 4, resource: this.stage.environment.envCubemapView },
                        { binding: 5, resource: this.stage.environment.envSampler },
                    ]
                });
                const pass = encoder.beginComputePass({ label: `WPT Miss b${bounce}` });
                pass.setPipeline(this.missPipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(r1D, 1, 1);
                pass.end();
            }
        }

        // ---- ReSTIR: Copy current reservoir to prev for next frame ----
        if (this.config.restirEnabled) {
            // Copy reservoirB (final spatial output) → prevReservoir for next frame's temporal
            encoder.copyBufferToBuffer(
                this.reservoirBufferB, 0,
                this.prevReservoirBuffer, 0,
                this.totalPixels * 64
            );
            // Copy pixel data → prevPixelData for next frame's temporal
            encoder.copyBufferToBuffer(
                this.pixelDataBuffer, 0,
                this.prevPixelDataBuffer, 0,
                this.totalPixels * 16
            );
            // Copy current camera → prevCamera for next frame's reprojection
            renderer.device.queue.writeBuffer(this.prevCameraBuffer, 0, this.camera.uniforms.buffer);
        }

        // ---- Pass: NRC Collect & Train (If enabled) ----
        if (this.stage.nrc.enabled) {
            const bg = dev.createBindGroup({
                layout: this.nrcCollectLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.stage.nrc.nrcUniformBuffer } },
                    { binding: 1, resource: { buffer: this.ptNRCTrainDataBuffer } },
                    { binding: 2, resource: { buffer: this.accumWorkBuffer } },
                    { binding: 3, resource: { buffer: this.stage.nrc.sampleCounterBuffer } },
                    { binding: 4, resource: { buffer: this.stage.nrc.trainingSamplesBuffer } },
                ]
            });
            const pass = encoder.beginComputePass({ label: 'WPT NRC Collect' });
            pass.setPipeline(this.nrcCollectPipeline);
            pass.setBindGroup(0, bg);
            // Dispatch 64 threads, max 4096 samples = 64 workgroups
            pass.dispatchWorkgroups(Math.ceil(NRC.MAX_TRAINING_SAMPLES / 64), 1, 1);
            pass.end();

            // Run MLP update!
            this.stage.nrc.trainOnly(encoder);
        }

        // ---- Pass: Accumulate (add frame to persistent sum) ----
        {
            const bg = dev.createBindGroup({
                layout: this.accumulateLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                    { binding: 1, resource: { buffer: this.accumWorkBuffer } },
                    { binding: 2, resource: { buffer: this.sampleSumBuffer } },
                ]
            });
            const pass = encoder.beginComputePass({ label: 'WPT Accumulate' });
            pass.setPipeline(this.accumulatePipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(rW, rH, 1);
            pass.end();
        }

        // ---- Pass: Tonemap Blit → Canvas ----
        {
            const canvasView = renderer.context.getCurrentTexture().createView();
            const bg = dev.createBindGroup({
                layout: this.tonemapLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.ptUniformBuffer } },
                    { binding: 1, resource: { buffer: this.sampleSumBuffer } },
                ]
            });
            const pass = encoder.beginRenderPass({
                label: 'WPT Tonemap Blit',
                colorAttachments: [{ view: canvasView, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }]
            });
            pass.setPipeline(this.tonemapPipeline);
            pass.setBindGroup(0, bg);
            pass.draw(3);
            pass.end();
        }

        dev.queue.submit([encoder.finish()]);
    }
}
