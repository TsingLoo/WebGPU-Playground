/**
 * path_tracing_wavefront.ts
 *
 * Wavefront Path Tracing Renderer for WebGPU.
 *
 * Architecture:
 *   Each frame = N bounce iterations of:
 *     1. ray_gen (bounce 0 only)   → rayBuffer[]
 *     2. intersect                 → hitBuffer[]
 *     3. shade + NEE               → new rayBuffer[], shadowBuffer[], accumBuffer[]
 *     4. shadow_test               → accumBuffer[] (direct light contribution)
 *     5. miss (env)                → accumBuffer[]
 *   Then:
 *     6. accumulate  → adds accumBuffer to sampleSumBuffer (persistent), clears accumBuffer
 *     7. tonemap blit → reads sampleSumBuffer / sample_count → canvas
 */

import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { Renderer } from '../renderer';

// ============================================================
// PTUniforms layout (matches pt_common.wgsl PTUniforms struct)
// 8 × 4 = 32 bytes
// ============================================================
interface PTConfig {
    maxBounces: number;
    clampRadiance: number;
    pixelScale: number;
}

export class WavefrontPathTracingRenderer extends Renderer {

    // ------------ Configuration ------------
    config: PTConfig = {
        maxBounces: 4,
        clampRadiance: 10.0,
        pixelScale: 1.0,
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
    private tonemapPipeline!: GPURenderPipeline;

    // ------------ Bind Group Layouts ------------
    private rayGenLayout!: GPUBindGroupLayout;
    private intersectLayout!: GPUBindGroupLayout;
    private shadeLayout!: GPUBindGroupLayout;
    private shadowTestLayout!: GPUBindGroupLayout;
    private missLayout!: GPUBindGroupLayout;
    private accumulateLayout!: GPUBindGroupLayout;
    private tonemapLayout!: GPUBindGroupLayout;

    constructor(stage: Stage) {
        super(stage);
        this.initGPUResources();
        this.initPipelines();
        console.log('[WPT] Wavefront Path Tracing Renderer initialized');
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

        const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

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

        this.uploadPTUniforms();
    }

    private initPipelines() {
        const dev = renderer.device;

        // ---- Bind Group Layouts ----

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

        this.shadowTestLayout = dev.createBindGroupLayout({
            label: 'PT ShadowTest Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // pt
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // shadow_buffer
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // accum_buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh nodes
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh pos
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh indices
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

        this.tonemapLayout = dev.createBindGroupLayout({
            label: 'PT Tonemap Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },          // pt
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // sample_sum_buf
            ]
        });

        // ---- Compute Pipelines ----
        const mkCompute = (label: string, code: string, layout: GPUBindGroupLayout) =>
            dev.createComputePipeline({
                label,
                layout: dev.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: dev.createShaderModule({ label, code }), entryPoint: 'main' }
            });

        this.rayGenPipeline     = mkCompute('WPT RayGen',     shaders.ptRayGenSrc,     this.rayGenLayout);
        this.intersectPipeline  = mkCompute('WPT Intersect',  shaders.ptIntersectSrc,  this.intersectLayout);
        this.shadePipeline      = mkCompute('WPT Shade',      shaders.ptShadeSrc,      this.shadeLayout);
        this.shadowTestPipeline = mkCompute('WPT ShadowTest', shaders.ptShadowTestSrc, this.shadowTestLayout);
        this.missPipeline       = mkCompute('WPT Miss',       shaders.ptMissSrc,       this.missLayout);
        this.accumulatePipeline = mkCompute('WPT Accumulate', shaders.ptAccumulateSrc, this.accumulateLayout);

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
        u32[7] = 0;
        renderer.device.queue.writeBuffer(this.ptUniformBuffer, 0, buf);
    }

    // ============================================================
    // Main Draw Loop
    // ============================================================
    protected override draw(): void {
        const dev   = renderer.device;
        const scene = this.stage.scene;

        if (this.checkCameraChanged()) {
            this.resetAccumulation();
        }

        this.sampleCount++;
        this.uploadPTUniforms();

        const bvhData = scene.bvhData;
        const encoder = dev.createCommandEncoder({ label: 'WPT Frame' });

        const rW  = Math.ceil(this.renderWidth  / 8);
        const rH  = Math.ceil(this.renderHeight / 8);
        const r1D = Math.ceil(this.totalPixels  / 64);

        const baseColorSampler = dev.createSampler({
            magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'repeat', addressModeV: 'repeat'
        });

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

            // Pass 1: Intersect
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
                    ]
                });
                const pass = encoder.beginComputePass({ label: `WPT Intersect b${bounce}` });
                pass.setPipeline(this.intersectPipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(r1D, 1, 1);
                pass.end();
            }

            // Pass 2: Shade + NEE
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
                const pass = encoder.beginComputePass({ label: `WPT Shade b${bounce}` });
                pass.setPipeline(this.shadePipeline);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(r1D, 1, 1);
                pass.end();
            }

            // Pass 3: Shadow Test
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
