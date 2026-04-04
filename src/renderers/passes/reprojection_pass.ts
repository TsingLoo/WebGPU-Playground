import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export interface ReprojectionPassDeps {
    reprojUniformBuffer: GPUBuffer;
}

/**
 * Frame Warp Reprojection Pass
 * 
 * Maintains history buffers (color + depth from previous frame).
 * Each frame:
 *   1. executeWarpAndBlit() — warp history to latest camera view → blit to canvas
 *   2. copyToHistory() — copy current frame's scene color+depth into history for next frame
 */
export class ReprojectionPass {
    // History buffers (stores previous frame's rendered output)
    private historyColorTexture: GPUTexture;
    private historyColorView: GPUTextureView;
    private historyDepthTexture: GPUTexture;
    private historyDepthView: GPUTextureView;

    // Warp output (what gets blitted to canvas)
    private outputTexture: GPUTexture;
    private _outputTextureView: GPUTextureView;

    private computePipeline: GPUComputePipeline;
    private bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup;

    // Blit pipeline: warp output → canvas
    private blitSampler: GPUSampler;
    private blitBindGroupLayout: GPUBindGroupLayout;
    private blitBindGroup: GPUBindGroup;
    private blitPipeline: GPURenderPipeline;

    // Depth copy pipeline: depth24plus → r32float
    private depthCopyBindGroupLayout: GPUBindGroupLayout;
    private depthCopyPipeline: GPURenderPipeline;

    get outputTextureView(): GPUTextureView {
        return this._outputTextureView;
    }

    constructor(deps: ReprojectionPassDeps) {
        const width = renderer.canvas.width;
        const height = renderer.canvas.height;

        // History color (stores the fully-rendered previous frame, same format as canvas)
        this.historyColorTexture = renderer.device.createTexture({
            label: "frame warp history color",
            size: [width, height],
            format: renderer.canvasFormat,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.historyColorView = this.historyColorTexture.createView();

        // History depth (raw depth from previous frame, stored as r32float)
        this.historyDepthTexture = renderer.device.createTexture({
            label: "frame warp history depth",
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.historyDepthView = this.historyDepthTexture.createView();

        // Output texture (the warped result displayed to screen)
        this.outputTexture = renderer.device.createTexture({
            label: "frame warp output",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this._outputTextureView = this.outputTexture.createView();

        // ============================
        // Warp compute pipeline
        // ============================
        this.bindGroupLayout = renderer.device.createBindGroupLayout({
            label: "reprojection bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ]
        });

        this.bindGroup = renderer.device.createBindGroup({
            label: "reprojection bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: deps.reprojUniformBuffer } },
                { binding: 1, resource: this.historyColorView },
                { binding: 2, resource: this.historyDepthView },
                { binding: 3, resource: this._outputTextureView },
            ]
        });

        this.computePipeline = renderer.device.createComputePipeline({
            label: "reprojection compute pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            compute: {
                module: renderer.device.createShaderModule({ code: shaders.reprojectionComputeSrc }),
                entryPoint: "main",
            }
        });

        // ============================
        // Blit pipeline: warped output → canvas
        // ============================
        this.blitSampler = renderer.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        this.blitBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "warp blit bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            ]
        });

        this.blitBindGroup = renderer.device.createBindGroup({
            layout: this.blitBindGroupLayout,
            entries: [
                { binding: 0, resource: this._outputTextureView },
                { binding: 1, resource: this.blitSampler },
            ]
        });

        this.blitPipeline = renderer.device.createRenderPipeline({
            label: "warp blit pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.blitBindGroupLayout] }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }),
                entryPoint: "main",
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitFragSrc }),
                entryPoint: "main",
                targets: [{ format: renderer.canvasFormat }],
            }
        });

        // ============================
        // Depth copy pipeline: depth24plus → r32float
        // ============================
        this.depthCopyBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "depth copy bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
            ]
        });

        this.depthCopyPipeline = renderer.device.createRenderPipeline({
            label: "depth copy pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.depthCopyBindGroupLayout] }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }),
                entryPoint: "main",
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.depthCopyFragSrc }),
                entryPoint: "main",
                targets: [{ format: "r32float" }],
            }
        });
    }

    /**
     * Step 1: Warp the history buffer to the latest camera view and blit to canvas.
     * Called at the START of the frame, before any heavy rendering.
     */
    executeWarpAndBlit(encoder: GPUCommandEncoder, canvasTextureView: GPUTextureView) {
        // Dispatch compute shader to warp history → output
        const warpPass = encoder.beginComputePass({ label: "Frame Warp compute" });
        warpPass.setPipeline(this.computePipeline);
        warpPass.setBindGroup(0, this.bindGroup);
        const workgroupsX = Math.ceil(renderer.canvas.width / 8);
        const workgroupsY = Math.ceil(renderer.canvas.height / 8);
        warpPass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        warpPass.end();

        // Blit warped output to canvas
        const blitPass = encoder.beginRenderPass({
            label: "Frame Warp blit to canvas",
            colorAttachments: [{
                view: canvasTextureView,
                clearValue: [0, 0, 0, 1],
                loadOp: "clear",
                storeOp: "store",
            }]
        });
        blitPass.setPipeline(this.blitPipeline);
        blitPass.setBindGroup(0, this.blitBindGroup);
        blitPass.draw(3);
        blitPass.end();
    }

    /**
     * Step 2: Copy the just-rendered scene color and depth into history buffers.
     * Called AFTER the full rendering pipeline completes.
     * 
     * @param sceneColorTexture - the rgba8unorm intermediate scene color texture
     * @param depthTextureView - the depth24plus depth buffer view
     */
    copyToHistory(encoder: GPUCommandEncoder, sceneColorTexture: GPUTexture, depthTextureView: GPUTextureView) {
        // Copy scene color → history color (same format, direct copy)
        encoder.copyTextureToTexture(
            { texture: sceneColorTexture },
            { texture: this.historyColorTexture },
            [renderer.canvas.width, renderer.canvas.height]
        );

        // Copy depth24plus → r32float via a render pass (format conversion)
        const depthCopyBG = renderer.device.createBindGroup({
            layout: this.depthCopyBindGroupLayout,
            entries: [{ binding: 0, resource: depthTextureView }]
        });

        const depthCopyPass = encoder.beginRenderPass({
            label: "depth copy to history",
            colorAttachments: [{
                view: this.historyDepthView,
                loadOp: "clear",
                clearValue: [0, 0, 0, 0],
                storeOp: "store",
            }]
        });
        depthCopyPass.setPipeline(this.depthCopyPipeline);
        depthCopyPass.setBindGroup(0, depthCopyBG);
        depthCopyPass.draw(3);
        depthCopyPass.end();
    }
}
