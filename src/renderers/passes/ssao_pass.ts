import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export interface SSAOPassDeps {
    cameraBuffer: GPUBuffer;
    gBufferPositionView: GPUTextureView;
    gBufferNormalView: GPUTextureView;
    ssaoUniformsBuffer: GPUBuffer;
}

export class SSAOPass {
    private ssaoTexture: GPUTexture;
    private ssaoTextureView: GPUTextureView;

    private blurredTexture: GPUTexture;
    private _blurredTextureView: GPUTextureView;

    private ssaoPipeline: GPURenderPipeline;
    private ssaoBindGroup: GPUBindGroup;

    private blurPipeline: GPURenderPipeline;
    private blurBindGroup: GPUBindGroup;

    /** Exposed for shading bind groups that need the AO result */
    get blurredTextureView(): GPUTextureView {
        return this._blurredTextureView;
    }

    constructor(deps: SSAOPassDeps) {
        const size = [renderer.canvas.width, renderer.canvas.height];

        this.ssaoTexture = renderer.device.createTexture({
            label: "ssao texture",
            size, format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.ssaoTextureView = this.ssaoTexture.createView();

        this.blurredTexture = renderer.device.createTexture({
            label: "ssao blurred texture",
            size, format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this._blurredTextureView = this.blurredTexture.createView();

        // SSAO generation
        const ssaoBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssao bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.ssaoBindGroup = renderer.device.createBindGroup({
            layout: ssaoBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: deps.cameraBuffer } },
                { binding: 1, resource: deps.gBufferPositionView },
                { binding: 2, resource: deps.gBufferNormalView },
                { binding: 3, resource: { buffer: deps.ssaoUniformsBuffer } },
            ]
        });
        this.ssaoPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [ssaoBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssaoFragSrc }), entryPoint: "main", targets: [{ format: "r16float" }] }
        });

        // SSAO blur
        const blurBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssao blur bgl",
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }]
        });
        this.blurBindGroup = renderer.device.createBindGroup({
            layout: blurBindGroupLayout,
            entries: [{ binding: 0, resource: this.ssaoTextureView }]
        });
        this.blurPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [blurBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssaoBlurFragSrc }), entryPoint: "main", targets: [{ format: "r16float" }] }
        });
    }

    execute(encoder: GPUCommandEncoder, enabled: boolean = true) {
        if (!enabled) {
            // Fast clear to white (no occlusion) so the shading passes don't read garbage old frames
            const clearPass = encoder.beginRenderPass({
                label: "SSAO disabled clear pass",
                colorAttachments: [{
                    view: this._blurredTextureView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
                }]
            });
            clearPass.end();
            return;
        }

        // SSAO generation
        const ssaoPass = encoder.beginRenderPass({
            label: "SSAO pass",
            colorAttachments: [{
                view: this.ssaoTextureView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
            }]
        });
        ssaoPass.setPipeline(this.ssaoPipeline);
        ssaoPass.setBindGroup(0, this.ssaoBindGroup);
        ssaoPass.draw(3);
        ssaoPass.end();

        // SSAO blur
        const blurPass = encoder.beginRenderPass({
            label: "SSAO blur pass",
            colorAttachments: [{
                view: this._blurredTextureView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
            }]
        });
        blurPass.setPipeline(this.blurPipeline);
        blurPass.setBindGroup(0, this.blurBindGroup);
        blurPass.draw(3);
        blurPass.end();
    }
}
