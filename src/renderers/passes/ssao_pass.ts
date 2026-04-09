import * as renderer from '../../renderer';
import { RenderTexManager, RenderResource, BindGroupCache } from '../../engine/RenderTexManager';
import * as shaders from '../../shaders/shaders';

export interface SSAOPassDeps {
    cameraBuffer: GPUBuffer;
    ssaoUniformsBuffer: GPUBuffer;
}

export class SSAOPass {
    private ssaoPipeline: GPURenderPipeline;
    private blurPipeline: GPURenderPipeline;

    private deps: SSAOPassDeps;

    /** Exposed for shading bind groups that need the AO result */
    get blurredTextureView(): GPUTextureView {
        return RenderTexManager.getTextureView(RenderResource.TransientFull_R16F_B, {
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
    }

    constructor(deps: SSAOPassDeps) {
        this.deps = deps;

        // Layouts
        const ssaoBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssao bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });

        this.ssaoPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [ssaoBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssaoFragSrc }), entryPoint: "main", targets: [{ format: "r16float" }] }
        });

        const blurBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssao blur bgl",
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }]
        });

        this.blurPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [blurBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssaoBlurFragSrc }), entryPoint: "main", targets: [{ format: "r16float" }] }
        });
    }

    execute(encoder: GPUCommandEncoder, enabled: boolean, hizView: GPUTextureView, normalView: GPUTextureView) {
        const ssaoView = RenderTexManager.getTextureView(RenderResource.TransientFull_R16F_A, {
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        const blurView = this.blurredTextureView;
        let blurBindGroup: GPUBindGroup;
        if (!enabled) {
            // Fast clear to white (no occlusion) so the shading passes don't read garbage old frames
            const clearPass = encoder.beginRenderPass({
                label: "SSAO disabled clear pass",
                colorAttachments: [{
                    view: blurView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
                }]
            });
            clearPass.end();
            return;
        }

        // SSAO generation
        const ssaoPass = encoder.beginRenderPass({
            label: "SSAO pass",
            colorAttachments: [{
                view: ssaoView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
            }]
        });
        ssaoPass.setPipeline(this.ssaoPipeline);
        
        // Use synchronous execution by assuming BindGroupCache import holds
        const mainBG = BindGroupCache.get({
            label: "ssao main bgl",
            layout: this.ssaoPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.deps.cameraBuffer } },
                { binding: 1, resource: hizView },
                { binding: 2, resource: normalView },
                { binding: 3, resource: { buffer: this.deps.ssaoUniformsBuffer } },
            ]
        });
        
        blurBindGroup = BindGroupCache.get({
            label: "ssao blur bgl",
            layout: this.blurPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: ssaoView }]
        });

        ssaoPass.setBindGroup(0, mainBG);
        ssaoPass.draw(3);
        ssaoPass.end();

        // SSAO blur
        const blurPass = encoder.beginRenderPass({
            label: "SSAO blur pass",
            colorAttachments: [{
                view: blurView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
            }]
        });
        blurPass.setPipeline(this.blurPipeline);
        blurPass.setBindGroup(0, blurBindGroup);
        blurPass.draw(3);
        blurPass.end();
    }
}
