import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';
import { RenderGraph, ResourceHandle } from '../../engine/RenderGraph';

export interface SSAOPassDeps {
    cameraBuffer: GPUBuffer;
    ssaoUniformsBuffer: GPUBuffer;
}

export class SSAOPass {
    private ssaoPipeline: GPURenderPipeline;
    private blurPipeline: GPURenderPipeline;

    private deps: SSAOPassDeps;

    // Textures entirely managed by RenderGraph now

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

    addToGraph(graph: RenderGraph, enabled: boolean, hizHandle: ResourceHandle, normalHandle: ResourceHandle): ResourceHandle {
        const blurHandle = graph.createTexture("SSAO_Blur_Res", {
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        if (!enabled) {
            graph.addRenderPass("SSAO Clear")
                .addColorAttachment(blurHandle, { clearValue: [1, 1, 1, 1] })
                .execute((clearPass, pass) => {
                    // Empty execution, clear is handled by attachment loadOp
                });
            return blurHandle;
        }

        const ssaoHandle = graph.createTexture("SSAO_Raw", {
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        graph.addRenderPass("SSAO Generation")
            .readTexture(hizHandle)
            .readTexture(normalHandle)
            .addColorAttachment(ssaoHandle, { clearValue: [1, 1, 1, 1] })
            .execute((ssaoPass, pass) => {
                ssaoPass.setPipeline(this.ssaoPipeline);
                
                const mainBG = renderer.device.createBindGroup({
                    label: "ssao main bgl",
                    layout: this.ssaoPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.deps.cameraBuffer } },
                        { binding: 1, resource: pass.getTextureView(hizHandle) },
                        { binding: 2, resource: pass.getTextureView(normalHandle) },
                        { binding: 3, resource: { buffer: this.deps.ssaoUniformsBuffer } },
                    ]
                });

                ssaoPass.setBindGroup(0, mainBG);
                ssaoPass.draw(3);
            });

        graph.addRenderPass("SSAO Blur")
            .readTexture(ssaoHandle)
            .addColorAttachment(blurHandle, { clearValue: [1, 1, 1, 1] })
            .execute((blurPass, pass) => {
                blurPass.setPipeline(this.blurPipeline);

                let blurBindGroup = renderer.device.createBindGroup({
                    label: "ssao blur bgl",
                    layout: this.blurPipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: pass.getTextureView(ssaoHandle) }]
                });

                blurPass.setBindGroup(0, blurBindGroup);
                blurPass.draw(3);
            });

        return blurHandle;
    }
}
