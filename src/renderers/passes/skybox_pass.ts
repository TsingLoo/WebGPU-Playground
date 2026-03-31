import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export interface SkyboxPassDeps {
    cameraBuffer: GPUBuffer;
    envCubemapView: GPUTextureView;
    envSampler: GPUSampler;
}

export class SkyboxPass {
    private pipeline: GPURenderPipeline;
    private bindGroup: GPUBindGroup;

    constructor(deps: SkyboxPassDeps) {
        const bindGroupLayout = renderer.device.createBindGroupLayout({
            label: "skybox bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });
        this.bindGroup = renderer.device.createBindGroup({
            label: "skybox bind group",
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: deps.cameraBuffer } },
                { binding: 1, resource: deps.envCubemapView },
                { binding: 2, resource: deps.envSampler }
            ]
        });
        this.pipeline = renderer.device.createRenderPipeline({
            label: "skybox pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.skyboxVertSrc }), entryPoint: "main" },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.skyboxFragSrc }),
                entryPoint: "main",
                targets: [{ format: renderer.canvasFormat }]
            },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" }
        });
    }

    execute(encoder: GPUCommandEncoder, canvasTextureView: GPUTextureView, depthTextureView: GPUTextureView) {
        const pass = encoder.beginRenderPass({
            label: "Skybox Pass",
            colorAttachments: [{ view: canvasTextureView, loadOp: "load", storeOp: "store" }],
            depthStencilAttachment: { view: depthTextureView, depthLoadOp: "load", depthStoreOp: "store" }
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(3);
        pass.end();
    }
}
