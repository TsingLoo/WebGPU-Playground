import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';
import { BindGroupCache } from '../../engine/RenderTexManager';
import { RenderGraph, ResourceHandle } from '../../engine/RenderGraph';
import { VSM } from '../../stage/vsm';

export interface VolumetricPassDeps {
    cameraBuffer: GPUBuffer;
    sunLightBuffer: GPUBuffer;
    vsm: VSM;
}

export class VolumetricPass {
    private generatorPipeline: GPURenderPipeline;
    private compositePipeline: GPURenderPipeline;
    private deps: VolumetricPassDeps;

    constructor(deps: VolumetricPassDeps) {
        this.deps = deps;

        // Generator pipeline
        const genBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "volumetric lighting bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.generatorPipeline = renderer.device.createRenderPipeline({
            label: "volumetric lighting pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [genBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.volumetricLightingVertSrc }), entryPoint: "main" },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricLightingFragSrc }),
                entryPoint: "main",
                targets: [{ format: "rgba16float" }]
            }
        });
        // Generator pipeline bindings are computed dynamically in execute

        // Composite pipeline (additive blend onto canvas)
        const compBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "volumetric composite bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.compositePipeline = renderer.device.createRenderPipeline({
            label: "volumetric composite pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [compBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.volumetricLightingVertSrc }), entryPoint: "main" },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricCompositeFragSrc }),
                entryPoint: "main",
                targets: [{
                    format: renderer.canvasFormat,
                    blend: {
                        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
                        alpha: { operation: 'add', srcFactor: 'zero', dstFactor: 'one' }
                    }
                }]
            }
        });
        // Composite pipeline bindings are computed dynamically in execute
    }

    addToGraph(graph: RenderGraph, canvasHandle: ResourceHandle, depthHandle: ResourceHandle) {
        const volumetricTextureHandle = graph.createTexture("VolumetricHalfRes", {
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            scale: 0.5
        });

        graph.addPass("Volumetric Lighting Generator")
            .readTexture(depthHandle)
            .writeTexture(volumetricTextureHandle)
            .execute((encoder, pass) => {
                const generatorBindGroup = BindGroupCache.get({
                    label: "volumetric lighting bind group",
                    layout: this.generatorPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.deps.cameraBuffer } },
                        { binding: 1, resource: pass.getTextureView(depthHandle) },
                        { binding: 2, resource: { buffer: this.deps.sunLightBuffer } },
                        { binding: 3, resource: this.deps.vsm.physicalAtlasView },
                        { binding: 4, resource: { buffer: this.deps.vsm.pageTableBuffer } },
                        { binding: 5, resource: { buffer: this.deps.vsm.vsmUniformBuffer } },
                    ]
                });

                // Generate volumetric scattering at half resolution
                const genPass = encoder.beginRenderPass({
                    label: "Volumetric Lighting Generator Pass",
                    colorAttachments: [{ view: pass.getTextureView(volumetricTextureHandle), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }]
                });
                genPass.setPipeline(this.generatorPipeline);
                genPass.setBindGroup(0, generatorBindGroup);
                genPass.draw(3);
                genPass.end();
            });

        graph.addPass("Volumetric Composite")
            .readTexture(volumetricTextureHandle)
            .readTexture(depthHandle)
            .writeTexture(canvasHandle)
            .execute((encoder, pass) => {
                const compositeBindGroup = BindGroupCache.get({
                    label: "volumetric composite bind group",
                    layout: this.compositePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: pass.getTextureView(volumetricTextureHandle) },
                        { binding: 1, resource: pass.getTextureView(depthHandle) },
                        { binding: 2, resource: { buffer: this.deps.cameraBuffer } },
                    ]
                });

                // Composite onto canvas with additive blending
                const compPass = encoder.beginRenderPass({
                    label: "Volumetric Composite Pass",
                    colorAttachments: [{ view: pass.getTextureView(canvasHandle), loadOp: "load", storeOp: "store" }]
                });
                compPass.setPipeline(this.compositePipeline);
                compPass.setBindGroup(0, compositeBindGroup);
                compPass.draw(3);
                compPass.end();
            });
    }
}
