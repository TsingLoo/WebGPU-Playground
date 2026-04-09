import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';
import { RenderGraph, ResourceHandle } from '../../engine/RenderGraph';

export interface SSRPassDeps {
    cameraBuffer: GPUBuffer;
    ssrUniformsBuffer: GPUBuffer;
}

export class SSRPass {
    private ssrPipeline: GPURenderPipeline;
    private compositePipeline: GPURenderPipeline;
    private deps: SSRPassDeps;

    constructor(deps: SSRPassDeps) {
        this.deps = deps;
        // SSR generation
        const ssrBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssr bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });

        // The camera is bound at group 0, common for our fullscreen passes because it's baked into standard layouts.
        // Wait, cameraBindGroup isn't passed in layout, but standard passes use geometryBindGroupLayout for camera
        // So we create pipeline layout with [cameraBindGroupLayout, ssrBindGroupLayout]
        
        let cameraBGL = renderer.device.createBindGroupLayout({
            label: "camera bgl",
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }]
        });

        this.ssrPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [cameraBGL, ssrBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssrFragSrc }), entryPoint: "main", targets: [{ format: "rgba16float" }] }
        });

        // Composite pass
        const compositeBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssr composite bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }, // scene color
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }, // ssr hit
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }, // albedo
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }, // specular
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }, // normal
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },                // depth
            ]
        });

        this.compositePipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [cameraBGL, compositeBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssrCompositeFragSrc }), entryPoint: "main", targets: [{ format: "bgra8unorm" }] } // outputs to canvas directly
        });
    }

    addToGraph(graph: RenderGraph, enabled: boolean, 
            cameraBindGroup: GPUBindGroup, 
            sceneColorHandle: ResourceHandle, 
            albedoHandle: ResourceHandle,
            normalHandle: ResourceHandle,
            specularHandle: ResourceHandle,
            depthHandle: ResourceHandle,
            hizHandle: ResourceHandle,
            canvasHandle: ResourceHandle) {
        
        let ssrHitHandle = graph.createTexture("SSR_Hit", { format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

        if (enabled) {
            graph.addPass("SSR Generate")
                .readTexture(hizHandle)
                .readTexture(normalHandle)
                .readTexture(specularHandle)
                .readTexture(depthHandle)
                .writeTexture(ssrHitHandle)
                .execute((encoder, pass) => {
                    const ssrBindGroup = renderer.device.createBindGroup({
                        label: "ssr params",
                        layout: this.ssrPipeline.getBindGroupLayout(1),
                        entries: [
                            { binding: 0, resource: pass.getTextureView(hizHandle) },
                            { binding: 1, resource: pass.getTextureView(normalHandle) },
                            { binding: 2, resource: pass.getTextureView(specularHandle) },
                            { binding: 3, resource: pass.getTextureView(depthHandle) },
                            { binding: 4, resource: { buffer: this.deps.ssrUniformsBuffer } },
                        ]
                    });

                    const ssrPass = encoder.beginRenderPass({
                        label: "SSR pass",
                        colorAttachments: [{
                            view: pass.getTextureView(ssrHitHandle), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store"
                        }]
                    });
                    ssrPass.setPipeline(this.ssrPipeline);
                    ssrPass.setBindGroup(0, cameraBindGroup);
                    ssrPass.setBindGroup(1, ssrBindGroup);
                    ssrPass.draw(3);
                    ssrPass.end();
                });
        } else {
            graph.addPass("SSR Clear")
                .writeTexture(ssrHitHandle)
                .execute((encoder, pass) => {
                    const ssrPass = encoder.beginRenderPass({
                        label: "SSR pass",
                        colorAttachments: [{
                            view: pass.getTextureView(ssrHitHandle), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store"
                        }]
                    });
                    ssrPass.end();
                });
        }

        graph.addPass("SSR Composite")
            .readTexture(sceneColorHandle)
            .readTexture(ssrHitHandle)
            .readTexture(albedoHandle)
            .readTexture(specularHandle)
            .readTexture(normalHandle)
            .readTexture(depthHandle)
            .writeTexture(canvasHandle)
            .execute((encoder, pass) => {
                const compositeBindGroup = renderer.device.createBindGroup({
                    label: "ssr composite",
                    layout: this.compositePipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: pass.getTextureView(sceneColorHandle) },
                        { binding: 1, resource: pass.getTextureView(ssrHitHandle) },
                        { binding: 2, resource: pass.getTextureView(albedoHandle) },
                        { binding: 3, resource: pass.getTextureView(specularHandle) },
                        { binding: 4, resource: pass.getTextureView(normalHandle) },
                        { binding: 5, resource: pass.getTextureView(depthHandle) }
                    ]
                });

                const compositePass = encoder.beginRenderPass({
                    label: "SSR composite pass",
                    colorAttachments: [{
                        view: pass.getTextureView(canvasHandle), loadOp: "clear", clearValue: [0,0,0,1], storeOp: "store"
                    }]
                });
                compositePass.setPipeline(this.compositePipeline);
                compositePass.setBindGroup(0, cameraBindGroup);
                compositePass.setBindGroup(1, compositeBindGroup);
                compositePass.draw(3);
                compositePass.end();
            });
    }
}
