import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export interface SSRPassDeps {
    cameraBuffer: GPUBuffer;
    hizTextureView: GPUTextureView;
    normalTextureView: GPUTextureView;
    specularTextureView: GPUTextureView;
    depthTextureView: GPUTextureView;
    ssrUniformsBuffer: GPUBuffer;
}

export class SSRPass {
    private ssrHitTexture: GPUTexture;
    private ssrHitTextureView: GPUTextureView;

    private ssrPipeline: GPURenderPipeline;
    private ssrBindGroup: GPUBindGroup;

    private compositePipeline: GPURenderPipeline;
    private deps: SSRPassDeps;

    private dummySsrTexture: GPUTexture;
    private dummySsrTextureView: GPUTextureView;

    constructor(deps: SSRPassDeps) {
        this.deps = deps;
        const size = [renderer.canvas.width, renderer.canvas.height];

        this.ssrHitTexture = renderer.device.createTexture({
            label: "ssr hit texture",
            size, format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.ssrHitTextureView = this.ssrHitTexture.createView();

        this.dummySsrTexture = renderer.device.createTexture({
            label: "ssr dummy texture",
            size: [1, 1], format: "rgba16float",
            usage: GPUTextureUsage.TEXTURE_BINDING
        });
        this.dummySsrTextureView = this.dummySsrTexture.createView();

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
        this.ssrBindGroup = renderer.device.createBindGroup({
            layout: ssrBindGroupLayout,
            entries: [
                { binding: 0, resource: deps.hizTextureView },
                { binding: 1, resource: deps.normalTextureView },
                { binding: 2, resource: deps.specularTextureView },
                { binding: 3, resource: deps.depthTextureView },
                { binding: 4, resource: { buffer: deps.ssrUniformsBuffer } },
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

    execute(encoder: GPUCommandEncoder, enabled: boolean, 
            cameraBindGroup: GPUBindGroup, 
            sceneColorView: GPUTextureView, 
            albedoView: GPUTextureView,
            canvasView: GPUTextureView) {
        
        if (enabled) {
            // Run SSR Hit Tracing
            const ssrPass = encoder.beginRenderPass({
                label: "SSR pass",
                colorAttachments: [{
                    view: this.ssrHitTextureView, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store"
                }]
            });
            ssrPass.setPipeline(this.ssrPipeline);
            ssrPass.setBindGroup(0, cameraBindGroup);
            ssrPass.setBindGroup(1, this.ssrBindGroup);
            ssrPass.draw(3);
            ssrPass.end();
        }

        // Run Composite Pass
        const compositeBGL = this.compositePipeline.getBindGroupLayout(1);
        const compositeBindGroup = renderer.device.createBindGroup({
            layout: compositeBGL,
            entries: [
                { binding: 0, resource: sceneColorView },
                { binding: 1, resource: enabled ? this.ssrHitTextureView : this.dummySsrTextureView },
                { binding: 2, resource: albedoView },
                { binding: 3, resource: this.deps.specularTextureView },
                { binding: 4, resource: this.deps.normalTextureView },
                { binding: 5, resource: this.deps.depthTextureView }
            ]
        });

        const compositePass = encoder.beginRenderPass({
            label: "SSR composite pass",
            colorAttachments: [{
                view: canvasView, loadOp: "clear", clearValue: [0,0,0,1], storeOp: "store"
            }]
        });
        compositePass.setPipeline(this.compositePipeline);
        compositePass.setBindGroup(0, cameraBindGroup);
        compositePass.setBindGroup(1, compositeBindGroup);
        compositePass.draw(3);
        compositePass.end();
    }
}
