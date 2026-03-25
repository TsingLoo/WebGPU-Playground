import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { BaseSceneRenderer } from './base_scene_renderer';

export class ClusteredDeferredRenderer extends BaseSceneRenderer {
    shadingOutputDeviceTexture: GPUTexture;
    shadingOutputDeviceTextureView: GPUTextureView;

    shadingStaticBindGroupLayout: GPUBindGroupLayout; 
    shadingStaticBindGroup: GPUBindGroup;

    giDynamicBindGroupLayout: GPUBindGroupLayout;
    giDynamicBindGroup!: GPUBindGroup;

    shadingComputePipeline: GPUComputePipeline;

    blitSampler: GPUSampler;
    blitBindGroupLayout: GPUBindGroupLayout;
    blitBindGroup: GPUBindGroup;
    blitPipeline: GPURenderPipeline;

    constructor(stage: Stage) {
        super(stage);

        let geometryDeviceTextureSize = [renderer.canvas.width, renderer.canvas.height];

        this.shadingOutputDeviceTexture = renderer.device.createTexture({
            label: "shading output Texture",
            size: geometryDeviceTextureSize,
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
        this.shadingOutputDeviceTextureView = this.shadingOutputDeviceTexture.createView();

        this.shadingStaticBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "shading static bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm", viewDimension: "2d" } },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 14, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                // shifted from 19
                { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 16, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
                { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 18, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 19, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 20, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 21, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 22, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 23, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
            ]
        });

        this.shadingStaticBindGroup = renderer.device.createBindGroup({
            label: "shading static bind group",
            layout: this.shadingStaticBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                { binding: 5, resource: this.gBufferAlbedoTextureView },
                { binding: 6, resource: this.gBufferNormalTextureView },
                { binding: 7, resource: this.gBufferPositionTextureView },
                { binding: 8, resource: this.gBufferSpecularTextureView },
                { binding: 9, resource: this.depthTextureView},
                { binding: 10, resource: this.shadingOutputDeviceTextureView },
                { binding: 11, resource: this.stageEnv.irradianceMapView },
                { binding: 12, resource: this.stageEnv.prefilteredMapView },
                { binding: 13, resource: this.stageEnv.brdfLutView },
                { binding: 14, resource: this.stageEnv.envSampler },
                { binding: 15, resource: { buffer: this.stage.sunLightBuffer } },
                { binding: 16, resource: this.stage.vsm.physicalAtlasView },
                { binding: 17, resource: { buffer: this.stage.vsm.pageTableBuffer } },
                { binding: 18, resource: { buffer: this.stage.vsm.vsmUniformBuffer } },
                { binding: 19, resource: this.stage.nrc.getInferenceView() },
                { binding: 20, resource: { buffer: this.stage.nrc.nrcUniformBuffer } },
                { binding: 21, resource: this.surfelIrradianceDeviceTextureView },
                { binding: 22, resource: { buffer: this.surfelParamsBuffer } },
                { binding: 23, resource: this.ssaoBlurredTextureView },
            ]
        });
        
        this.giDynamicBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "GI dynamic bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: {} }
            ]
        });

        this.shadingComputePipeline = renderer.device.createComputePipeline({
            label: "shading compute pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.shadingStaticBindGroupLayout, this.giDynamicBindGroupLayout]
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    code: shaders.clusteredDeferredComputeSrc
                }),
                entryPoint: "main"
            }
        });

        this.blitSampler = renderer.device.createSampler({
            magFilter: 'linear', minFilter: 'linear'
        });
        
        this.blitBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "blit bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });

        this.blitBindGroup = renderer.device.createBindGroup({
            label: "blit bind group",
            layout: this.blitBindGroupLayout,
            entries: [
                { binding: 0, resource: this.shadingOutputDeviceTextureView },
                { binding: 1, resource: this.blitSampler }
            ]
        });

        this.blitPipeline = renderer.device.createRenderPipeline({
            label: "blit pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.blitBindGroupLayout]
            }),
            vertex: {
                module: renderer.device.createShaderModule({
                    code: shaders.clusteredDeferredFullscreenVertSrc
                }),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    code: shaders.clusteredDeferredFullscreenFragSrc
                }),
                entryPoint: "main",
                targets: [{ format: renderer.canvasFormat }]
            }
        });
    }

    protected override createShadingBindGroup() {
        this.giDynamicBindGroup = renderer.device.createBindGroup({
            label: "GI dynamic bind group",
            layout: this.giDynamicBindGroupLayout,
            entries: [
                { binding: 0, resource: this.ddgi.getCurrentIrradianceView() },
                { binding: 1, resource: this.ddgi.getCurrentVisibilityView() },
                { binding: 2, resource: { buffer: this.ddgi.ddgiUniformBuffer } },
                { binding: 3, resource: this.ddgi.ddgiSampler }
            ]
        });
    }

    protected override executeShadingPass(encoder: GPUCommandEncoder, canvasTextureView: GPUTextureView) {
        // Deferred shading compute pass
        const shadingComputePass = encoder.beginComputePass();
        shadingComputePass.setPipeline(this.shadingComputePipeline);
        shadingComputePass.setBindGroup(0, this.shadingStaticBindGroup);
        shadingComputePass.setBindGroup(1, this.giDynamicBindGroup);
        
        const workgroupsX = Math.max(1, Math.ceil(renderer.canvas.width / 8));
        const workgroupsY = Math.max(1, Math.ceil(renderer.canvas.height / 8));
        shadingComputePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        shadingComputePass.end();

        // Blit pass (no depth needed — fullscreen quad)
        const blitPass = encoder.beginRenderPass({
            label: "Blit Pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    clearValue: [0, 0, 0, 1],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ]
        });
        blitPass.setPipeline(this.blitPipeline);
        blitPass.setBindGroup(0, this.blitBindGroup);
        blitPass.draw(3);
        blitPass.end();
    }
}
