import * as renderer from '../renderer';
import { RenderGraph } from '../engine/RenderGraph';
import { GBufferHandles, BaseSceneRenderer } from './base_scene_renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class ClusteredDeferredRenderer extends BaseSceneRenderer {
    shadingStaticBindGroupLayout: GPUBindGroupLayout;

    giDynamicBindGroupLayout: GPUBindGroupLayout;
    giDynamicBindGroup!: GPUBindGroup;

    shadingComputePipeline: GPUComputePipeline;

    blitSampler: GPUSampler;
    blitBindGroupLayout: GPUBindGroupLayout;
    blitPipeline: GPURenderPipeline;

    constructor(stage: Stage) {
        super(stage);

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
                { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } },
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
                { binding: 24, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
            ]
        });

        this.giDynamicBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "GI dynamic bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
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

        this.blitPipeline = renderer.device.createRenderPipeline({
            label: "blit pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.blitBindGroupLayout]
            }),
            vertex: {
                module: renderer.device.createShaderModule({
                    code: shaders.fullscreenBlitVertSrc
                }),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    code: shaders.fullscreenBlitFragSrc
                }),
                entryPoint: "main",
                targets: [{ format: "rgba16float" }]
            }
        });
    }

    private cachedGiBindGroup: GPUBindGroup | null = null;
    private lastGiActive = false;

    protected override createShadingBindGroup() {
        const giActive = this.stage.ddgi.enabled || this.stage.radianceCascades.enabled;
        
        if (this.cachedGiBindGroup && !giActive && !this.lastGiActive) {
            this.giDynamicBindGroup = this.cachedGiBindGroup;
            return;
        }
        this.lastGiActive = giActive;

        this.giDynamicBindGroup = renderer.device.createBindGroup({
            label: "GI dynamic bind group",
            layout: this.giDynamicBindGroupLayout,
            entries: [
                { binding: 0, resource: this.stage.ddgi.getCurrentIrradianceView() },
                { binding: 1, resource: this.stage.ddgi.getCurrentVisibilityView() },
                { binding: 2, resource: { buffer: this.stage.ddgi.ddgiUniformBuffer } },
                { binding: 3, resource: this.stage.ddgi.ddgiSampler },
                { binding: 4, resource: this.stage.radianceCascades.getCurrentIrradianceView() },
                { binding: 5, resource: { buffer: this.stage.radianceCascades.rcUniformBuffer } },
                { binding: 6, resource: this.stage.radianceCascades.rcSampler },
                { binding: 7, resource: { buffer: this.stage.ddgi.probeDataBuffer || this.dummyStorageBuffer } }
            ]
        });
        
        if (!giActive) {
            this.cachedGiBindGroup = this.giDynamicBindGroup;
        }
    }

    protected override addToGraphShading(graph: RenderGraph, handles: GBufferHandles) {
        const shadingOutputHandle = graph.createTexture("ShadingOutput", {
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        graph.addComputePass("Deferred Compute Execution")
            .readTexture(handles.albedo)
            .readTexture(handles.normal)
            .readTexture(handles.position)
            .readTexture(handles.specular)
            .readTexture(handles.depth)
            .readTexture(handles.ssao)
            .readTexture(handles.emissive)
            .writeTexture(shadingOutputHandle)
            .execute((shadingComputePass, pass) => {
                this.createShadingBindGroup();
                
                const shadingBG = renderer.device.createBindGroup({
                    label: "shading static bind group",
                    layout: this.shadingStaticBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                        { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                        { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                        { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                        { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                        { binding: 5, resource: pass.getTextureView(handles.albedo) },
                        { binding: 6, resource: pass.getTextureView(handles.normal) },
                        { binding: 7, resource: pass.getTextureView(handles.position) },
                        { binding: 8, resource: pass.getTextureView(handles.specular) },
                        { binding: 9, resource: pass.getTextureView(handles.depth) },
                        { binding: 10, resource: pass.getTextureView(shadingOutputHandle) },
                        { binding: 11, resource: this.stageEnv.irradianceMapView },
                        { binding: 12, resource: this.stageEnv.prefilteredMapView },
                        { binding: 13, resource: this.stageEnv.brdfLutView },
                        { binding: 14, resource: this.stageEnv.envSampler },
                        { binding: 15, resource: { buffer: this.stage.sunLightBuffer } },
                        { binding: 16, resource: this.stage.vsm.physicalAtlasView },
                        { binding: 17, resource: { buffer: this.stage.vsm.pageTableBuffer } },
                        { binding: 18, resource: { buffer: this.stage.vsm.vsmUniformBuffer } },
                        { binding: 19, resource: this.dummyTextureView },
                        { binding: 20, resource: { buffer: this.dummyBuffer } },
                        { binding: 21, resource: this.dummyTextureView },
                        { binding: 22, resource: { buffer: this.dummyBuffer } },
                        { binding: 23, resource: pass.getTextureView(handles.ssao) },
                        { binding: 24, resource: pass.getTextureView(handles.emissive) },
                    ]
                });

                shadingComputePass.setPipeline(this.shadingComputePipeline);
                shadingComputePass.setBindGroup(0, shadingBG);
                shadingComputePass.setBindGroup(1, this.giDynamicBindGroup);
                
                const workgroupsX = Math.max(1, Math.ceil(renderer.canvas.width / 8));
                const workgroupsY = Math.max(1, Math.ceil(renderer.canvas.height / 8));
                shadingComputePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
            });

        graph.addRenderPass("Deferred Blit")
            .readTexture(shadingOutputHandle)
            .addColorAttachment(handles.sceneColor, { clearValue: [0, 0, 0, 1] })
            .execute((blitPass, pass) => {
                const blitBG = renderer.device.createBindGroup({
                    label: "blit bind group",
                    layout: this.blitBindGroupLayout,
                    entries: [
                        { binding: 0, resource: pass.getTextureView(shadingOutputHandle) },
                        { binding: 1, resource: this.blitSampler }
                    ]
                });

                blitPass.setPipeline(this.blitPipeline);
                blitPass.setBindGroup(0, blitBG);
                blitPass.draw(3);
            });
    }
}
