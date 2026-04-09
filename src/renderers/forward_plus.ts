import * as renderer from '../renderer';
import { BindGroupCache } from '../engine/RenderTexManager';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { BaseSceneRenderer } from './base_scene_renderer';

export class ForwardPlusRenderer extends BaseSceneRenderer {
    shadingStaticBindGroupLayout: GPUBindGroupLayout;
    
    giDynamicBindGroupLayout: GPUBindGroupLayout;
    giDynamicBindGroup!: GPUBindGroup;

    shadingPipelineCache = new Map<string, GPURenderPipeline>();

    constructor(stage: Stage) {
        super(stage);

        this.shadingStaticBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "shading static bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                // shifted from 13
                { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 10, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 11, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
                { binding: 12, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 13, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 14, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 15, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 16, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 17, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 18, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 19, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 20, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 21, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }
            ]
        });

        // BindGroup is now dynamically cached in executeShadingPass

        this.giDynamicBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "GI dynamic bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }
            ]
        });

        // Shading pipelines are now dynamically generated via getOrCreateShadingPipeline
    }

    private cachedGiBindGroup: GPUBindGroup | null = null;
    private lastGiActive = false;

    protected override createShadingBindGroup() {
        const giActive = this.stage.ddgi.enabled || this.stage.radianceCascades.enabled;
        
        // Only recreate when GI is active (ping-pong changes views each frame)
        // or when GI state changed (need to switch between real and dummy views)
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

    protected override executeShadingPass(encoder: GPUCommandEncoder, canvasTextureView: GPUTextureView) {
        const shadingStaticBindGroup = BindGroupCache.get({
            label: "shading static bind group",
            layout: this.shadingStaticBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                { binding: 5, resource: this.stageEnv.irradianceMapView },
                { binding: 6, resource: this.stageEnv.prefilteredMapView },
                { binding: 7, resource: this.stageEnv.brdfLutView },
                { binding: 8, resource: this.stageEnv.envSampler },
                { binding: 9, resource: { buffer: this.stage.sunLightBuffer } },
                { binding: 10, resource: this.vsm.physicalAtlasView },
                { binding: 11, resource: this.vsm.shadowComparisonSampler },
                { binding: 12, resource: { buffer: this.vsm.pageTableBuffer } },
                { binding: 13, resource: { buffer: this.vsm.vsmUniformBuffer } },
                { binding: 14, resource: this.dummyTextureView },
                { binding: 15, resource: { buffer: this.dummyBuffer } },
                { binding: 16, resource: this.gBufferPositionTextureView },
                { binding: 17, resource: this.gBufferNormalTextureView },
                { binding: 18, resource: this.gBufferAlbedoTextureView },
                { binding: 19, resource: this.dummyTextureView },
                { binding: 20, resource: { buffer: this.dummyBuffer } },
                { binding: 21, resource: this.ssaoPass.blurredTextureView },
            ]
        });

        const shadingRenderPass = encoder.beginRenderPass({
            label: "Shading Pass",
            colorAttachments: [ { view: canvasTextureView, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" } ],
            depthStencilAttachment: { view: this.depthTextureView, depthReadOnly: true }
        });
        // Opaque queue
        shadingRenderPass.setBindGroup(0, shadingStaticBindGroup);
        shadingRenderPass.setBindGroup(3, this.giDynamicBindGroup);
        
        let currentPipeline: GPURenderPipeline | null = null;
        
        this.scene.iterate(mr => {
            shadingRenderPass.setBindGroup(1, mr.modelBindGroup!);
        }, material => {
            const pipeline = this.getOrCreateShadingPipeline(material.type, true);
            if (currentPipeline !== pipeline) {
                shadingRenderPass.setPipeline(pipeline);
                currentPipeline = pipeline;
            }
            shadingRenderPass.setBindGroup(2, material.materialBindGroup);
        }, primitive => {
            shadingRenderPass.setVertexBuffer(0, primitive.vertexBuffer);
            shadingRenderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            shadingRenderPass.drawIndexed(primitive.numIndices);
        }, true);

        // Alpha-cutout queue
        currentPipeline = null;
        this.scene.iterate(mr => {
            shadingRenderPass.setBindGroup(1, mr.modelBindGroup!);
        }, material => {
            const pipeline = this.getOrCreateShadingPipeline(material.type, false);
            if (currentPipeline !== pipeline) {
                shadingRenderPass.setPipeline(pipeline);
                currentPipeline = pipeline;
            }
            shadingRenderPass.setBindGroup(2, material.materialBindGroup);
        }, primitive => {
            shadingRenderPass.setVertexBuffer(0, primitive.vertexBuffer);
            shadingRenderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            shadingRenderPass.drawIndexed(primitive.numIndices);
        }, false);
        shadingRenderPass.end();
    }

    private getOrCreateShadingPipeline(materialType: string, isOpaque: boolean): GPURenderPipeline {
        const variantKey = `${materialType}_${isOpaque ? 'opaque' : 'cutout'}`;
        if (this.shadingPipelineCache.has(variantKey)) {
            return this.shadingPipelineCache.get(variantKey)!;
        }

        const shaderSrc = shaders.buildForwardPlusShader(materialType, isOpaque);
        const pipeline = renderer.device.createRenderPipeline({
            label: `fwd+ shading pipeline (${variantKey})`,
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingStaticBindGroupLayout, 
                    renderer.modelBindGroupLayout, 
                    renderer.materialBindGroupLayout,
                    this.giDynamicBindGroupLayout
                ]
            }),
            depthStencil: { depthWriteEnabled: false, depthCompare: "greater-equal", format: "depth24plus" },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.buildVertexShader(materialType) }), buffers: [renderer.vertexBufferLayout] },
            fragment: { module: renderer.device.createShaderModule({ code: shaderSrc }), entryPoint: "main", targets: [{ format: "rgba16float" }] }
        });

        this.shadingPipelineCache.set(variantKey, pipeline);
        return pipeline;
    }
}
