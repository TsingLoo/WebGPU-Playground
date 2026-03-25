import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { BaseSceneRenderer } from './base_scene_renderer';

export class ForwardPlusRenderer extends BaseSceneRenderer {
    shadingStaticBindGroupLayout: GPUBindGroupLayout; 
    shadingStaticBindGroup: GPUBindGroup;
    
    giDynamicBindGroupLayout: GPUBindGroupLayout;
    giDynamicBindGroup!: GPUBindGroup;

    shadingPipeline: GPURenderPipeline;
    shadingOpaquePipeline: GPURenderPipeline;

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

        this.shadingStaticBindGroup = renderer.device.createBindGroup({
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
                { binding: 21, resource: this.ssaoBlurredTextureView },
            ]
        });

        this.giDynamicBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "GI dynamic bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });

        this.shadingPipeline = renderer.device.createRenderPipeline({
            label: "fwd+ shading cutout pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingStaticBindGroupLayout, 
                    renderer.modelBindGroupLayout, 
                    renderer.materialBindGroupLayout,
                    this.giDynamicBindGroupLayout
                ]
            }),
            depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }), buffers: [renderer.vertexBufferLayout] },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.forwardPlusFragSrc }), entryPoint: "main", targets: [{ format: renderer.canvasFormat }] }
        });

        this.shadingOpaquePipeline = renderer.device.createRenderPipeline({
            label: "fwd+ shading opaque pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingStaticBindGroupLayout, 
                    renderer.modelBindGroupLayout, 
                    renderer.materialBindGroupLayout,
                    this.giDynamicBindGroupLayout
                ]
            }),
            depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }), buffers: [renderer.vertexBufferLayout] },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.forwardPlusOpaqueFragSrc }), entryPoint: "main", targets: [{ format: renderer.canvasFormat }] }
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
        const shadingRenderPass = encoder.beginRenderPass({
            label: "Shading Pass",
            colorAttachments: [ { view: canvasTextureView, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" } ],
            depthStencilAttachment: { view: this.depthTextureView, depthReadOnly: true }
        });
        // Opaque queue
        shadingRenderPass.setBindGroup(0, this.shadingStaticBindGroup);
        shadingRenderPass.setBindGroup(3, this.giDynamicBindGroup);
        shadingRenderPass.setPipeline(this.shadingOpaquePipeline);
        this.scene.iterate(node => {
            shadingRenderPass.setBindGroup(1, node.modelBindGroup);
        }, material => {
            shadingRenderPass.setBindGroup(2, material.materialBindGroup);
        }, primitive => {
            shadingRenderPass.setVertexBuffer(0, primitive.vertexBuffer);
            shadingRenderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            shadingRenderPass.drawIndexed(primitive.numIndices);
        }, true);

        // Alpha-cutout queue
        shadingRenderPass.setPipeline(this.shadingPipeline);
        this.scene.iterate(node => {
            shadingRenderPass.setBindGroup(1, node.modelBindGroup);
        }, material => {
            shadingRenderPass.setBindGroup(2, material.materialBindGroup);
        }, primitive => {
            shadingRenderPass.setVertexBuffer(0, primitive.vertexBuffer);
            shadingRenderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            shadingRenderPass.drawIndexed(primitive.numIndices);
        }, false);
        shadingRenderPass.end();
    }
}
