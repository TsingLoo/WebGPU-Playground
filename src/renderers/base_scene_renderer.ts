import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { RadianceCascades } from '../stage/radiance_cascades';
import { VSM } from '../stage/vsm';

import { SkyboxPass } from './passes/skybox_pass';
import { VolumetricPass } from './passes/volumetric_pass';
import { SSAOPass } from './passes/ssao_pass';
import { DebugPass } from './passes/debug_pass';
import { DDGIDebugPass } from './passes/ddgi_debug_pass';
import { ClusteringPass } from './passes/clustering_pass';
import { HiZPass } from './passes/hiz_pass';

export abstract class BaseSceneRenderer extends renderer.Renderer {
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    gBufferAlbedoTexture: GPUTexture;
    gBufferAlbedoTextureView: GPUTextureView;
    gBufferNormalTexture: GPUTexture;
    gBufferNormalTextureView: GPUTextureView;
    gBufferPositionTexture: GPUTexture;
    gBufferPositionTextureView: GPUTextureView;
    gBufferSpecularTexture: GPUTexture;
    gBufferSpecularTextureView: GPUTextureView;

    tileOffsetsDeviceBuffer: GPUBuffer;
    globalLightIndicesDeviceBuffer: GPUBuffer;
    zeroDeviceBuffer: GPUBuffer;
    clusterSetDeviceBuffer: GPUBuffer;

    dummyTextureView: GPUTextureView;
    dummyBuffer: GPUBuffer;
    dummyStorageBuffer: GPUBuffer;

    zPrepassPipelineCache = new Map<string, GPURenderPipeline>();
    geometryPipelineCache = new Map<string, GPURenderPipeline>();
    geometryBindGroupLayout: GPUBindGroupLayout;
    geometryBindGroup: GPUBindGroup;

    clusteringPass: ClusteringPass;

    // Pass modules
    skyboxPass: SkyboxPass;
    volumetricPass: VolumetricPass;
    ssaoPass: SSAOPass;
    debugPass: DebugPass;
    ddgiDebugPass: DDGIDebugPass;
    hizPass: HiZPass;

    radianceCascades: RadianceCascades;
    vsm: VSM;
    protected stageEnv: import('../stage/environment').Environment;
    protected stage: import('../stage/stage').Stage;

    constructor(stage: Stage) {
        super(stage);

        const gBufSize = [renderer.canvas.width, renderer.canvas.height];
        this.stageEnv = stage.environment;
        this.stage = stage;
        this.radianceCascades = stage.radianceCascades;
        this.vsm = stage.vsm;

        // Dummy texture and buffer for unused GI bindings
        const dummyTex = renderer.device.createTexture({
            size: [1, 1], format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        });
        this.dummyTextureView = dummyTex.createView();
        this.dummyBuffer = renderer.device.createBuffer({
            size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.dummyStorageBuffer = renderer.device.createBuffer({
            size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // Depth buffer
        this.depthTexture = renderer.device.createTexture({
            size: gBufSize,
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        // G-Buffer textures
        this.gBufferAlbedoTexture = renderer.device.createTexture({
            label: "G-Buffer Albedo Texture",
            size: gBufSize, format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferAlbedoTextureView = this.gBufferAlbedoTexture.createView();

        this.gBufferNormalTexture = renderer.device.createTexture({
            label: "G-Buffer Normal Texture",
            size: gBufSize, format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferNormalTextureView = this.gBufferNormalTexture.createView();

        this.gBufferPositionTexture = renderer.device.createTexture({
            label: "G-Buffer Position Texture",
            size: gBufSize, format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferPositionTextureView = this.gBufferPositionTexture.createView();

        this.gBufferSpecularTexture = renderer.device.createTexture({
            label: "G-Buffer Specular Texture",
            size: gBufSize, format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferSpecularTextureView = this.gBufferSpecularTexture.createView();

        // Cluster buffers
        this.tileOffsetsDeviceBuffer = renderer.device.createBuffer({
            size: shaders.constants.numTotalClustersConfig * 2 * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.zeroDeviceBuffer = renderer.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });
        new Uint32Array(this.zeroDeviceBuffer.getMappedRange()).set([0]);
        this.zeroDeviceBuffer.unmap();

        this.clusterSetDeviceBuffer = renderer.device.createBuffer({
            size: 4 * 5,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        const mappedRange = this.clusterSetDeviceBuffer.getMappedRange();
        const uintView = new Uint32Array(mappedRange);
        uintView[0] = renderer.canvas.width;
        uintView[1] = renderer.canvas.height;
        uintView[2] = shaders.constants.numClustersX;
        uintView[3] = shaders.constants.numClustersY;
        uintView[4] = shaders.constants.numClustersZ;
        this.clusterSetDeviceBuffer.unmap();

        const maxIndices = shaders.constants.numTotalClustersConfig * shaders.constants.averageLightsPerCluster;
        this.globalLightIndicesDeviceBuffer = renderer.device.createBuffer({
            size: 4 + maxIndices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: "global light indices buffer"
        });

        // Geometry / Z-Prepass bind groups
        this.geometryBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "geometry bind group layout",
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }]
        });
        this.geometryBindGroup = renderer.device.createBindGroup({
            label: "geometry bind group",
            layout: this.geometryBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.camera.uniformsBuffer } }]
        });

        this.hizPass = new HiZPass();
        this.hizPass.resize(renderer.canvas.width, renderer.canvas.height);

        // Light Clustering
        this.clusteringPass = new ClusteringPass({
            cameraBuffer: this.camera.uniformsBuffer,
            lightSetStorageBuffer: this.lights.lightSetStorageBuffer,
            tileOffsetsBuffer: this.tileOffsetsDeviceBuffer,
            globalLightIndicesBuffer: this.globalLightIndicesDeviceBuffer,
            clusterSetBuffer: this.clusterSetDeviceBuffer,
            zeroBuffer: this.zeroDeviceBuffer,
            hizTextureView: this.hizPass.hizTexture.createView()
        });

        // Initialize pass modules
        this.ssaoPass = new SSAOPass({
            cameraBuffer: this.camera.uniformsBuffer,
            hizTextureView: this.hizPass.hizTexture.createView(),
            gBufferNormalView: this.gBufferNormalTextureView,
            ssaoUniformsBuffer: this.stage.ssao.uniformsBuffer,
        });

        this.skyboxPass = new SkyboxPass({
            cameraBuffer: this.camera.uniformsBuffer,
            envCubemapView: this.stageEnv.envCubemapView,
            envSampler: this.stageEnv.envSampler,
        });

        this.volumetricPass = new VolumetricPass({
            cameraBuffer: this.camera.uniformsBuffer,
            depthTextureView: this.depthTextureView,
            sunLightBuffer: this.stage.sunLightBuffer,
            vsm: this.vsm,
        });

        this.debugPass = new DebugPass({
            cameraBindGroupLayout: this.geometryBindGroupLayout,
            cameraBindGroup: this.geometryBindGroup,
        });

        this.ddgiDebugPass = new DDGIDebugPass();
    }

    // ==== Template Method Architecture ====

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        // 1. Update active sun light
        this.stage.updateSunLight();

        // 2. Z-Prepass
        const zPrepass = encoder.beginRenderPass({
            label: "z prepass",
            colorAttachments: [],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 0.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });
        zPrepass.setBindGroup(shaders.constants.bindGroup_scene, this.geometryBindGroup);

        // Opaque queue
        let currentPipeline: GPURenderPipeline | null = null;
        this.scene.iterate(mr => { zPrepass.setBindGroup(shaders.constants.bindGroup_model, mr.modelBindGroup!); },
                           material => {
                               const pipeline = this.getOrCreateZPrepassPipeline(material.type, true);
                               if (currentPipeline !== pipeline) {
                                   zPrepass.setPipeline(pipeline);
                                   currentPipeline = pipeline;
                               }
                               zPrepass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
                           },
                           primitive => {
                               zPrepass.setVertexBuffer(0, primitive.vertexBuffer);
                               zPrepass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               zPrepass.drawIndexed(primitive.numIndices);
                           }, true);

        // Alpha-cutout queue
        currentPipeline = null;
        this.scene.iterate(mr => { zPrepass.setBindGroup(shaders.constants.bindGroup_model, mr.modelBindGroup!); },
                           material => {
                               const pipeline = this.getOrCreateZPrepassPipeline(material.type, false);
                               if (currentPipeline !== pipeline) {
                                   zPrepass.setPipeline(pipeline);
                                   currentPipeline = pipeline;
                               }
                               zPrepass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
                           },
                           primitive => {
                               zPrepass.setVertexBuffer(0, primitive.vertexBuffer);
                               zPrepass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               zPrepass.drawIndexed(primitive.numIndices);
                           }, false);
        zPrepass.end();

        // 2.5 Hi-Z Generation
        this.hizPass.execute(encoder, this.depthTextureView);

        // 3. VSM Shadow Map Pass
        this.stage.renderShadowMap(encoder, this.depthTextureView);

        // 4. G-Buffer Pass
        const gBufferPass = encoder.beginRenderPass({
            label: "G-buffer pass",
            colorAttachments: [
                { view: this.gBufferAlbedoTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
                { view: this.gBufferNormalTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
                { view: this.gBufferPositionTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
                { view: this.gBufferSpecularTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
            ],
            depthStencilAttachment: { view: this.depthTextureView, depthReadOnly: true }
        });
        gBufferPass.setBindGroup(shaders.constants.bindGroup_scene, this.geometryBindGroup);

        // Opaque queue
        let currentGeometryPipeline: GPURenderPipeline | null = null;
        this.scene.iterate(mr => { gBufferPass.setBindGroup(shaders.constants.bindGroup_model, mr.modelBindGroup!); },
                           material => {
                               const pipeline = this.getOrCreateGeometryPipeline(material.type, true);
                               if (currentGeometryPipeline !== pipeline) {
                                   gBufferPass.setPipeline(pipeline);
                                   currentGeometryPipeline = pipeline;
                               }
                               gBufferPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
                           },
                           primitive => {
                               gBufferPass.setVertexBuffer(0, primitive.vertexBuffer);
                               gBufferPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               gBufferPass.drawIndexed(primitive.numIndices);
                           }, true);

        // Alpha-cutout queue
        currentGeometryPipeline = null;
        this.scene.iterate(mr => { gBufferPass.setBindGroup(shaders.constants.bindGroup_model, mr.modelBindGroup!); },
                           material => {
                               const pipeline = this.getOrCreateGeometryPipeline(material.type, false);
                               if (currentGeometryPipeline !== pipeline) {
                                   gBufferPass.setPipeline(pipeline);
                                   currentGeometryPipeline = pipeline;
                               }
                               gBufferPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
                           },
                           primitive => {
                               gBufferPass.setVertexBuffer(0, primitive.vertexBuffer);
                               gBufferPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               gBufferPass.drawIndexed(primitive.numIndices);
                           }, false);
        gBufferPass.end();

        // 5. GI Updates
        if (this.stage.ddgi.enabled) {
            this.stage.ddgi.update(
                encoder, 
                this.stage.scene.bvhData, 
                this.stage.scene.globalMaterialBuffer,
                this.stage.sunLightBuffer,
                this.vsm.physicalAtlasView, 
                this.vsm.vsmUniformBuffer,
                this.stage.scene.bvhData.uvBuffer,
                this.stage.scene.baseColorTexArrayView,
            );
        }
        if (this.stage.radianceCascades.enabled) {
            this.stage.radianceCascades.update(
                encoder, this.stage.scene.voxelGridView, this.stage.sunLightBuffer,
                this.vsm.physicalAtlasView, this.vsm.vsmUniformBuffer
            );
            this.radianceCascades.updateUniforms();
        }

        // 6. Light Clustering
        this.clusteringPass.execute(encoder);

        // 7. Create shading bind group (sub-class hook)
        this.createShadingBindGroup();

        // 8. SSAO
        this.ssaoPass.execute(encoder, this.stage.ssao.enabled);

        // 9. Sub-class shading pass
        this.executeShadingPass(encoder, canvasTextureView);

        // 10. Skybox
        this.skyboxPass.execute(encoder, canvasTextureView, this.depthTextureView);

        // 11. Volumetric lighting
        if (this.stage.sunVolumetricEnabled) {
            this.volumetricPass.execute(encoder, canvasTextureView);
        }

        // 12. Debug bounds
        if (this.stage.showGIBounds && (this.stage.ddgi.enabled || this.stage.radianceCascades.enabled)) {
            const isDDGI = this.stage.ddgi.enabled;
            const minPos = isDDGI ? this.stage.ddgi.gridMin : this.stage.radianceCascades.gridMin;
            const maxPos = isDDGI ? this.stage.ddgi.gridMax : this.stage.radianceCascades.gridMax;
            const color = isDDGI ? [0.0, 1.0, 0.0, 1.0] : [1.0, 0.5, 0.0, 1.0];
            this.debugPass.execute(encoder, canvasTextureView, this.depthTextureView, this.geometryBindGroup, minPos, maxPos, color);
        }

        // 13. DDGI Probes
        if (this.stage.ddgi.enabled && (this.stage.ddgi as any).showProbes) {
            this.ddgiDebugPass.execute(encoder, canvasTextureView, this.depthTextureView, {
                cameraBindGroupLayout: this.geometryBindGroupLayout,
                cameraBindGroup: this.geometryBindGroup,
                ddgi: this.stage.ddgi
            });
        }

        renderer.device.queue.submit([encoder.finish()]);
    }

    // Sub-classes implement these
    protected abstract createShadingBindGroup(): void;
    protected abstract executeShadingPass(encoder: GPUCommandEncoder, canvasTextureView: GPUTextureView): void;

    // ==== Dynamic Pipeline Caches ====

    protected getOrCreateGeometryPipeline(materialType: string, isOpaque: boolean): GPURenderPipeline {
        const variantKey = `${materialType}_${isOpaque ? 'opaque' : 'cutout'}`;
        if (this.geometryPipelineCache.has(variantKey)) {
            return this.geometryPipelineCache.get(variantKey)!;
        }

        const shaderSrc = shaders.buildGeometryShader(materialType, isOpaque);
        const pipeline = renderer.device.createRenderPipeline({
            label: `geometry pipeline (${variantKey})`,
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'equal' },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.buildVertexShader(materialType) }), buffers: [renderer.vertexBufferLayout] },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaderSrc }),
                entryPoint: 'main',
                targets: [
                    { format: 'rgba16float' }, // albedo
                    { format: 'rgba16float' }, // normal
                    { format: 'rgba16float' }, // position
                    { format: 'rgba8unorm' },  // specular
                ]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' }
        });

        this.geometryPipelineCache.set(variantKey, pipeline);
        return pipeline;
    }

    protected getOrCreateZPrepassPipeline(materialType: string, isOpaque: boolean): GPURenderPipeline {
        const variantKey = `${materialType}_${isOpaque ? 'opaque' : 'cutout'}`;
        if (this.zPrepassPipelineCache.has(variantKey)) {
            return this.zPrepassPipelineCache.get(variantKey)!;
        }

        const fragConfig = isOpaque ? undefined : {
            module: renderer.device.createShaderModule({ code: shaders.buildZPrepassShader(materialType) }),
            entryPoint: "main",
            targets: [] as Iterable<GPUColorTargetState>
        };

        const pipeline = renderer.device.createRenderPipeline({
            label: `Z-Prepass pipeline (${variantKey})`,
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { depthWriteEnabled: true, depthCompare: "greater", format: "depth24plus" },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.buildVertexShader(materialType) }), buffers: [renderer.vertexBufferLayout] },
            fragment: fragConfig,
            primitive: { topology: 'triangle-list', cullMode: 'none' }
        });

        this.zPrepassPipelineCache.set(variantKey, pipeline);
        return pipeline;
    }
}
