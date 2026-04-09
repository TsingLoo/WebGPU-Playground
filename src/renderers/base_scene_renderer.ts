import * as renderer from '../renderer';
import { RenderGraph } from '../engine/RenderGraph';
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
import { SSRPass } from './passes/ssr_pass';

export interface GBufferHandles {
    albedo: import('../engine/RenderGraph').ResourceHandle;
    normal: import('../engine/RenderGraph').ResourceHandle;
    position: import('../engine/RenderGraph').ResourceHandle;
    specular: import('../engine/RenderGraph').ResourceHandle;
    depth: import('../engine/RenderGraph').ResourceHandle;
    sceneColor: import('../engine/RenderGraph').ResourceHandle;
    ssao: import('../engine/RenderGraph').ResourceHandle;
}

export abstract class BaseSceneRenderer extends renderer.Renderer {
    // These views will be dynamically updated each frame by the RenderGraph
    depthTextureView!: GPUTextureView;
    sceneColorTextureView!: GPUTextureView;

    gBufferAlbedoTextureView!: GPUTextureView;
    gBufferNormalTextureView!: GPUTextureView;
    gBufferPositionTextureView!: GPUTextureView;
    gBufferSpecularTextureView!: GPUTextureView;

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
    ssrPass: SSRPass;

    radianceCascades: RadianceCascades;
    vsm: VSM;
    protected stageEnv: import('../stage/environment').Environment;
    protected stage: import('../stage/stage').Stage;

    protected finalBlitPipeline: GPURenderPipeline;
    protected finalBlitSampler: GPUSampler;

    protected sharedRenderGraph = new RenderGraph(); // Persist pool across frames
    
    protected requiresGBuffer(): boolean {
        // Base renderer assumes GBuffer is needed by default (Deferred always needs it).
        // Forward+ can override this to optionally skip it when post-processing is off.
        return true; 
    }

    constructor(stage: Stage) {
        super(stage);

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
            hizTextureView: this.hizPass.hizFullTextureView // Wait, might need dynamic update later if HiZ uses RenderGraph
        });

        // Initialize pass modules
        this.ssaoPass = new SSAOPass({
            cameraBuffer: this.camera.uniformsBuffer,
            ssaoUniformsBuffer: this.stage.ssao.uniformsBuffer,
        });

        this.ssrPass = new SSRPass({
            cameraBuffer: this.camera.uniformsBuffer,
            ssrUniformsBuffer: this.stage.ssr.uniformsBuffer,
        });

        this.skyboxPass = new SkyboxPass({
            cameraBuffer: this.camera.uniformsBuffer,
            envCubemapView: this.stageEnv.envCubemapView,
            envSampler: this.stageEnv.envSampler,
        });

        this.volumetricPass = new VolumetricPass({
            cameraBuffer: this.camera.uniformsBuffer,
            sunLightBuffer: this.stage.sunLightBuffer,
            vsm: this.vsm,
        });

        this.debugPass = new DebugPass({
            cameraBindGroupLayout: this.geometryBindGroupLayout,
            cameraBindGroup: this.geometryBindGroup,
        });

        this.ddgiDebugPass = new DDGIDebugPass();

        this.finalBlitSampler = renderer.device.createSampler({});
        const blitBGL = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });
        this.finalBlitPipeline = renderer.device.createRenderPipeline({
            label: "final blit pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [blitBGL] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.fullscreenBlitFragSrc }), entryPoint: "main", targets: [{ format: renderer.canvasFormat }] }
        });
    }
    // ==== Template Method Architecture ====

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        const graph = this.sharedRenderGraph;

        // --- Render Graph Setup ---
        const canvasHandle = graph.importTexture("Canvas", canvasTextureView);

        const depthHandle = graph.createTexture("SceneDepth", {
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
        });

        const sceneColorHandle = graph.createTexture("SceneColor", {
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
        });

        const albedoHandle = graph.createTexture("GBufferAlbedo", { format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        const normalHandle = graph.createTexture("GBufferNormal", { format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        const positionHandle = graph.createTexture("GBufferPosition", { format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        const specularHandle = graph.createTexture("GBufferSpecular", { format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

        // 1. Stage Data Update Pass
        graph.addGenericPass("Stage Updates")
            .markRoot()
            .execute((_, _pass) => {
                this.stage.updateSunLight();
            });

        // 2. Z-Prepass
        graph.addRenderPass("Z-Prepass")
            .setDepthStencilAttachment(depthHandle, { clearValue: 0.0 })
            .execute((zPrepass, pass) => {
                // Update our class property so legacy systems can read it
                this.depthTextureView = pass.getTextureView(depthHandle);

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
            });

        // 2.5 Hi-Z Generation
        graph.addGenericPass("HiZ")
            .markRoot()
            .readTexture(depthHandle)
            .execute((enc, _pass) => {
                this.hizPass.execute(enc, this.depthTextureView);
            });

        const hizHandle = graph.importTexture("HiZ_Import", this.hizPass.hizFullTextureView);

        // 3. VSM Shadow Map Pass
        if (this.stage.vsmEnabled) {
            graph.addGenericPass("Shadow Map")
                .markRoot()
                .readTexture(depthHandle)
                .execute((enc, _pass) => {
                    this.stage.renderShadowMap(enc, this.depthTextureView);
                });
        }

        // 4. G-Buffer Pass
        graph.addRenderPass("G-Buffer")
            .readTexture(depthHandle)
            .addColorAttachment(albedoHandle, { clearValue: [0,0,0,0] })
            .addColorAttachment(normalHandle, { clearValue: [0,0,0,0] })
            .addColorAttachment(positionHandle, { clearValue: [0,0,0,0] })
            .addColorAttachment(specularHandle, { clearValue: [0,0,0,0] })
            .setDepthStencilAttachment(depthHandle, { depthReadOnly: true })
            .execute((gBufferPass, pass) => {
                // Bridge views (legacy support)
                this.gBufferAlbedoTextureView = pass.getTextureView(albedoHandle);
                this.gBufferNormalTextureView = pass.getTextureView(normalHandle);
                this.gBufferPositionTextureView = pass.getTextureView(positionHandle);
                this.gBufferSpecularTextureView = pass.getTextureView(specularHandle);
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
            });

        // 5. GI Updates
        graph.addGenericPass("GI Updates")
            .markRoot()
            .execute((enc, _pass) => {
                if (this.stage.ddgi.enabled) {
                    this.stage.ddgi.update(
                        enc, 
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
                        enc, this.stage.scene.voxelGridView, this.stage.sunLightBuffer,
                        this.vsm.physicalAtlasView, this.vsm.vsmUniformBuffer
                    );
                    this.radianceCascades.updateUniforms();
                }
            });

        // 6. Light Clustering
        graph.addGenericPass("Light Clustering")
            .markRoot()
            .execute((enc, _pass) => {
                this.clusteringPass.execute(enc);
            });

        const ssaoHandle = this.ssaoPass.addToGraph(graph, this.stage.ssao.enabled, hizHandle, normalHandle);

        this.addToGraphShading(graph, { depth: depthHandle, albedo: albedoHandle, normal: normalHandle, position: positionHandle, specular: specularHandle, sceneColor: sceneColorHandle, ssao: ssaoHandle });

        graph.addRenderPass("Skybox & Debug")
             .addColorAttachment(sceneColorHandle)
             .setDepthStencilAttachment(depthHandle, { depthReadOnly: true })
             .execute((renderPass, _pass) => {
                 this.skyboxPass.execute(renderPass);

                 if (this.stage.showGIBounds && (this.stage.ddgi.enabled || this.stage.radianceCascades.enabled)) {
                     const isDDGI = this.stage.ddgi.enabled;
                     const minPos = isDDGI ? this.stage.ddgi.gridMin : this.stage.radianceCascades.gridMin;
                     const maxPos = isDDGI ? this.stage.ddgi.gridMax : this.stage.radianceCascades.gridMax;
                     const color = isDDGI ? [0.0, 1.0, 0.0, 1.0] : [1.0, 0.5, 0.0, 1.0];
                     this.debugPass.execute(renderPass, this.geometryBindGroup, minPos, maxPos, color);
                 }

                 if (this.stage.ddgi.enabled && (this.stage.ddgi as any).showProbes) {
                     this.ddgiDebugPass.execute(renderPass, {
                         cameraBindGroupLayout: this.geometryBindGroupLayout,
                         cameraBindGroup: this.geometryBindGroup,
                         ddgi: this.stage.ddgi
                     });
                 }
             });

        if (this.stage.ssr.enabled) {
            this.ssrPass.addToGraph(graph, this.stage.ssr.enabled, this.geometryBindGroup, sceneColorHandle, albedoHandle, normalHandle, specularHandle, depthHandle, hizHandle, canvasHandle);
        } else {
            // Graceful fallback for final output when post processing is omitted
            graph.addRenderPass("Final Output Blit")
                .readTexture(sceneColorHandle)
                .addColorAttachment(canvasHandle, { clearValue: [0,0,0,1] })
                .execute((blitPass, pass) => {
                    blitPass.setPipeline(this.finalBlitPipeline);
                    // Create a temporary bindgroup for the simple copy
                    const blitBG = renderer.device.createBindGroup({
                        layout: this.finalBlitPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: pass.getTextureView(sceneColorHandle) },
                            { binding: 1, resource: this.finalBlitSampler }
                        ]
                    });
                    blitPass.setBindGroup(0, blitBG);
                    blitPass.draw(3);
                });
        }

        // 12. Volumetric lighting
        if (this.stage.sunVolumetricEnabled) {
            this.volumetricPass.addToGraph(graph, canvasHandle, depthHandle);
        }

        // Compile and execute the RenderGraph!
        graph.execute(encoder);

        renderer.device.queue.submit([encoder.finish()]);
    }

    // Sub-classes implement these
    protected abstract createShadingBindGroup(): void;
    protected abstract addToGraphShading(graph: RenderGraph, handles: GBufferHandles): void;

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
