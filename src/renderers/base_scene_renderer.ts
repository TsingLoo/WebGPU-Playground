import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { DDGI } from '../stage/ddgi';
import { VSM } from '../stage/vsm';

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

    zPrepassPipeline: GPURenderPipeline;
    zPrepassOpaquePipeline: GPURenderPipeline;
    geometryBindGroupLayout: GPUBindGroupLayout;
    geometryBindGroup: GPUBindGroup;
    geometryPipeline: GPURenderPipeline;
    geometryOpaquePipeline: GPURenderPipeline;

    cullingBindGroupLayout: GPUBindGroupLayout;
    cullingBindGroup: GPUBindGroup;
    cullingPipeline: GPUComputePipeline;

    skyboxPipeline!: GPURenderPipeline;
    skyboxBindGroupLayout!: GPUBindGroupLayout;
    skyboxBindGroup!: GPUBindGroup;

    // Volumetric Lighting
    volumetricTexture: GPUTexture;
    volumetricTextureView: GPUTextureView;
    volumetricPipeline!: GPURenderPipeline;
    volumetricBindGroupLayout!: GPUBindGroupLayout;
    volumetricBindGroup!: GPUBindGroup;
    
    volumetricCompositePipeline!: GPURenderPipeline;
    volumetricCompositeBindGroupLayout!: GPUBindGroupLayout;
    volumetricCompositeBindGroup!: GPUBindGroup;

    ssaoTexture: GPUTexture;
    ssaoTextureView: GPUTextureView;
    ssaoBlurredTexture: GPUTexture;
    ssaoBlurredTextureView: GPUTextureView;

    ssaoBindGroupLayout: GPUBindGroupLayout;
    ssaoBindGroup: GPUBindGroup;
    ssaoPipeline: GPURenderPipeline;

    ssaoBlurBindGroupLayout: GPUBindGroupLayout;
    ssaoBlurBindGroup: GPUBindGroup;
    ssaoBlurPipeline: GPURenderPipeline;

    ddgi: DDGI;
    // Replaced NRC and Surfel with Dummy
    vsm: VSM;
    protected stageEnv: import('../stage/environment').Environment;
    protected stage: import('../stage/stage').Stage;

    constructor(stage: Stage) {
        super(stage);
        
        const gBufSize = [renderer.canvas.width, renderer.canvas.height];
        this.stageEnv = stage.environment;
        this.stage = stage;
        this.ddgi = stage.ddgi;
        // Dummy texture and buffer for unused NRC and Surfel bindings
        const dummyTex = renderer.device.createTexture({
            size: [1, 1], format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        });
        this.dummyTextureView = dummyTex.createView();
        this.dummyBuffer = renderer.device.createBuffer({
            size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.vsm = stage.vsm;

        this.depthTexture = renderer.device.createTexture({
            size: gBufSize,
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        const volWidth = Math.max(1, Math.floor(renderer.canvas.width / 2));
        const volHeight = Math.max(1, Math.floor(renderer.canvas.height / 2));
        this.volumetricTexture = renderer.device.createTexture({
            label: "volumetric downsampled texture",
            size: [volWidth, volHeight],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.volumetricTextureView = this.volumetricTexture.createView();

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

        this.ssaoTexture = renderer.device.createTexture({
            label: "ssao texture",
            size: gBufSize, format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.ssaoTextureView = this.ssaoTexture.createView();

        this.ssaoBlurredTexture = renderer.device.createTexture({
            label: "ssao blurred texture",
            size: gBufSize, format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.ssaoBlurredTextureView = this.ssaoBlurredTexture.createView();



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

        // -------------------------------------------------------------
        // Geometry / Z-Prepass
        // -------------------------------------------------------------
        this.geometryBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "geometry bind group layout",
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }]
        });
        this.geometryBindGroup = renderer.device.createBindGroup({
            label: "geometry bind group",
            layout: this.geometryBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.camera.uniformsBuffer } }]
        });

        this.geometryPipeline = renderer.device.createRenderPipeline({
            label: "geometry pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'equal' },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }), buffers: [renderer.vertexBufferLayout] },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.geometryFragSrc }),
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

        this.geometryOpaquePipeline = renderer.device.createRenderPipeline({
            label: "geometry opaque pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'equal' },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }), buffers: [renderer.vertexBufferLayout] },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.geometryOpaqueFragSrc }),
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

        this.zPrepassPipeline = renderer.device.createRenderPipeline({
            label: "Z-Prepass cutout pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }), buffers: [ renderer.vertexBufferLayout ] },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.zPrepassFragSrc }), entryPoint: "main", targets: [] },
            primitive: { topology: 'triangle-list', cullMode: 'none' }
        });

        this.zPrepassOpaquePipeline = renderer.device.createRenderPipeline({
            label: "Z-Prepass opaque pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
            vertex: { module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }), buffers: [ renderer.vertexBufferLayout ] },
            primitive: { topology: 'triangle-list', cullMode: 'none' } // No fragment block enables Early-Z hardware double-rate depth writes
        });

        // -------------------------------------------------------------
        // Light Clustering
        // -------------------------------------------------------------
        this.cullingBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "culling bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });
        this.cullingBindGroup = renderer.device.createBindGroup({
            label: "culling bind group",
            layout: this.cullingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer } },
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer } },
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer } },
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer } },
            ]
        });
        this.cullingPipeline = renderer.device.createComputePipeline({
            label: "culling compute pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.cullingBindGroupLayout] }),
            compute: { module: renderer.device.createShaderModule({ code: shaders.clusteringComputeSrc }), entryPoint: "main" }
        });

        // -------------------------------------------------------------
        // SSAO
        // -------------------------------------------------------------
        this.ssaoBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssao bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.ssaoBindGroup = renderer.device.createBindGroup({
            layout: this.ssaoBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: this.gBufferPositionTextureView },
                { binding: 2, resource: this.gBufferNormalTextureView },
                { binding: 3, resource: { buffer: this.stage.ssao.uniformsBuffer } },
            ]
        });
        this.ssaoPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.ssaoBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFullscreenVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssaoFragSrc }), entryPoint: "main", targets: [{ format: "r16float" }] }
        });

        this.ssaoBlurBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "ssao blur bgl",
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }]
        });
        this.ssaoBlurBindGroup = renderer.device.createBindGroup({
            layout: this.ssaoBlurBindGroupLayout,
            entries: [{ binding: 0, resource: this.ssaoTextureView }]
        });
        this.ssaoBlurPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.ssaoBlurBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFullscreenVertSrc }), entryPoint: "main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.ssaoBlurFragSrc }), entryPoint: "main", targets: [{ format: "r16float" }] }
        });

        // -------------------------------------------------------------
        // Skybox and Volumetrics
        // -------------------------------------------------------------
        this.createSkyboxPipeline();
        this.createVolumetricPipelines();
    }

    private createSkyboxPipeline() {
        this.skyboxBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "skybox bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });
        this.skyboxBindGroup = renderer.device.createBindGroup({
            label: "skybox bind group",
            layout: this.skyboxBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: this.stageEnv.envCubemapView },
                { binding: 2, resource: this.stageEnv.envSampler }
            ]
        });
        this.skyboxPipeline = renderer.device.createRenderPipeline({
            label: "skybox pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.skyboxBindGroupLayout] }),
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

    private createVolumetricPipelines() {
        this.volumetricBindGroupLayout = renderer.device.createBindGroupLayout({
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
        this.volumetricPipeline = renderer.device.createRenderPipeline({
            label: "volumetric lighting pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.volumetricBindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.volumetricLightingVertSrc }), entryPoint: "main" },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricLightingFragSrc }),
                entryPoint: "main",
                targets: [{ format: "rgba16float" }]
            }
        });

        this.volumetricCompositeBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "volumetric composite bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.volumetricCompositePipeline = renderer.device.createRenderPipeline({
            label: "volumetric composite pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.volumetricCompositeBindGroupLayout] }),
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

        this.volumetricBindGroup = renderer.device.createBindGroup({
            label: "volumetric lighting bind group",
            layout: this.volumetricBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: this.depthTextureView },
                { binding: 2, resource: { buffer: this.stage.sunLightBuffer } },
                { binding: 3, resource: this.vsm.physicalAtlasView },
                { binding: 4, resource: { buffer: this.vsm.pageTableBuffer } },
                { binding: 5, resource: { buffer: this.vsm.vsmUniformBuffer } },
            ]
        });

        this.volumetricCompositeBindGroup = renderer.device.createBindGroup({
            label: "volumetric composite bind group",
            layout: this.volumetricCompositeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.volumetricTextureView },
                { binding: 1, resource: this.depthTextureView },
                { binding: 2, resource: { buffer: this.camera.uniformsBuffer } },
            ]
        });
    }

    // ==== Template Method Architecture ====

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        // 1. Update active sun light
        this.stage.updateSunLight();

        // 2. Z-Prepass (Required for VSM shadow page marking)
        const zPrepass = encoder.beginRenderPass({
            label: "z prepass",
            colorAttachments: [],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });
        zPrepass.setBindGroup(shaders.constants.bindGroup_scene, this.geometryBindGroup);
        
        // Opaque queue (Double-rate Z limits hardware writes)
        zPrepass.setPipeline(this.zPrepassOpaquePipeline);
        this.scene.iterate(node => { zPrepass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup); }, 
                           material => { zPrepass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup); }, 
                           primitive => {
                               zPrepass.setVertexBuffer(0, primitive.vertexBuffer);
                               zPrepass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               zPrepass.drawIndexed(primitive.numIndices);
                           }, true);

        // Alpha-cutout queue
        zPrepass.setPipeline(this.zPrepassPipeline);
        this.scene.iterate(node => { zPrepass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup); }, 
                           material => { zPrepass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup); }, 
                           primitive => {
                               zPrepass.setVertexBuffer(0, primitive.vertexBuffer);
                               zPrepass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               zPrepass.drawIndexed(primitive.numIndices);
                           }, false);
        zPrepass.end();

        // 3. VSM Shadow Map Pass
        this.stage.renderShadowMap(encoder, this.depthTextureView);

        // 4. Geometry Pass (G-Buffer)
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
        gBufferPass.setPipeline(this.geometryOpaquePipeline);
        this.scene.iterate(node => { gBufferPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup); }, 
                           material => { gBufferPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup); }, 
                           primitive => {
                               gBufferPass.setVertexBuffer(0, primitive.vertexBuffer);
                               gBufferPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               gBufferPass.drawIndexed(primitive.numIndices);
                           }, true);

        // Alpha-cutout queue
        gBufferPass.setPipeline(this.geometryPipeline);
        this.scene.iterate(node => { gBufferPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup); }, 
                           material => { gBufferPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup); }, 
                           primitive => {
                               gBufferPass.setVertexBuffer(0, primitive.vertexBuffer);
                               gBufferPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                               gBufferPass.drawIndexed(primitive.numIndices);
                           }, false);
        gBufferPass.end();

        if (this.stage.ddgi.enabled) {
            this.stage.ddgi.update(
                encoder, this.stage.scene.voxelGridView, this.stage.sunLightBuffer,
                this.vsm.physicalAtlasView, this.vsm.vsmUniformBuffer
            );
            this.ddgi.updateUniforms();
        }

        // 6. Light Clustering Reset & Compute
        encoder.copyBufferToBuffer(this.zeroDeviceBuffer, 0, this.globalLightIndicesDeviceBuffer, 0, 4);
        const cullingComputePass = encoder.beginComputePass();
        cullingComputePass.setPipeline(this.cullingPipeline);
        cullingComputePass.setBindGroup(shaders.constants.bindGroup_scene, this.cullingBindGroup);
        cullingComputePass.dispatchWorkgroups(shaders.constants.numClustersX, shaders.constants.numClustersY, shaders.constants.numClustersZ);
        cullingComputePass.end();

        // 7. Dynamic sub-class hook: create specific shading bind group
        this.createShadingBindGroup();

        // 8. SSAO Passes
        const ssaoPass = encoder.beginRenderPass({
            label: "SSAO pass",
            colorAttachments: [{
                view: this.ssaoTextureView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
            }]
        });
        ssaoPass.setPipeline(this.ssaoPipeline);
        ssaoPass.setBindGroup(0, this.ssaoBindGroup);
        ssaoPass.draw(3);
        ssaoPass.end();

        const ssaoBlurPass = encoder.beginRenderPass({
            label: "SSAO blur pass",
            colorAttachments: [{
                view: this.ssaoBlurredTextureView, clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store"
            }]
        });
        ssaoBlurPass.setPipeline(this.ssaoBlurPipeline);
        ssaoBlurPass.setBindGroup(0, this.ssaoBlurBindGroup);
        ssaoBlurPass.draw(3);
        ssaoBlurPass.end();

        // 9. Sub-class shading pass implementation
        this.executeShadingPass(encoder, canvasTextureView);

        // 10. Abstract Skybox implementation
        const skyboxPass = encoder.beginRenderPass({
            label: "Skybox Pass",
            colorAttachments: [ { view: canvasTextureView, loadOp: "load", storeOp: "store" } ],
            depthStencilAttachment: { view: this.depthTextureView, depthLoadOp: "load", depthStoreOp: "store" }
        });
        skyboxPass.setPipeline(this.skyboxPipeline);
        skyboxPass.setBindGroup(0, this.skyboxBindGroup);
        skyboxPass.draw(3);
        skyboxPass.end();

        // 11. Volumetrics
        if (this.stage.sunVolumetricEnabled) {
            const volumetricPass = encoder.beginRenderPass({
                label: "Volumetric Lighting Generator Pass",
                colorAttachments: [ { view: this.volumetricTextureView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" } ]
            });
            volumetricPass.setPipeline(this.volumetricPipeline);
            volumetricPass.setBindGroup(0, this.volumetricBindGroup);
            volumetricPass.draw(3);
            volumetricPass.end();

            const compositePass = encoder.beginRenderPass({
                label: "Volumetric Composite Pass",
                colorAttachments: [ { view: canvasTextureView, loadOp: "load", storeOp: "store" } ]
            });
            compositePass.setPipeline(this.volumetricCompositePipeline);
            compositePass.setBindGroup(0, this.volumetricCompositeBindGroup);
            compositePass.draw(3);
            compositePass.end();
        }

        renderer.device.queue.submit([encoder.finish()]);
    }

    // Sub-classes implement these
    protected abstract createShadingBindGroup(): void;
    protected abstract executeShadingPass(encoder: GPUCommandEncoder, canvasTextureView: GPUTextureView): void;
}
