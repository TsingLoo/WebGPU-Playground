import { device } from '../renderer';
import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from './camera';
import { pipelineCache } from '../engine/PipelineCache';

/**
 * Virtual Shadow Map (VSM) Manager
 * 
 * Implements clipmap-based virtual shadow mapping for directional light.
 * Uses GPU-driven page marking and allocation to only render shadow depth
 * for visible pages, achieving high-resolution shadows across large distances.
 * 
 * Architecture:
 *   1. Clear pass: reset page flags & allocation state
 *   2. Mark pass: depth buffer → virtual page requests  
 *   3. Allocate pass: assign physical pages from pool
 *   4. Render pass: rasterize shadow depth per clipmap level
 *   5. Sampling: fragment shader reads page table + physical atlas
 */
export class VSM {
    // --- Configuration (adjustable at runtime via GUI) ---
    pageSize: number = 128;              // texels per page axis
    physAtlasSize: number = 4096;        // physical atlas texels per axis
    numClipmapLevels: number = 6;
    pagesPerLevelAxis: number = 128;     // virtual pages per level per axis

    // --- Derived (recomputed on config change) ---
    get physPagesPerAxis(): number { return Math.floor(this.physAtlasSize / this.pageSize); }
    get maxPhysPages(): number { return this.physPagesPerAxis * this.physPagesPerAxis; }
    get virtualSize(): number { return this.pagesPerLevelAxis * this.pageSize; }
    get totalVirtualPages(): number { return this.numClipmapLevels * this.pagesPerLevelAxis * this.pagesPerLevelAxis; }

    // Static defaults for backward compatibility where static access is needed
    static readonly PAGE_SIZE = 128;
    static readonly PHYS_ATLAS_SIZE = 4096;
    static readonly PHYS_PAGES_PER_AXIS = 32;
    static readonly MAX_PHYS_PAGES = 1024;
    static readonly NUM_CLIPMAP_LEVELS = 6;
    static readonly PAGES_PER_LEVEL_AXIS = 128;
    static readonly VIRTUAL_SIZE = 16384;
    static readonly TOTAL_VIRTUAL_PAGES = 6 * 128 * 128;

    // --- GPU Resources ---
    // Physical depth atlas
    physicalAtlas!: GPUTexture;
    physicalAtlasView!: GPUTextureView;
    shadowComparisonSampler!: GPUSampler;

    // Page table: virtual page → physical page index (u32 per page)
    pageTableBuffer!: GPUBuffer;
    // Page request flags: atomically written marks
    pageRequestBuffer!: GPUBuffer;
    // Allocation state: counter + debug info
    allocStateBuffer!: GPUBuffer;

    // VSM uniform buffer: clipmap VP matrices + params
    // Layout: 8 × mat4x4f (512 bytes) + mat4x4f inv_view_proj (64 bytes) + 4 × u32 params (16 bytes) = 592 bytes
    static readonly MAX_CLIPMAP_LEVELS = 8;
    vsmUniformBuffer!: GPUBuffer;

    // Light VP buffer per clipmap level (for shadow rendering)
    clipmapVPBuffers!: GPUBuffer[];

    // --- Pipelines ---
    clearPipeline!: GPUComputePipeline;
    clearBindGroupLayout!: GPUBindGroupLayout;

    markPagesPipeline!: GPUComputePipeline;
    markPagesBindGroupLayout!: GPUBindGroupLayout;

    allocatePagesPipeline!: GPUComputePipeline;
    allocateBindGroupLayout!: GPUBindGroupLayout;

    shadowPipeline!: GPURenderPipeline;
    shadowBindGroupLayout!: GPUBindGroupLayout;

    // --- State ---
    private camera: Camera;
    sunDirection: [number, number, number] = [0.5, 0.8, 0.3];
    sceneBoundsMin: [number, number, number] = [-15, -2, -8];
    sceneBoundsMax: [number, number, number] = [15, 10, 8];

    constructor(camera: Camera) {
        this.camera = camera;
        this.createGPUResources();
        this.createComputePipelines();
        this.createShadowPipeline();
    }

    /**
     * Recreate all GPU resources after config change.
     * Call this after modifying pageSize, physAtlasSize, numClipmapLevels, or pagesPerLevelAxis.
     */
    recreate() {
        // Capture old resources — they may still be referenced by in-flight GPU commands
        const oldAtlas = this.physicalAtlas;
        const oldPageTable = this.pageTableBuffer;
        const oldPageRequest = this.pageRequestBuffer;
        const oldAllocState = this.allocStateBuffer;
        const oldUniforms = this.vsmUniformBuffer;
        const oldVPBuffers = this.clipmapVPBuffers;

        // Create new resources immediately
        this.createGPUResources();
        this.createComputePipelines();
        this.createShadowPipeline();

        // Invalidate cached bind groups — they reference old buffers
        this.cachedClearBG = null;
        this.cachedAllocBG = null;
        this.cachedMarkBG = null;
        this.cachedMarkDepthView = null;
        this.cachedShadowBGs = [];

        // Defer destruction of old resources until the GPU finishes processing
        // any previously submitted commands that may still reference them
        device.queue.onSubmittedWorkDone().then(() => {
            oldAtlas.destroy();
            oldPageTable.destroy();
            oldPageRequest.destroy();
            oldAllocState.destroy();
            oldUniforms.destroy();
            for (const buf of oldVPBuffers) buf.destroy();
        });
    }

    private createGPUResources() {
        // Physical depth atlas
        this.physicalAtlas = device.createTexture({
            label: "VSM Physical Atlas",
            size: [this.physAtlasSize, this.physAtlasSize],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.physicalAtlasView = this.physicalAtlas.createView();

        // Comparison sampler for PCF
        this.shadowComparisonSampler = device.createSampler({
            label: "VSM Comparison Sampler",
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Page table buffer
        this.pageTableBuffer = device.createBuffer({
            label: "VSM Page Table",
            size: this.totalVirtualPages * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Page request flags
        this.pageRequestBuffer = device.createBuffer({
            label: "VSM Page Request Flags",
            size: this.totalVirtualPages * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Allocation state (counter + total_requested = 8 bytes, pad to 16)
        this.allocStateBuffer = device.createBuffer({
            label: "VSM Allocation State",
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // VSM uniform buffer
        // 8 mat4x4f (8*64=512) + inv_view_proj mat4x4f (64) + 4 u32 params (16) = 592
        this.vsmUniformBuffer = device.createBuffer({
            label: "VSM Uniforms",
            size: 592,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Per-clipmap-level VP buffer for shadow render pass
        this.clipmapVPBuffers = [];
        for (let i = 0; i < this.numClipmapLevels; i++) {
            this.clipmapVPBuffers.push(device.createBuffer({
                label: `VSM Clipmap VP Level ${i}`,
                size: 64,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }
    }

    private createComputePipelines() {
        // --- Clear Pipeline ---
        this.clearBindGroupLayout = device.createBindGroupLayout({
            label: "VSM Clear BGL",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // pageRequestFlags
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // allocState
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // pageTable
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // vsmUniforms
            ],
        });

        this.clearPipeline = pipelineCache.getComputePipeline(device, {
            label: "VSM Clear Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.clearBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ label: "VSM Clear", code: shaders.vsmClearSrc }),
                entryPoint: 'main',
            },
        }, "VSM_Clear");

        // --- Mark Pages Pipeline ---
        this.markPagesBindGroupLayout = device.createBindGroupLayout({
            label: "VSM Mark Pages BGL",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // camera
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },  // depthTex
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // pageRequestFlags
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // vsmUniforms
            ],
        });

        this.markPagesPipeline = pipelineCache.getComputePipeline(device, {
            label: "VSM Mark Pages Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.markPagesBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ label: "VSM Mark Pages", code: shaders.vsmMarkPagesSrc }),
                entryPoint: 'main',
            },
        }, "VSM_MarkPages");

        // --- Allocate Pages Pipeline ---
        this.allocateBindGroupLayout = device.createBindGroupLayout({
            label: "VSM Allocate Pages BGL",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // pageRequestFlags (read)
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // pageTable
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // allocState
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // vsmUniforms
            ],
        });

        this.allocatePagesPipeline = pipelineCache.getComputePipeline(device, {
            label: "VSM Allocate Pages Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.allocateBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ label: "VSM Allocate Pages", code: shaders.vsmAllocatePagesSrc }),
                entryPoint: 'main',
            },
        }, "VSM_AllocatePages");
    }

    private createShadowPipeline() {
        // Shadow render pipeline — renders depth per clipmap level
        this.shadowBindGroupLayout = device.createBindGroupLayout({
            label: "VSM Shadow BGL",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },  // lightViewProj
            ],
        });

        this.shadowPipeline = pipelineCache.getRenderPipeline(device, {
            label: "VSM Shadow Pipeline",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadowBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout,
                ],
            }),
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
                depthBias: 4,
                depthBiasSlopeScale: 2.5,
            },
            vertex: {
                module: device.createShaderModule({ label: "VSM Shadow VS", code: shaders.shadowVertSrc }),
                entryPoint: 'main',
                buffers: [renderer.vertexBufferLayout],
            },
            fragment: {
                module: device.createShaderModule({ label: "VSM Shadow FS", code: shaders.shadowFragSrc }),
                entryPoint: 'main',
                targets: [],
            },
        }, "VSM_Shadow");
    }

    /**
     * Compute clipmap VP matrices centered on the camera position.
     * Each level covers a radius of 2^(level + 4) units from the camera.
     * Returns an array of 6 Float32Array(16) VP matrices.
     */
    computeClipmapVPMatricesInPlace(cameraPos: [number, number, number]) {
        const d = this.sunDirection;
        const len = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const lightDir = [d[0] / len, d[1] / len, d[2] / len]; // direction TO light

        for (let level = 0; level < this.numClipmapLevels; level++) {
            const radius = Math.pow(2, level + 4); // 16, 32, 64, 128, 256, 512

            // Forward direction (eye toward center) = -lightDir
            const fwd = [-lightDir[0], -lightDir[1], -lightDir[2]];

            // Up vector
            let up = [0, 1, 0];
            if (Math.abs(fwd[1]) > 0.99) {
                up = [0, 0, 1];
            }

            // Right = fwd × up
            const right = [
                fwd[1] * up[2] - fwd[2] * up[1],
                fwd[2] * up[0] - fwd[0] * up[2],
                fwd[0] * up[1] - fwd[1] * up[0],
            ];
            const rLen = Math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
            right[0] /= rLen; right[1] /= rLen; right[2] /= rLen;

            // Recalculate up = right × fwd
            up = [
                right[1] * fwd[2] - right[2] * fwd[1],
                right[2] * fwd[0] - right[0] * fwd[2],
                right[0] * fwd[1] - right[1] * fwd[0],
            ];

            // ---- Snap in light-space to eliminate sub-texel jitter ----
            const camDotRight = right[0] * cameraPos[0] + right[1] * cameraPos[1] + right[2] * cameraPos[2];
            const camDotUp = up[0] * cameraPos[0] + up[1] * cameraPos[1] + up[2] * cameraPos[2];
            const camDotFwd = fwd[0] * cameraPos[0] + fwd[1] * cameraPos[1] + fwd[2] * cameraPos[2];

            const gridCols = Math.ceil(Math.sqrt(this.numClipmapLevels));
            const tilePixels = Math.floor(this.physAtlasSize / gridCols);
            const texelWorldSize = (2 * radius) / tilePixels;

            const snappedRight = Math.floor(camDotRight / texelWorldSize) * texelWorldSize;
            const snappedUp = Math.floor(camDotUp / texelWorldSize) * texelWorldSize;

            const snappedCenter = [
                snappedRight * right[0] + snappedUp * up[0] + camDotFwd * fwd[0],
                snappedRight * right[1] + snappedUp * up[1] + camDotFwd * fwd[1],
                snappedRight * right[2] + snappedUp * up[2] + camDotFwd * fwd[2],
            ];

            const lightDist = radius * 2;
            const eyeX = snappedCenter[0] + lightDir[0] * lightDist;
            const eyeY = snappedCenter[1] + lightDir[1] * lightDist;
            const eyeZ = snappedCenter[2] + lightDir[2] * lightDist;

            // View matrix (column-major) — reuse pre-allocated buffer
            const view = this.vpViewMat;
            view[0] = right[0]; view[1] = up[0]; view[2] = -fwd[0]; view[3] = 0;
            view[4] = right[1]; view[5] = up[1]; view[6] = -fwd[1]; view[7] = 0;
            view[8] = right[2]; view[9] = up[2]; view[10] = -fwd[2]; view[11] = 0;
            view[12] = -(right[0] * eyeX + right[1] * eyeY + right[2] * eyeZ);
            view[13] = -(up[0] * eyeX + up[1] * eyeY + up[2] * eyeZ);
            view[14] = (fwd[0] * eyeX + fwd[1] * eyeY + fwd[2] * eyeZ);
            view[15] = 1;

            // Orthographic projection — reuse pre-allocated buffer
            const nearZ = 0;
            const farZ = lightDist * 2;
            const ortho = this.vpOrthoMat;
            ortho.fill(0);
            ortho[0] = 1 / radius;
            ortho[5] = 1 / radius;
            ortho[10] = 1 / (nearZ - farZ);
            ortho[14] = -nearZ / (nearZ - farZ);
            ortho[15] = 1;

            // VP = ortho × view — write to pool
            const vp = this.vpMatricesPool[level];
            for (let col = 0; col < 4; col++) {
                for (let row = 0; row < 4; row++) {
                    let sum = 0;
                    for (let k = 0; k < 4; k++) {
                        sum += ortho[k * 4 + row] * view[col * 4 + k];
                    }
                    vp[col * 4 + row] = sum;
                }
            }
        }
    }

    private uniformData = new ArrayBuffer(592);
    private uniformF32 = new Float32Array(this.uniformData);
    private uniformU32 = new Uint32Array(this.uniformData);
    private invViewProjOut = new Float32Array(16);

    // Pre-allocated VP matrices to avoid per-frame allocation
    private vpMatricesPool: Float32Array[] = [];
    private vpViewMat = new Float32Array(16);
    private vpOrthoMat = new Float32Array(16);

    private ensureVPPool() {
        if (this.vpMatricesPool.length < this.numClipmapLevels) {
            this.vpMatricesPool = [];
            for (let i = 0; i < this.numClipmapLevels; i++) {
                this.vpMatricesPool.push(new Float32Array(16));
            }
        }
    }

    /**
     * Update VSM uniform buffer with current clipmap matrices and params.
     */
    updateUniforms(cameraPos: [number, number, number]) {
        this.ensureVPPool();
        this.computeClipmapVPMatricesInPlace(cameraPos);

        const viewProjMat = new Float32Array(this.camera.uniforms.buffer, 0, 16);
        this.invertMatrix4InPlace(viewProjMat, this.invViewProjOut);

        const f32 = this.uniformF32;
        const u32 = this.uniformU32;

        for (let i = 0; i < this.numClipmapLevels; i++) {
            f32.set(this.vpMatricesPool[i], i * 16);
        }

        const vpOffset = VSM.MAX_CLIPMAP_LEVELS * 16;
        f32.set(this.invViewProjOut, vpOffset);

        const paramsOffset = vpOffset + 16;
        u32[paramsOffset + 0] = this.numClipmapLevels;
        u32[paramsOffset + 1] = this.pagesPerLevelAxis;
        u32[paramsOffset + 2] = this.physAtlasSize;
        u32[paramsOffset + 3] = this.physPagesPerAxis;

        device.queue.writeBuffer(this.vsmUniformBuffer, 0, this.uniformData);

        for (let i = 0; i < this.numClipmapLevels; i++) {
            device.queue.writeBuffer(this.clipmapVPBuffers[i], 0, this.vpMatricesPool[i].buffer);
        }
    }

    /**
     * Runs the full VSM pipeline: clear → mark → allocate → render shadows.
     */
    // Cached bind groups for compute passes (buffers never change between frames)
    private cachedClearBG: GPUBindGroup | null = null;
    private cachedAllocBG: GPUBindGroup | null = null;
    // Mark pass BG is per-depthTextureView, cache last one
    private cachedMarkBG: GPUBindGroup | null = null;
    private cachedMarkDepthView: GPUTextureView | null = null;
    // Shadow pass per-level bind groups
    private cachedShadowBGs: GPUBindGroup[] = [];

    // Pre-allocated workgroup counts
    private clearWorkgroups = 0;
    private allocWorkgroups = 0;
    private markWorkgroupsX = 0;
    private markWorkgroupsY = 0;

    private ensureBindGroups(depthTextureView: GPUTextureView) {
        if (!this.cachedClearBG) {
            this.cachedClearBG = device.createBindGroup({
                layout: this.clearBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.pageRequestBuffer } },
                    { binding: 1, resource: { buffer: this.allocStateBuffer } },
                    { binding: 2, resource: { buffer: this.pageTableBuffer } },
                    { binding: 3, resource: { buffer: this.vsmUniformBuffer } },
                ],
            });
            this.cachedAllocBG = device.createBindGroup({
                layout: this.allocateBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.pageRequestBuffer } },
                    { binding: 1, resource: { buffer: this.pageTableBuffer } },
                    { binding: 2, resource: { buffer: this.allocStateBuffer } },
                    { binding: 3, resource: { buffer: this.vsmUniformBuffer } },
                ],
            });
            this.clearWorkgroups = Math.ceil(this.totalVirtualPages / 256);
            this.allocWorkgroups = Math.ceil(this.totalVirtualPages / 256);
            this.markWorkgroupsX = Math.ceil(renderer.canvas.width / 8);
            this.markWorkgroupsY = Math.ceil(renderer.canvas.height / 8);

            // Cache shadow per-level bind groups
            this.cachedShadowBGs = [];
            for (let i = 0; i < this.numClipmapLevels; i++) {
                this.cachedShadowBGs.push(device.createBindGroup({
                    layout: this.shadowBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.clipmapVPBuffers[i] } },
                    ],
                }));
            }
        }
        if (this.cachedMarkDepthView !== depthTextureView) {
            this.cachedMarkBG = device.createBindGroup({
                layout: this.markPagesBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                    { binding: 1, resource: depthTextureView },
                    { binding: 2, resource: { buffer: this.pageRequestBuffer } },
                    { binding: 3, resource: { buffer: this.vsmUniformBuffer } },
                ],
            });
            this.cachedMarkDepthView = depthTextureView;
        }
    }

    update(
        encoder: GPUCommandEncoder,
        depthTextureView: GPUTextureView,
        scene: import('../engine/Scene').Scene,
        cameraPos: [number, number, number],
    ) {
        this.updateUniforms(cameraPos);
        this.ensureBindGroups(depthTextureView);

        // 1. Clear pass
        const clearPass = encoder.beginComputePass({ label: "VSM Clear" });
        clearPass.setPipeline(this.clearPipeline);
        clearPass.setBindGroup(0, this.cachedClearBG!);
        clearPass.dispatchWorkgroups(this.clearWorkgroups);
        clearPass.end();

        // 2. Mark pages pass
        const markPass = encoder.beginComputePass({ label: "VSM Mark Pages" });
        markPass.setPipeline(this.markPagesPipeline);
        markPass.setBindGroup(0, this.cachedMarkBG!);
        markPass.dispatchWorkgroups(this.markWorkgroupsX, this.markWorkgroupsY);
        markPass.end();

        // 3. Allocate pages pass
        const allocPass = encoder.beginComputePass({ label: "VSM Allocate Pages" });
        allocPass.setPipeline(this.allocatePagesPipeline);
        allocPass.setBindGroup(0, this.cachedAllocBG!);
        allocPass.dispatchWorkgroups(this.allocWorkgroups);
        allocPass.end();

        // 4. Render shadow depth per clipmap level
        this.renderShadowLevels(encoder, scene);
    }

    /**
     * Render shadow depth for all clipmap levels into the physical atlas.
     * Each level renders the entire scene with its VP matrix into the full atlas.
     * The hardware depth test ensures only closest geometry is kept per texel.
     */
    private renderShadowLevels(encoder: GPUCommandEncoder, scene: import('../engine/Scene').Scene) {
        // Single pass rendering of all clipmap levels into the atlas
        // We render each level separately so the VP matrix is correct
        const shadowPass = encoder.beginRenderPass({
            label: "VSM Shadow Render",
            colorAttachments: [],
            depthStencilAttachment: {
                view: this.physicalAtlasView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        shadowPass.setPipeline(this.shadowPipeline);

        // Square grid layout: arrange levels as ceil(sqrt(N)) columns
        // This ensures each level gets a square tile instead of a stretched band
        const gridCols = Math.ceil(Math.sqrt(this.numClipmapLevels));
        const tileSize = Math.floor(this.physAtlasSize / gridCols);

        for (let level = 0; level < this.numClipmapLevels; level++) {
            shadowPass.setBindGroup(0, this.cachedShadowBGs[level]);

            // Grid position for this level
            const col = level % gridCols;
            const row = Math.floor(level / gridCols);
            const xOffset = col * tileSize;
            const yOffset = row * tileSize;
            shadowPass.setViewport(xOffset, yOffset, tileSize, tileSize, 0, 1);
            shadowPass.setScissorRect(xOffset, yOffset, tileSize, tileSize);

            scene.iterate(
                mr => { shadowPass.setBindGroup(1, mr.modelBindGroup!); },
                material => { shadowPass.setBindGroup(2, material.materialBindGroup); },
                primitive => {
                    shadowPass.setVertexBuffer(0, primitive.vertexBuffer);
                    shadowPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                    shadowPass.drawIndexed(primitive.numIndices);
                },
            );
        }

        shadowPass.end();
    }

    /**
     * 4x4 matrix inversion (column-major).
     */
    private invertMatrix4InPlace(m: Float32Array, out: Float32Array) {
        const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
        const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
        const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
        const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

        const b00 = a00 * a11 - a01 * a10;
        const b01 = a00 * a12 - a02 * a10;
        const b02 = a00 * a13 - a03 * a10;
        const b03 = a01 * a12 - a02 * a11;
        const b04 = a01 * a13 - a03 * a11;
        const b05 = a02 * a13 - a03 * a12;
        const b06 = a20 * a31 - a21 * a30;
        const b07 = a20 * a32 - a22 * a30;
        const b08 = a20 * a33 - a23 * a30;
        const b09 = a21 * a32 - a22 * a31;
        const b10 = a21 * a33 - a23 * a31;
        const b11 = a22 * a33 - a23 * a32;

        let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        if (!det) return;
        det = 1.0 / det;

        out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
        out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
        out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
        out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
        out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
        out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
        out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
        out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
        out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
        out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
        out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
        out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
        out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
        out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
        out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
        out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    }
}
