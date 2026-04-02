import { device } from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from './camera';
import { Environment } from './environment';
import { pipelineCache } from '../engine/PipelineCache';

/**
 * Radiance Cascades manager.
 * Replaces DDGI with deterministic inline hierarchy of radiance evaluation.
 *
 * GPU resources are lazily allocated on first enable to save memory.
 */
export class RadianceCascades {
    static readonly GRID_X = shaders.constants.rcProbeGridX;
    static readonly GRID_Y = shaders.constants.rcProbeGridY;
    static readonly GRID_Z = shaders.constants.rcProbeGridZ;
    static readonly TOTAL_PROBES = RadianceCascades.GRID_X * RadianceCascades.GRID_Y * RadianceCascades.GRID_Z;

    static readonly TEXELS = shaders.constants.rcIrradianceTexels; // 8
    static readonly TEXELS_WITH_BORDER = RadianceCascades.TEXELS + 2; // 10

    static readonly PROBES_PER_ROW = 800; // Fixed width to prevent exceeding WebGPU maxTextureDimension2D (8192)
    static readonly ATLAS_W = RadianceCascades.PROBES_PER_ROW * RadianceCascades.TEXELS_WITH_BORDER;
    static readonly ATLAS_H = Math.ceil(RadianceCascades.TOTAL_PROBES / RadianceCascades.PROBES_PER_ROW) * RadianceCascades.TEXELS_WITH_BORDER;

    // World-space bounds
    gridMin: [number, number, number] = [-14, -4, -7];
    gridMax: [number, number, number] = [14, 16, 7];

    hysteresis = 0.95;
    intensity = 1.0;
    ambient = 0.05;
    enabled = false;
    debugMode = 0; // 0=Off, 1=GI Only, 2=Atlas Overlay

    // GPU resources (lazily allocated)
    private _initialized = false;

    rcAtlasA!: GPUTexture;
    rcAtlasB!: GPUTexture;
    rcAtlasAView!: GPUTextureView;
    rcAtlasBView!: GPUTextureView;

    private pingPong = 0;

    rcUniformBuffer: GPUBuffer;
    rcSampler: GPUSampler;

    tracePipeline!: GPUComputePipeline;
    traceLayout!: GPUBindGroupLayout;

    // Dummy resources for bind group compatibility when not initialized
    private dummyTextureView: GPUTextureView;

    private camera: Camera;
    private environment: Environment;

    constructor(camera: Camera, environment: Environment) {
        this.camera = camera;
        this.environment = environment;

        // Create minimal dummy resources for bind group compatibility
        const dummyTex = device.createTexture({
            label: "RC Dummy Texture",
            size: [1, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.dummyTextureView = dummyTex.createView();

        // Uniform buffer is always needed (shader reads rc_enabled flag)
        this.rcUniformBuffer = device.createBuffer({
            label: "RC Uniforms",
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sampler is always needed for bind groups
        this.rcSampler = device.createSampler({
            label: "RC Atlas Sampler",
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        this.updateUniforms();

        console.log(`Radiance Cascades configured: ${RadianceCascades.TOTAL_PROBES} Probes (resources deferred)`);
    }

    private ensureResources() {
        if (this._initialized) return;
        this._initialized = true;

        console.log(`Radiance Cascades: allocating GPU resources...`);

        const atlasDesc: GPUTextureDescriptor = {
            label: "RC Atlas",
            size: [RadianceCascades.ATLAS_W, RadianceCascades.ATLAS_H],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        };
        this.rcAtlasA = device.createTexture({ ...atlasDesc, label: "RC Atlas A" });
        this.rcAtlasB = device.createTexture({ ...atlasDesc, label: "RC Atlas B" });
        this.rcAtlasAView = this.rcAtlasA.createView();
        this.rcAtlasBView = this.rcAtlasB.createView();

        this.traceLayout = this.createTraceLayout();
        this.tracePipeline = pipelineCache.getComputePipeline(device, {
            label: "RC Trace Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.traceLayout] }),
            compute: { module: device.createShaderModule({ code: shaders.rcTraceSrc }), entryPoint: 'main' }
        }, "RC_Trace");

        console.log(`Radiance Cascades initialized: ${RadianceCascades.TOTAL_PROBES} Probes, Atlas ${RadianceCascades.ATLAS_W}x${RadianceCascades.ATLAS_H}`);
    }

    private uniformData = new ArrayBuffer(128);

    updateUniforms() {
        const spacing = [
            (this.gridMax[0] - this.gridMin[0]) / (RadianceCascades.GRID_X - 1),
            (this.gridMax[1] - this.gridMin[1]) / (RadianceCascades.GRID_Y - 1),
            (this.gridMax[2] - this.gridMin[2]) / (RadianceCascades.GRID_Z - 1),
        ];

        const data = this.uniformData;
        const i32View = new Int32Array(data);
        const f32View = new Float32Array(data);

        // rcParams structure:
        // grid_count (vec4i)
        // grid_min (vec4f)
        // grid_max (vec4f)
        // grid_spacing (vec4f)
        // atlas_dims (vec4f)
        // rc_params (vec4f: hysteresis, intensity, ambient, enabled)
        
        i32View[0] = RadianceCascades.GRID_X; i32View[1] = RadianceCascades.GRID_Y; i32View[2] = RadianceCascades.GRID_Z; i32View[3] = RadianceCascades.TOTAL_PROBES;
        f32View[4] = this.gridMin[0]; f32View[5] = this.gridMin[1]; f32View[6] = this.gridMin[2]; f32View[7] = 0;
        f32View[8] = this.gridMax[0]; f32View[9] = this.gridMax[1]; f32View[10] = this.gridMax[2]; f32View[11] = 0;
        f32View[12] = spacing[0]; f32View[13] = spacing[1]; f32View[14] = spacing[2]; f32View[15] = 0;
        f32View[16] = RadianceCascades.TEXELS; f32View[17] = RadianceCascades.TEXELS_WITH_BORDER; f32View[18] = RadianceCascades.ATLAS_W; f32View[19] = RadianceCascades.ATLAS_H;
        f32View[20] = this.hysteresis; f32View[21] = this.intensity; f32View[22] = this.ambient; f32View[23] = this.enabled ? 1.0 : 0.0;
        f32View[24] = this.debugMode; f32View[25] = 0; f32View[26] = 0; f32View[27] = 0;
        
        device.queue.writeBuffer(this.rcUniformBuffer, 0, data);
    }

    private createTraceLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "RC Trace Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: 'cube' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            ]
        });
    }

    update(encoder: GPUCommandEncoder, voxelGridView: GPUTextureView, sunLightBuffer: GPUBuffer, shadowMapView: GPUTextureView, vsmUniformBuffer: GPUBuffer) {
        if (!this.enabled) return;

        // Lazy allocation on first use
        this.ensureResources();

        this.updateUniforms();

        const readAtlas = this.pingPong === 0 ? this.rcAtlasAView : this.rcAtlasBView;
        const writeAtlas = this.pingPong === 0 ? this.rcAtlasBView : this.rcAtlasAView;

        const traceBindGroup = device.createBindGroup({
            layout: this.traceLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.rcUniformBuffer } },
                { binding: 2, resource: voxelGridView },
                { binding: 3, resource: this.environment.envCubemapView },
                { binding: 4, resource: this.environment.envSampler },
                { binding: 5, resource: { buffer: sunLightBuffer } },
                { binding: 6, resource: shadowMapView },
                { binding: 7, resource: { buffer: vsmUniformBuffer } },
                { binding: 8, resource: readAtlas },
                { binding: 9, resource: writeAtlas },
            ]
        });

        const tracePass = encoder.beginComputePass({ label: "RC Trace" });
        tracePass.setPipeline(this.tracePipeline);
        tracePass.setBindGroup(0, traceBindGroup);
        // Dispatch as 2D grid to bypass WebGPU maxComputeWorkgroupsPerDimension (65535) limit
        tracePass.dispatchWorkgroups(RadianceCascades.GRID_X, RadianceCascades.GRID_Y * RadianceCascades.GRID_Z, 1);
        tracePass.end();

        this.pingPong = 1 - this.pingPong;
    }

    /**
     * Returns the current irradiance view. Before resources are allocated, returns a 1×1 dummy view.
     */
    getCurrentIrradianceView(): GPUTextureView {
        if (!this._initialized) return this.dummyTextureView;
        return this.pingPong === 0 ? this.rcAtlasBView : this.rcAtlasAView;
    }
}
