import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';
import { DDGI } from '../../stage/ddgi';

export interface DDGIDebugPassDeps {
    cameraBindGroupLayout: GPUBindGroupLayout;
    cameraBindGroup: GPUBindGroup;
    ddgi: DDGI;
}

export class DDGIDebugPass {
    private pipeline!: GPURenderPipeline;
    private bindGroupLayout!: GPUBindGroupLayout;
    private bindGroup!: GPUBindGroup | null;
    
    // We recreate bind group dynamically because DDGI ping-pongs its textures
    private initialized = false;

    constructor() {
    }

    private init(deps: DDGIDebugPassDeps) {
        if (this.initialized) return;

        this.bindGroupLayout = renderer.device.createBindGroupLayout({
            label: "DDGI Debug Probes BGL",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // ddgiUniforms
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // probeData
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } }, // irradiance atlas
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // ddgiSampler
            ]
        });

        this.pipeline = renderer.device.createRenderPipeline({
            label: "DDGI Debug Probes Pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [deps.cameraBindGroupLayout, this.bindGroupLayout] }),
            vertex: { 
                module: renderer.device.createShaderModule({ label: "DDGI Debug VS", code: shaders.ddgiDebugProbesSrc }), 
                entryPoint: "vs_main" 
            },
            fragment: { 
                module: renderer.device.createShaderModule({ label: "DDGI Debug FS", code: shaders.ddgiDebugProbesSrc }), 
                entryPoint: "fs_main", 
                targets: [{ format: renderer.canvasFormat }] 
            },
            primitive: { topology: "triangle-list", cullMode: 'back' },
            depthStencil: { depthWriteEnabled: true, depthCompare: "greater", format: "depth24plus" }
        });

        this.initialized = true;
    }

    execute(
        pass: GPURenderPassEncoder,
        deps: DDGIDebugPassDeps
    ) {
        if (!deps.ddgi.enabled || !deps.ddgi.irradianceAtlasAView) return;

        this.init(deps);

        // Recreate bind group every frame due to ping-ponging irradiance view
        this.bindGroup = renderer.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: deps.ddgi.ddgiUniformBuffer } },
                { binding: 1, resource: { buffer: deps.ddgi.probeDataBuffer } },
                { binding: 2, resource: deps.ddgi.getCurrentIrradianceView() },
                { binding: 3, resource: deps.ddgi.ddgiSampler }
            ]
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, deps.cameraBindGroup);
        pass.setBindGroup(1, this.bindGroup);
        
        // Sphere index count per probe (ico sphere 3 subdiv) is ~240 indices, max probes is 22*22*22 = 10648
        // Instanced draw
        const sphereIndices = 240 * 3; // Approx 240 triangles = 720 vertices
        const instanceCount = (deps.ddgi as any).gridDimensions[0] * (deps.ddgi as any).gridDimensions[1] * (deps.ddgi as any).gridDimensions[2];
        
        pass.draw(sphereIndices, instanceCount);
    }
}
