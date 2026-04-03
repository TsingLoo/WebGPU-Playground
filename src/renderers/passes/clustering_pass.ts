import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export interface ClusteringPassDeps {
    cameraBuffer: GPUBuffer;
    lightSetStorageBuffer: GPUBuffer;
    tileOffsetsBuffer: GPUBuffer;
    globalLightIndicesBuffer: GPUBuffer;
    clusterSetBuffer: GPUBuffer;
    zeroBuffer: GPUBuffer;
    hizTextureView: GPUTextureView;
}

export class ClusteringPass {
    private pipeline: GPUComputePipeline;
    private bindGroup: GPUBindGroup;
    private zeroBuffer: GPUBuffer;
    private globalLightIndicesBuffer: GPUBuffer;

    constructor(deps: ClusteringPassDeps) {
        this.zeroBuffer = deps.zeroBuffer;
        this.globalLightIndicesBuffer = deps.globalLightIndicesBuffer;

        const bindGroupLayout = renderer.device.createBindGroupLayout({
            label: "culling bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } }
            ]
        });
        this.bindGroup = renderer.device.createBindGroup({
            label: "culling bind group",
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: deps.cameraBuffer } },
                { binding: 1, resource: { buffer: deps.lightSetStorageBuffer } },
                { binding: 2, resource: { buffer: deps.tileOffsetsBuffer } },
                { binding: 3, resource: { buffer: deps.globalLightIndicesBuffer } },
                { binding: 4, resource: { buffer: deps.clusterSetBuffer } },
                { binding: 5, resource: deps.hizTextureView }
            ]
        });
        this.pipeline = renderer.device.createComputePipeline({
            label: "culling compute pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: renderer.device.createShaderModule({ code: shaders.clusteringComputeSrc }), entryPoint: "main" }
        });
    }

    execute(encoder: GPUCommandEncoder) {
        // Reset atomic counter
        encoder.copyBufferToBuffer(this.zeroBuffer, 0, this.globalLightIndicesBuffer, 0, 4);
        
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(shaders.constants.bindGroup_scene, this.bindGroup);
        pass.dispatchWorkgroups(
            shaders.constants.numClustersX,
            shaders.constants.numClustersY,
            shaders.constants.numClustersZ
        );
        pass.end();
    }
}
