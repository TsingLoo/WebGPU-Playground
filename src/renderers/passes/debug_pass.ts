import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export interface DebugPassDeps {
    cameraBindGroupLayout: GPUBindGroupLayout;
    cameraBindGroup: GPUBindGroup;
}

export class DebugPass {
    private pipeline: GPURenderPipeline;
    private bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup;
    private boxBuffer: GPUBuffer;

    constructor(deps: DebugPassDeps) {
        this.bindGroupLayout = renderer.device.createBindGroupLayout({
            label: "debug box bgl",
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }]
        });
        this.pipeline = renderer.device.createRenderPipeline({
            label: "debug box pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [deps.cameraBindGroupLayout, this.bindGroupLayout] }),
            vertex: { module: renderer.device.createShaderModule({ code: shaders.debugBoxSrc }), entryPoint: "vs_main" },
            fragment: { module: renderer.device.createShaderModule({ code: shaders.debugBoxSrc }), entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
            primitive: { topology: "line-list" },
            depthStencil: { depthWriteEnabled: false, depthCompare: "greater-equal", format: "depth24plus" }
        });

        // box buffer: minPos vec4 + maxPos vec4 + color vec4 -> 48 bytes
        this.boxBuffer = renderer.device.createBuffer({
            size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.bindGroup = renderer.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.boxBuffer } }]
        });
    }

    execute(
        pass: GPURenderPassEncoder,
        cameraBindGroup: GPUBindGroup,
        minPos: number[],
        maxPos: number[],
        color: number[]
    ) {
        const boxData = new Float32Array([...minPos, 0, ...maxPos, 0, ...color]);
        renderer.device.queue.writeBuffer(this.boxBuffer, 0, boxData);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, cameraBindGroup);
        pass.setBindGroup(1, this.bindGroup);
        pass.draw(24);
    }
}
