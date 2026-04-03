import { device } from "../renderer";

export class SSR {
    enabled: boolean = false;
    maxDistance: number = 100.0;
    resolutionScale: number = 0.5;
    maxSteps: number = 64;
    thickness: number = 0.5;

    uniformsBuffer: GPUBuffer;
    private uniformData = new Float32Array(8);
    
    constructor() {
        // enabled(1) + maxDistance(1) + resolutionScale(1) + maxSteps(1)
        // thickness(1) + padding(3)
        // total = 8 floats (32 bytes)
        this.uniformsBuffer = device.createBuffer({
            label: "SSR Uniforms",
            size: 32, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.updateUniforms();
    }
    
    updateUniforms() {
        this.uniformData[0] = this.enabled ? 1.0 : 0.0;
        this.uniformData[1] = this.maxDistance;
        this.uniformData[2] = this.resolutionScale;
        this.uniformData[3] = this.maxSteps;
        this.uniformData[4] = this.thickness;
        // 5, 6, 7 are padding
        
        device.queue.writeBuffer(this.uniformsBuffer, 0, this.uniformData.buffer);
    }
}
