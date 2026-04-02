import { device } from "../renderer";

export class SSAO {
    enabled: boolean = true;
    radius: number = 0.5;
    bias: number = 0.025;
    power: number = 1.0;

    uniformsBuffer: GPUBuffer;
    
    constructor() {
        // radius(1) + bias(1) + power(1) + enabled(1) = 4 floats (16 bytes)
        // kernel (64 vec4) = 64 * 4 = 256 floats (1024 bytes)
        // total = 1040 bytes
        this.uniformsBuffer = device.createBuffer({
            label: "SSAO Uniforms",
            size: 1040, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize kernel once
        const data = new Float32Array(1040 / 4);
        
        for (let i = 0; i < 64; ++i) {
            let sample = [
                Math.random() * 2.0 - 1.0, 
                Math.random() * 2.0 - 1.0, 
                Math.random()
            ];
            
            let dist = Math.sqrt(sample[0]*sample[0] + sample[1]*sample[1] + sample[2]*sample[2]);
            sample[0] /= dist; sample[1] /= dist; sample[2] /= dist;
            
            sample[0] *= Math.random();
            sample[1] *= Math.random();
            sample[2] *= Math.random();
            
            let scale = i / 64.0;
            scale = 0.1 + scale * scale * (1.0 - 0.1); 
            sample[0] *= scale;
            sample[1] *= scale;
            sample[2] *= scale;
            
            data[4 + i * 4 + 0] = sample[0];
            data[4 + i * 4 + 1] = sample[1];
            data[4 + i * 4 + 2] = sample[2];
            data[4 + i * 4 + 3] = 0.0;
        }

        device.queue.writeBuffer(this.uniformsBuffer, 0, data.buffer);
        this.updateUniforms();
    }
    
    private uniformData = new Float32Array(4);

    updateUniforms() {
        this.uniformData[0] = this.radius;
        this.uniformData[1] = this.bias;
        this.uniformData[2] = this.power;
        this.uniformData[3] = this.enabled ? 1.0 : 0.0;
        device.queue.writeBuffer(this.uniformsBuffer, 0, this.uniformData.buffer);
    }
}
