import { Component } from '../Component';
import { device, modelBindGroupLayout } from '../../renderer';

// Assuming Mesh gets exported from stage/scene or extracted later
import { Mesh } from '../GLTFLoader';

export class MeshRenderer extends Component {
    public mesh: Mesh | null = null;
    
    // WebGPU buffers for the model transform
    public modelMatUniformBuffer: GPUBuffer | null = null;
    public modelBindGroup: GPUBindGroup | null = null;

    override onAwake(): void {
        this.initGPUResources();
    }

    public initGPUResources() {
        if (!this.mesh) return;
        
        // Recreate or reuse? 
        if (!this.modelMatUniformBuffer) {
            this.modelMatUniformBuffer = device.createBuffer({
                label: "model mat uniform",
                size: 16 * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            this.modelBindGroup = device.createBindGroup({
                label: "model bind group",
                layout: modelBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.modelMatUniformBuffer }
                    }
                ]
            });
        }
    }

    override onUpdate(_dt: number): void {
        // Sync transform to GPU if it changed
        // We could optimize this by storing last sync frame or reading isTransformDirty
        if (this.modelMatUniformBuffer && this.entity && this.entity.worldTransform) {
            device.queue.writeBuffer(this.modelMatUniformBuffer, 0, this.entity.worldTransform as any);
        }
    }

    public setMesh(mesh: Mesh) {
        this.mesh = mesh;
        this.initGPUResources();
    }
}
