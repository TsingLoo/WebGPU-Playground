import { Component } from '../Component';
import { device, globalUniformPool, modelBindGroupLayout } from '../../renderer';

// Assuming Mesh gets exported from stage/scene or extracted later
import { Mesh } from '../GLTFLoader';

export class MeshRenderer extends Component {
    public mesh: Mesh | null = null;
    
    // WebGPU buffers for the model transform
    public modelBindGroup: GPUBindGroup | null = null;
    
    private poolAlloc: { offset: number, sizeBytes: number, view: Float32Array } | null = null;

    override onAwake(): void {
        this.initGPUResources();
    }

    public initGPUResources() {
        if (!this.mesh) return;
        
        // Recreate or reuse? 
        if (!this.poolAlloc) {
            // Allocate 16 floats (64 bytes) for the model transform matrix
            this.poolAlloc = globalUniformPool.allocate(64);

            this.modelBindGroup = device.createBindGroup({
                label: "model bind group",
                layout: modelBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: globalUniformPool.gpuBuffer,
                            offset: this.poolAlloc.offset,
                            size: this.poolAlloc.sizeBytes 
                        }
                    }
                ]
            });
        }
    }

    override onUpdate(_dt: number): void {
        // Sync transform to GPU pool array if component has an entity
        if (this.poolAlloc && this.entity && this.entity.worldTransform) {
            // Write directly to the CPU view and mark dirty
            this.poolAlloc.view.set(this.entity.worldTransform);
            globalUniformPool.markDirty(this.poolAlloc.offset, this.poolAlloc.sizeBytes);
        }
    }

    public setMesh(mesh: Mesh) {
        this.mesh = mesh;
        this.initGPUResources();
    }
}

