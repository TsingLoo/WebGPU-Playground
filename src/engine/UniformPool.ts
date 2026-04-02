export class UniformPool {
    private cpuBuffer: ArrayBuffer;
    public gpuBuffer: GPUBuffer;
    
    // Simple bump allocator state
    private capacityBytes: number;
    private currentOffsetBytes: number = 0;
    
    // Dirty tracking
    // We could optimize by tracking overlapping regions. For now, track a list of disjoint or overlapping regions.
    private dirtyRegions: { offset: number, size: number }[] = [];

    constructor(device: GPUDevice, capacityBytes: number, label: string = "Global Uniform Pool") {
        this.capacityBytes = capacityBytes;
        this.cpuBuffer = new ArrayBuffer(capacityBytes);
        
        this.gpuBuffer = device.createBuffer({
            label,
            size: capacityBytes,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /**
     * Allocates a contiguous block of memory in the pool.
     * Returns the byte offset, a reference to the typed view, and a helper to mark it dirty.
     * @param sizeBytes Must be a multiple of 4 (for Float32 compatibility), and aligned to 256 for bind groups.
     */
    public allocate(sizeBytes: number): { offset: number, sizeBytes: number, view: Float32Array } {
        // WebGPU requires dynamic uniform buffer offsets to be a multiple of 256
        // So we align all allocations to 256 bytes to be safe for any binding type
        const alignment = 256;
        const remainder = this.currentOffsetBytes % alignment;
        
        if (remainder !== 0) {
            this.currentOffsetBytes += (alignment - remainder);
        }

        if (this.currentOffsetBytes + sizeBytes > this.capacityBytes) {
            throw new Error(`UniformPool Out of Memory. Capacity: ${this.capacityBytes}, Requested: ${sizeBytes} at offset: ${this.currentOffsetBytes}`);
        }

        const offset = this.currentOffsetBytes;
        this.currentOffsetBytes += sizeBytes;

        // View for exactly this block
        const view = new Float32Array(this.cpuBuffer, offset, sizeBytes / 4);

        return {
            offset,
            sizeBytes,
            view
        };
    }

    /**
     * Mark a specific region as dirty so it gets uploaded to GPU on next sync.
     */
    public markDirty(offset: number, sizeBytes: number) {
        this.dirtyRegions.push({ offset, size: sizeBytes });
    }

    /**
     * Re-creates the backing buffer if it needs to grow. 
     * In an actual engine, we might do a copy-over to the new buffer. 
     * Here, simple bump allocation resets on scene load so we just reset.
     */
    public reset() {
        this.currentOffsetBytes = 0;
        this.dirtyRegions = [];
    }

    /**
     * Syncs all dirty regions to the GPU and clears the dirty list.
     */
    public syncToGPU(device: GPUDevice) {
        if (this.dirtyRegions.length === 0) return;

        // Naive merge: if many regions, we could sort and merge overlaps to reduce API calls
        // For now, we perform one writeBuffer per region, WebGPU handles queuing efficiently
        
        // A simple optimization: sort by offset, then merge contiguous or overlapping ranges
        this.dirtyRegions.sort((a, b) => a.offset - b.offset);
        
        let currentMerge = this.dirtyRegions[0];
        const merged: { offset: number, size: number }[] = [];

        for (let i = 1; i < this.dirtyRegions.length; i++) {
            const next = this.dirtyRegions[i];
            
            // If next region is contiguous or overlaps, merge them
            if (next.offset <= currentMerge.offset + currentMerge.size) {
                const endA = currentMerge.offset + currentMerge.size;
                const endB = next.offset + next.size;
                currentMerge.size = Math.max(endA, endB) - currentMerge.offset;
            } else {
                merged.push(currentMerge);
                currentMerge = next;
            }
        }
        merged.push(currentMerge);

        // Upload merged regions
        for (const region of merged) {
            // device.queue.writeBuffer requires a source offset in bytes, but when passing an ArrayBuffer, 
            // the dataOffset argument is the offset into the ArrayBuffer source, which is also in bytes, Wait! 
            // the writeBuffer signature with ArrayBuffer is:
            // writeBuffer(buffer, bufferOffset, data, dataOffset?, size?)
            // Note: data is a BufferSource, which is ArrayBuffer or TypedArray.
            device.queue.writeBuffer(
                this.gpuBuffer, 
                region.offset, 
                this.cpuBuffer, 
                region.offset, 
                region.size
            );
        }

        this.dirtyRegions = [];
    }
}
