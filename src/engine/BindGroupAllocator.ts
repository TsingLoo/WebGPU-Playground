export class BindGroupAllocator {
    private static instance: BindGroupAllocator;
    private bindGroups: Map<string, GPUBindGroup> = new Map();

    private constructor() {}

    public static getInstance(): BindGroupAllocator {
        if (!BindGroupAllocator.instance) {
            BindGroupAllocator.instance = new BindGroupAllocator();
        }
        return BindGroupAllocator.instance;
    }

    /**
     * Gets a cached BindGroup or creates a new one.
     * @param cacheKey A unique string identifying this bind group. 
     *                 For materials, this can be something like `material_${materialId}`
     *                 or a hash of the layout and resource IDs if known.
     */
    public getBindGroup(device: GPUDevice, descriptor: GPUBindGroupDescriptor, cacheKey: string): GPUBindGroup {
        if (this.bindGroups.has(cacheKey)) {
            return this.bindGroups.get(cacheKey)!;
        }

        const bindGroup = device.createBindGroup(descriptor);
        this.bindGroups.set(cacheKey, bindGroup);
        return bindGroup;
    }

    public clear() {
        this.bindGroups.clear();
    }
}

export const bindGroupAllocator = BindGroupAllocator.getInstance();
