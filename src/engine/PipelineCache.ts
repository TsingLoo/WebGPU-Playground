export class PipelineCache {
    private static instance: PipelineCache;
    private renderPipelines: Map<string, GPURenderPipeline> = new Map();
    private computePipelines: Map<string, GPUComputePipeline> = new Map();

    private constructor() {}

    public static getInstance(): PipelineCache {
        if (!PipelineCache.instance) {
            PipelineCache.instance = new PipelineCache();
        }
        return PipelineCache.instance;
    }

    /**
     * Gets a cached RenderPipeline or creates a new one if not found.
     */
    public getRenderPipeline(device: GPUDevice, descriptor: GPURenderPipelineDescriptor, cacheKey: string): GPURenderPipeline {
        if (this.renderPipelines.has(cacheKey)) {
            return this.renderPipelines.get(cacheKey)!;
        }

        const pipeline = device.createRenderPipeline(descriptor);
        this.renderPipelines.set(cacheKey, pipeline);
        return pipeline;
    }

    /**
     * Gets a cached ComputePipeline or creates a new one if not found.
     */
    public getComputePipeline(device: GPUDevice, descriptor: GPUComputePipelineDescriptor, cacheKey: string): GPUComputePipeline {
        if (this.computePipelines.has(cacheKey)) {
            return this.computePipelines.get(cacheKey)!;
        }

        const pipeline = device.createComputePipeline(descriptor);
        this.computePipelines.set(cacheKey, pipeline);
        return pipeline;
    }

    public clear() {
        this.renderPipelines.clear();
        this.computePipelines.clear();
    }
}

export const pipelineCache = PipelineCache.getInstance();
