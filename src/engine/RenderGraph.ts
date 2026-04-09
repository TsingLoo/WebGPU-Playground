import { device, canvas } from '../renderer';

export type ResourceHandle = number;

export interface TextureDescriptor {
    format: GPUTextureFormat;
    usage?: GPUTextureUsageFlags;
    width?: number; // if undefined, uses canvas width
    height?: number; // if undefined, uses canvas height
    scale?: number; // multiplier for canvas size
}

export interface BufferDescriptor {
    size: number;
    usage?: GPUBufferUsageFlags;
}

export interface SubresourceRange {
    baseMipLevel?: number;
    mipLevelCount?: number;
    baseArrayLayer?: number;
    arrayLayerCount?: number;
}

export function encodeSubresourceRange(range?: SubresourceRange): number {
    if (!range) return 0;
    const baseMip = range.baseMipLevel || 0;
    const mipCount = range.mipLevelCount || 1;
    const baseLayer = range.baseArrayLayer || 0;
    const layerCount = range.arrayLayerCount || 1;
    return (baseMip & 0xF) | ((mipCount & 0xF) << 4) | ((baseLayer & 0xFFF) << 8) | ((layerCount & 0xFFF) << 20);
}

interface PhysicalTexture {
    texture: GPUTexture;
    view: GPUTextureView;
    desc: TextureDescriptor;
    width: number;
    height: number;
    framesIdle: number;
    subViews: Map<number, GPUTextureView>;
}

interface PhysicalBuffer {
    buffer: GPUBuffer;
    desc: BufferDescriptor;
    framesIdle: number;
}

export enum PassType {
    Render,
    Compute,
    Generic
}

export interface ResourceAccess {
    handle: ResourceHandle;
    usage: number; // Cast via bitwise combinations depending on resource
    range?: SubresourceRange;
}

export class RenderPassData {
    name: string;
    type: PassType;
    reads: ResourceAccess[] = [];
    writes: ResourceAccess[] = [];
    isRoot: boolean = false;

    // Execution closure generic binding
    executeFn!: (encoder: any, resolver: PassResolver) => void;

    // Render pass specific attributes
    colorAttachments: Map<ResourceHandle, { loadOp?: GPULoadOp, storeOp?: GPUStoreOp, clearValue?: GPUColorDict | number[] }> = new Map();
    depthStencilAttachment?: { handle: ResourceHandle, loadOp?: GPULoadOp, storeOp?: GPUStoreOp, depthReadOnly?: boolean, clearValue?: number };

    constructor(name: string, type: PassType) {
        this.name = name;
        this.type = type;
    }
}

export interface PassResolver {
    getTextureView(handle: ResourceHandle, range?: SubresourceRange): GPUTextureView;
    getBuffer(handle: ResourceHandle): GPUBuffer;
}

export class PassBuilder {
    constructor(protected passData: RenderPassData) { }

    markRoot(): this {
        this.passData.isRoot = true;
        return this;
    }

    readTexture(handle: ResourceHandle, usage: GPUTextureUsageFlags = 0, range?: SubresourceRange): this {
        this.passData.reads.push({ handle, usage, range });
        return this;
    }

    writeTexture(handle: ResourceHandle, usage: GPUTextureUsageFlags = 0, range?: SubresourceRange): this {
        this.passData.writes.push({ handle, usage, range });
        return this;
    }

    readBuffer(handle: ResourceHandle, usage: GPUBufferUsageFlags = 0): this {
        this.passData.reads.push({ handle, usage });
        return this;
    }

    writeBuffer(handle: ResourceHandle, usage: GPUBufferUsageFlags = 0): this {
        this.passData.writes.push({ handle, usage });
        return this;
    }
}

export class RenderPassBuilder extends PassBuilder {
    addColorAttachment(handle: ResourceHandle, options?: { loadOp?: GPULoadOp, storeOp?: GPUStoreOp, clearValue?: GPUColorDict | number[] }): this {
        this.passData.colorAttachments.set(handle, options || {});
        this.writeTexture(handle, GPUTextureUsage.RENDER_ATTACHMENT);
        return this;
    }

    setDepthStencilAttachment(handle: ResourceHandle, options?: { loadOp?: GPULoadOp, storeOp?: GPUStoreOp, depthReadOnly?: boolean, clearValue?: number }): this {
        this.passData.depthStencilAttachment = { handle, ...options };
        if (options?.depthReadOnly) {
            this.readTexture(handle, GPUTextureUsage.RENDER_ATTACHMENT);
        } else {
            this.writeTexture(handle, GPUTextureUsage.RENDER_ATTACHMENT);
        }
        return this;
    }

    execute(fn: (encoder: GPURenderPassEncoder, resolver: PassResolver) => void): void {
        this.passData.executeFn = fn as any;
    }
}

export class ComputePassBuilder extends PassBuilder {
    execute(fn: (encoder: GPUComputePassEncoder, resolver: PassResolver) => void): void {
        this.passData.executeFn = fn as any;
    }
}

export class GenericPassBuilder extends PassBuilder {
    execute(fn: (encoder: GPUCommandEncoder, resolver: PassResolver) => void): void {
        this.passData.executeFn = fn as any;
    }
}

interface CompiledPlan {
    hash: string;
    activePassIndices: number[]; // the indices of required passes in sorted order
    allocations: ResourceHandle[][];
    deallocations: ResourceHandle[][];
    textureUsages: Map<ResourceHandle, GPUTextureUsageFlags>;
    bufferUsages: Map<ResourceHandle, GPUBufferUsageFlags>;
    // Cached implicit loads and stores for auto-deduction
    renderPassLoadStoreOps: Map<number, {
        colors: Map<ResourceHandle, { loadOp: GPULoadOp, storeOp: GPUStoreOp }>,
        depthStencil?: { loadOp: GPULoadOp, storeOp: GPUStoreOp }
    }>;
    textureAliases: Map<ResourceHandle, ResourceHandle>;
    bufferAliases: Map<ResourceHandle, ResourceHandle>;
    aliasBinds: { texture: { child: ResourceHandle, root: ResourceHandle }[], buffer: { child: ResourceHandle, root: ResourceHandle }[] }[];
    mermaidString: string;
}

export class RenderGraph implements PassResolver {
    private nextHandle = 1;

    // Logical declarations for current frame
    private textureDescs = new Map<ResourceHandle, TextureDescriptor>();
    private bufferDescs = new Map<ResourceHandle, BufferDescriptor>();
    private resourceNames = new Map<ResourceHandle, string>();
    private importedTextureViews = new Map<ResourceHandle, GPUTextureView>();
    private importedBuffers = new Map<ResourceHandle, GPUBuffer>();
    private passes: RenderPassData[] = [];

    // Compiled plan cache
    private cachedPlan: CompiledPlan | null = null;
    private canvasDimensions = { width: 0, height: 0 };

    // Physical runtime state
    private handleToPhysTex = new Map<ResourceHandle, PhysicalTexture>();
    private handleToPhysBuf = new Map<ResourceHandle, PhysicalBuffer>();
    private poolTextures = new Map<string, PhysicalTexture[]>();
    private poolBuffers = new Map<string, PhysicalBuffer[]>();

    public createTexture(name: string, desc: TextureDescriptor): ResourceHandle {
        const handle = this.nextHandle++;
        if (desc.usage === undefined) desc.usage = 0;
        this.textureDescs.set(handle, desc);
        this.resourceNames.set(handle, name);
        return handle;
    }

    public importTexture(name: string, view: GPUTextureView): ResourceHandle {
        const handle = this.nextHandle++;
        this.importedTextureViews.set(handle, view);
        this.resourceNames.set(handle, name);
        return handle;
    }

    public createBuffer(name: string, desc: BufferDescriptor): ResourceHandle {
        const handle = this.nextHandle++;
        if (desc.usage === undefined) desc.usage = 0;
        this.bufferDescs.set(handle, desc);
        this.resourceNames.set(handle, name);
        return handle;
    }

    public importBuffer(name: string, buffer: GPUBuffer): ResourceHandle {
        const handle = this.nextHandle++;
        this.importedBuffers.set(handle, buffer);
        this.resourceNames.set(handle, name);
        return handle;
    }

    public addRenderPass(name: string): RenderPassBuilder {
        const pass = new RenderPassData(name, PassType.Render);
        this.passes.push(pass);
        return new RenderPassBuilder(pass);
    }

    public addComputePass(name: string): ComputePassBuilder {
        const pass = new RenderPassData(name, PassType.Compute);
        this.passes.push(pass);
        return new ComputePassBuilder(pass);
    }

    public addGenericPass(name: string): GenericPassBuilder {
        const pass = new RenderPassData(name, PassType.Generic);
        this.passes.push(pass);
        return new GenericPassBuilder(pass);
    }

    // PassResolver implementation
    public getTextureView(handle: ResourceHandle, range?: SubresourceRange): GPUTextureView {
        if (this.importedTextureViews.has(handle)) {
            return this.importedTextureViews.get(handle)!;
        }

        const phys = this.handleToPhysTex.get(handle);
        if (!phys) {
            throw new Error(`Texture handle ${handle} (${this.resourceNames.get(handle)}) has no physical texture bound.`);
        }

        if (!range) return phys.view;

        const key = encodeSubresourceRange(range);
        if (!phys.subViews.has(key)) {
            phys.subViews.set(key, phys.texture.createView(range as GPUTextureViewDescriptor));
        }
        return phys.subViews.get(key)!;
    }

    public getBuffer(handle: ResourceHandle): GPUBuffer {
        if (this.importedBuffers.has(handle)) {
            return this.importedBuffers.get(handle)!;
        }

        const phys = this.handleToPhysBuf.get(handle);
        if (!phys) {
            throw new Error(`Buffer handle ${handle} (${this.resourceNames.get(handle)}) has no physical buffer bound.`);
        }
        return phys.buffer;
    }

    private resolveDimensions(desc: TextureDescriptor): { w: number, h: number } {
        let w = desc.width !== undefined ? desc.width : canvas.width;
        let h = desc.height !== undefined ? desc.height : canvas.height;
        if (desc.scale !== undefined) {
            w = Math.max(1, Math.floor(w * desc.scale));
            h = Math.max(1, Math.floor(h * desc.scale));
        }
        return { w, h };
    }

    private getPhysicalTexture(desc: TextureDescriptor, name: string): PhysicalTexture {
        const { w, h } = this.resolveDimensions(desc);
        const key = `${desc.format}_${w}_${h}_${desc.usage}`;

        const bucket = this.poolTextures.get(key);
        if (bucket && bucket.length > 0) {
            const pt = bucket.pop()!;
            pt.framesIdle = 0;
            return pt;
        }

        const texture = device.createTexture({
            label: `RG_PhysTex_${name}`,
            size: [w, h],
            format: desc.format,
            usage: desc.usage!
        });

        return {
            texture,
            view: texture.createView(),
            desc: { ...desc },
            width: w,
            height: h,
            framesIdle: 0,
            subViews: new Map()
        };
    }

    private getPhysicalBuffer(desc: BufferDescriptor, name: string): PhysicalBuffer {
        const key = `${desc.size}_${desc.usage}`;
        const bucket = this.poolBuffers.get(key);
        if (bucket && bucket.length > 0) {
            const pb = bucket.pop()!;
            pb.framesIdle = 0;
            return pb;
        }

        const buffer = device.createBuffer({
            label: `RG_PhysBuf_${name}`,
            size: desc.size,
            usage: desc.usage!
        });

        return { buffer, desc: { ...desc }, framesIdle: 0 };
    }

    private releasePhysicalTexture(handle: ResourceHandle, phys: PhysicalTexture) {
        phys.framesIdle = 0;
        const key = `${phys.desc.format}_${phys.width}_${phys.height}_${phys.desc.usage}`;
        if (!this.poolTextures.has(key)) this.poolTextures.set(key, []);
        this.poolTextures.get(key)!.push(phys);
        this.handleToPhysTex.delete(handle);
    }

    private releasePhysicalBuffer(handle: ResourceHandle, phys: PhysicalBuffer) {
        phys.framesIdle = 0;
        const key = `${phys.desc.size}_${phys.desc.usage}`;
        if (!this.poolBuffers.has(key)) this.poolBuffers.set(key, []);
        this.poolBuffers.get(key)!.push(phys);
        this.handleToPhysBuf.delete(handle);
    }

    private buildTopologyHash(): string {
        let hash = `${canvas.width}_${canvas.height};`;
        for (let i = 0; i < this.passes.length; i++) {
            const p = this.passes[i];
            hash += p.name;
            for (let j = 0; j < p.reads.length; j++) hash += `R${p.reads[j].handle}`;
            for (let j = 0; j < p.writes.length; j++) hash += `W${p.writes[j].handle}`;
            for (const c of p.colorAttachments.keys()) hash += `C${c}`;
            hash += `|`;
        }
        return hash;
    }

    private compile(): CompiledPlan {
        const hash = this.buildTopologyHash();
        if (this.cachedPlan && this.cachedPlan.hash === hash) {
            return this.cachedPlan;
        }

        // console.log("[RenderGraph] Topological changes detected, recompiling dependencies.");

        // 1. Dependency Analysis
        const deps = new Map<RenderPassData, Set<RenderPassData>>();
        const lastWriter = new Map<ResourceHandle, RenderPassData>();
        const lastReaders = new Map<ResourceHandle, RenderPassData[]>();

        for (const pass of this.passes) {
            deps.set(pass, new Set());

            for (const r of pass.reads) {
                if (lastWriter.has(r.handle) && lastWriter.get(r.handle) !== pass) {
                    deps.get(pass)!.add(lastWriter.get(r.handle)!);
                }
                if (!lastReaders.has(r.handle)) lastReaders.set(r.handle, []);
                // Only add itself as a reader once
                if (!lastReaders.get(r.handle)!.includes(pass)) {
                    lastReaders.get(r.handle)!.push(pass);
                }
            }

            for (const w of pass.writes) {
                if (lastWriter.has(w.handle) && lastWriter.get(w.handle) !== pass) {
                    deps.get(pass)!.add(lastWriter.get(w.handle)!);
                }
                if (lastReaders.has(w.handle)) {
                    for (const reader of lastReaders.get(w.handle)!) {
                        if (reader !== pass) {
                            deps.get(pass)!.add(reader);
                        }
                    }
                }
                lastWriter.set(w.handle, pass);
                lastReaders.set(w.handle, []);
            }
        }

        // 2. Identify required passes natively reachable backwards
        const visitedRequired = new Set<RenderPassData>();
        const rootsToVisit = new Set<RenderPassData>();

        for (const pass of this.passes) {
            if (pass.isRoot) {
                rootsToVisit.add(pass);
            } else {
                for (const w of pass.writes) {
                    if (this.importedTextureViews.has(w.handle) || this.importedBuffers.has(w.handle)) {
                        rootsToVisit.add(pass);
                        break;
                    }
                }
            }
        }

        const workQueue = Array.from(rootsToVisit);
        while (workQueue.length > 0) {
            const pass = workQueue.pop()!;
            if (!visitedRequired.has(pass)) {
                visitedRequired.add(pass);
                for (const dep of deps.get(pass)!) {
                    workQueue.push(dep);
                }
            }
        }

        // 3. Topological Sort using Kahn
        const indegree = new Map<RenderPassData, number>();
        const adj = new Map<RenderPassData, RenderPassData[]>();

        for (const pass of visitedRequired) {
            indegree.set(pass, 0);
            adj.set(pass, []);
        }

        for (const pass of visitedRequired) {
            for (const dep of deps.get(pass)!) {
                if (visitedRequired.has(dep)) {
                    adj.get(dep)!.push(pass); // dep -> pass constraint
                    indegree.set(pass, indegree.get(pass)! + 1);
                }
            }
        }

        const activePasses: RenderPassData[] = [];
        const activePassIndices: number[] = [];
        const queue: RenderPassData[] = [];

        const queueInsert = (pass: RenderPassData) => {
            const priority = this.passes.indexOf(pass);
            let low = 0, high = queue.length;
            while (low < high) {
                const mid = (low + high) >>> 1;
                if (this.passes.indexOf(queue[mid]) < priority) low = mid + 1;
                else high = mid;
            }
            queue.splice(low, 0, pass);
        };

        // Sorting logic enforces structural consistency
        for (const [pass, deg] of indegree.entries()) {
            if (deg === 0) queueInsert(pass);
        }

        while (queue.length > 0) {
            const current = queue.shift()!;
            activePasses.push(current);
            activePassIndices.push(this.passes.indexOf(current));

            for (const neighbor of adj.get(current)!) {
                indegree.set(neighbor, indegree.get(neighbor)! - 1);
                if (indegree.get(neighbor) === 0) {
                    queueInsert(neighbor);
                }
            }
        }

        if (activePasses.length !== visitedRequired.size) {
            throw new Error("RenderGraph Compile Error: Circular pass dependency detected.");
        }

        // 4. Lifetimes & Usages calculation
        const textureUsages = new Map<ResourceHandle, GPUTextureUsageFlags>();
        const bufferUsages = new Map<ResourceHandle, GPUBufferUsageFlags>();

        for (const [handle, desc] of this.textureDescs.entries()) {
            textureUsages.set(handle, desc.usage || 0);
        }
        for (const [handle, desc] of this.bufferDescs.entries()) {
            bufferUsages.set(handle, desc.usage || 0);
        }

        const allocations: ResourceHandle[][] = activePasses.map(() => []);
        const deallocations: ResourceHandle[][] = activePasses.map(() => []);

        const firstUsageIdx = new Map<ResourceHandle, number>();
        const lastUsageIdx = new Map<ResourceHandle, number>();

        for (let i = 0; i < activePasses.length; i++) {
            const pass = activePasses[i];
            const allHandles = new Set([
                ...pass.reads.map(r => r.handle),
                ...pass.writes.map(w => w.handle)
            ]);

            for (const handle of allHandles) {
                if (!firstUsageIdx.has(handle)) firstUsageIdx.set(handle, i);
                lastUsageIdx.set(handle, i);
            }

            for (const read of pass.reads) {
                if (this.textureDescs.has(read.handle)) {
                    textureUsages.set(read.handle, (textureUsages.get(read.handle) || 0) | read.usage);
                } else if (this.bufferDescs.has(read.handle)) {
                    bufferUsages.set(read.handle, (bufferUsages.get(read.handle) || 0) | read.usage);
                }
            }
            for (const write of pass.writes) {
                if (this.textureDescs.has(write.handle)) {
                    textureUsages.set(write.handle, (textureUsages.get(write.handle) || 0) | write.usage);
                } else if (this.bufferDescs.has(write.handle)) {
                    bufferUsages.set(write.handle, (bufferUsages.get(write.handle) || 0) | write.usage);
                }
            }
        }

        const textureAliases = new Map<ResourceHandle, ResourceHandle>();

        // 4.1 Memory Aliasing for Textures
        const textureGroups = new Map<string, ResourceHandle[]>();
        for (const handle of this.textureDescs.keys()) {
            if (!firstUsageIdx.has(handle) || this.importedTextureViews.has(handle)) continue;
            const desc = this.textureDescs.get(handle)!;
            const usage = textureUsages.get(handle) || 0;
            const { w, h } = this.resolveDimensions(desc);
            const formatKey = `${desc.format}_${w}_${h}_${usage}`;
            if (!textureGroups.has(formatKey)) textureGroups.set(formatKey, []);
            textureGroups.get(formatKey)!.push(handle);
        }

        for (const group of textureGroups.values()) {
            group.sort((a, b) => firstUsageIdx.get(a)! - firstUsageIdx.get(b)!);
            const activeIntervals: { rootHandle: ResourceHandle, endIdx: number }[] = [];

            for (const handle of group) {
                const start = firstUsageIdx.get(handle)!;
                const end = lastUsageIdx.get(handle)!;

                let reusedRoot: ResourceHandle | null = null;
                for (let i = 0; i < activeIntervals.length; i++) {
                    if (activeIntervals[i].endIdx < start) {
                        reusedRoot = activeIntervals[i].rootHandle;
                        activeIntervals[i].endIdx = end; // Extend physical lifetime
                        break;
                    }
                }

                if (reusedRoot !== null) {
                    textureAliases.set(handle, reusedRoot);
                    // Extend the root's reported last usage index to cover this child's lifetime implicitly
                    lastUsageIdx.set(reusedRoot, Math.max(lastUsageIdx.get(reusedRoot)!, end));
                } else {
                    textureAliases.set(handle, handle);
                    activeIntervals.push({ rootHandle: handle, endIdx: end });
                }
            }
        }

        for (const handle of this.textureDescs.keys()) {
            if (firstUsageIdx.has(handle) && !this.importedTextureViews.has(handle)) {
                if (textureAliases.get(handle) === handle) {
                    allocations[firstUsageIdx.get(handle)!].push(handle);
                    deallocations[lastUsageIdx.get(handle)!].push(handle);
                }
            }
        }

        const bufferAliases = new Map<ResourceHandle, ResourceHandle>();

        // 4.2 Memory Aliasing for Buffers
        const bufferGroups = new Map<string, ResourceHandle[]>();
        for (const handle of this.bufferDescs.keys()) {
            if (!firstUsageIdx.has(handle) || this.importedBuffers.has(handle)) continue;
            const desc = this.bufferDescs.get(handle)!;
            const usage = bufferUsages.get(handle) || 0;
            const sizeKey = `${desc.size}_${usage}`;
            if (!bufferGroups.has(sizeKey)) bufferGroups.set(sizeKey, []);
            bufferGroups.get(sizeKey)!.push(handle);
        }

        for (const group of bufferGroups.values()) {
            group.sort((a, b) => firstUsageIdx.get(a)! - firstUsageIdx.get(b)!);
            const activeIntervals: { rootHandle: ResourceHandle, endIdx: number }[] = [];

            for (const handle of group) {
                const start = firstUsageIdx.get(handle)!;
                const end = lastUsageIdx.get(handle)!;

                let reusedRoot: ResourceHandle | null = null;
                for (let i = 0; i < activeIntervals.length; i++) {
                    if (activeIntervals[i].endIdx < start) {
                        reusedRoot = activeIntervals[i].rootHandle;
                        activeIntervals[i].endIdx = end; // Extend physical lifetime
                        break;
                    }
                }

                if (reusedRoot !== null) {
                    bufferAliases.set(handle, reusedRoot);
                    lastUsageIdx.set(reusedRoot, Math.max(lastUsageIdx.get(reusedRoot)!, end));
                } else {
                    bufferAliases.set(handle, handle);
                    activeIntervals.push({ rootHandle: handle, endIdx: end });
                }
            }
        }

        for (const handle of this.bufferDescs.keys()) {
            if (firstUsageIdx.has(handle) && !this.importedBuffers.has(handle)) {
                if (bufferAliases.get(handle) === handle) {
                    allocations[firstUsageIdx.get(handle)!].push(handle);
                    deallocations[lastUsageIdx.get(handle)!].push(handle);
                }
            }
        }

        // 4.3 Precompile pass-local alias bindings execution mapping map
        const aliasBinds = activePasses.map(() => ({ texture: [] as any[], buffer: [] as any[] }));
        for (const handle of this.textureDescs.keys()) {
            if (textureAliases.get(handle) !== handle && textureAliases.has(handle)) {
                if (firstUsageIdx.has(handle)) {
                    aliasBinds[firstUsageIdx.get(handle)!].texture.push({ child: handle, root: textureAliases.get(handle)! });
                }
            }
        }
        for (const handle of this.bufferDescs.keys()) {
            if (bufferAliases.get(handle) !== handle && bufferAliases.has(handle)) {
                if (firstUsageIdx.has(handle)) {
                    aliasBinds[firstUsageIdx.get(handle)!].buffer.push({ child: handle, root: bufferAliases.get(handle)! });
                }
            }
        }

        // 5. Render Pass Load/Store operations inference
        const renderPassLoadStoreOps = new Map();
        for (let i = 0; i < activePasses.length; i++) {
            const pass = activePasses[i];
            if (pass.type === PassType.Render) {
                const colors = new Map();
                for (const [handle, attach] of pass.colorAttachments) {
                    let loadOp = attach.loadOp;
                    let storeOp = attach.storeOp;
                    if (!loadOp) {
                        loadOp = firstUsageIdx.get(handle) === i ? "clear" : "load";
                    }
                    if (!storeOp) {
                        const isImported = this.importedTextureViews.has(handle);
                        const isLastUsage = lastUsageIdx.get(handle) === i;
                        storeOp = (isLastUsage && !isImported) ? "discard" : "store";
                    }
                    colors.set(handle, { loadOp, storeOp });
                }

                let depthStencil;
                if (pass.depthStencilAttachment) {
                    const handle = pass.depthStencilAttachment.handle;
                    let loadOp = pass.depthStencilAttachment.loadOp;
                    let storeOp = pass.depthStencilAttachment.storeOp;
                    if (!loadOp && !pass.depthStencilAttachment.depthReadOnly) {
                        loadOp = firstUsageIdx.get(handle) === i ? "clear" : "load";
                    }
                    if (!storeOp && !pass.depthStencilAttachment.depthReadOnly) {
                        const isImported = this.importedTextureViews.has(handle);
                        const isLastUsage = lastUsageIdx.get(handle) === i;
                        storeOp = (isLastUsage && !isImported) ? "discard" : "store";
                    }
                    depthStencil = { loadOp, storeOp };
                }

                renderPassLoadStoreOps.set(i, { colors, depthStencil });
            }
        }

        // 6. Precompile Mermaid Diagram
        const mermaidLines: string[] = ["graph TD"];
        for (let i = 0; i < activePassIndices.length; i++) {
            const pass = this.passes[activePassIndices[i]];
            const safeNameA = pass.name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
            mermaidLines.push(`  ${safeNameA}["${pass.name}"]`);

            for (const write of pass.writes) {
                const resName = this.resourceNames.get(write.handle) || `H${write.handle}`;
                const tAlias = textureAliases.get(write.handle);
                const bAlias = bufferAliases.get(write.handle);
                const isTAlias = tAlias && tAlias !== write.handle;
                const isBAlias = bAlias && bAlias !== write.handle;
                
                let nodeStr = `R${write.handle}(("Res: ${resName}"))`;
                if (isTAlias) nodeStr = `R${tAlias}(("Res: ${resName} (Aliased)"))`;
                if (isBAlias) nodeStr = `B${bAlias}(("Res: ${resName} (Aliased Buffer)"))`;
                
                mermaidLines.push(`  ${safeNameA} -- Write --> ${nodeStr}`);
            }
            for (const read of pass.reads) {
                const resName = this.resourceNames.get(read.handle) || `H${read.handle}`;
                const tAlias = textureAliases.get(read.handle);
                const bAlias = bufferAliases.get(read.handle);
                const isTAlias = tAlias && tAlias !== read.handle;
                const isBAlias = bAlias && bAlias !== read.handle;

                let nodeStr = `R${read.handle}(("Res: ${resName}"))`;
                if (isTAlias) nodeStr = `R${tAlias}(("Res: ${resName} (Aliased)"))`;
                if (isBAlias) nodeStr = `B${bAlias}(("Res: ${resName} (Aliased Buffer)"))`;
                
                mermaidLines.push(`  ${nodeStr} -- Read --> ${safeNameA}`);
            }
        }

        this.cachedPlan = {
            hash,
            activePassIndices,
            allocations,
            deallocations,
            textureUsages,
            bufferUsages,
            renderPassLoadStoreOps,
            textureAliases,
            bufferAliases,
            aliasBinds,
            mermaidString: mermaidLines.join("\n")
        };

        const passOrderNames = activePassIndices.map(idx => this.passes[idx].name).join(" -> ");
        console.log(`[RenderGraph] Stream Updated: ${passOrderNames}`);

        return this.cachedPlan;
    }

    public execute(encoder: GPUCommandEncoder) {
        const plan = this.compile();

        // Push required cumulative deduced usages into logical descriptors
        for (const [handle, usage] of plan.textureUsages) {
            if (this.textureDescs.has(handle)) this.textureDescs.get(handle)!.usage = usage;
        }
        for (const [handle, usage] of plan.bufferUsages) {
            if (this.bufferDescs.has(handle)) this.bufferDescs.get(handle)!.usage = usage;
        }

        for (let i = 0; i < plan.activePassIndices.length; i++) {
            const passIndex = plan.activePassIndices[i];
            const pass = this.passes[passIndex];

            // 1. Allocate resources starting their lifetime here
            for (const handle of plan.allocations[i]) {
                if (this.textureDescs.has(handle)) {
                    const pt = this.getPhysicalTexture(this.textureDescs.get(handle)!, this.resourceNames.get(handle)!);
                    this.handleToPhysTex.set(handle, pt);
                } else if (this.bufferDescs.has(handle)) {
                    const pb = this.getPhysicalBuffer(this.bufferDescs.get(handle)!, this.resourceNames.get(handle)!);
                    this.handleToPhysBuf.set(handle, pb);
                }
            }

            // Map aliases to their active roots unconditionally
            for (const map of plan.aliasBinds[i].texture) {
                if (this.handleToPhysTex.has(map.root)) {
                    this.handleToPhysTex.set(map.child, this.handleToPhysTex.get(map.root)!);
                }
            }
            for (const map of plan.aliasBinds[i].buffer) {
                if (this.handleToPhysBuf.has(map.root)) {
                    this.handleToPhysBuf.set(map.child, this.handleToPhysBuf.get(map.root)!);
                }
            }

            // 2. Execution wrappers
            if (pass.type === PassType.Render) {
                const ops = plan.renderPassLoadStoreOps.get(i)!;
                const colorAttachments: GPURenderPassColorAttachment[] = [];
                for (const [handle, attachMap] of pass.colorAttachments) {
                    const op = ops.colors.get(handle)!;
                    colorAttachments.push({
                        view: this.getTextureView(handle),
                        loadOp: op.loadOp,
                        storeOp: op.storeOp,
                        clearValue: attachMap.clearValue !== undefined ? attachMap.clearValue : [0, 0, 0, 0]
                    });
                }

                let depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined;
                if (pass.depthStencilAttachment) {
                    const ds = pass.depthStencilAttachment;
                    depthStencilAttachment = {
                        view: this.getTextureView(ds.handle),
                        depthReadOnly: ds.depthReadOnly
                    };
                    if (!ds.depthReadOnly) {
                        const op = ops.depthStencil!;
                        depthStencilAttachment.depthLoadOp = op.loadOp;
                        depthStencilAttachment.depthStoreOp = op.storeOp;
                        depthStencilAttachment.depthClearValue = ds.clearValue !== undefined ? ds.clearValue : 1.0;
                    }
                }

                const renderPassEnc = encoder.beginRenderPass({
                    label: pass.name,
                    colorAttachments,
                    depthStencilAttachment
                });

                pass.executeFn(renderPassEnc, this);
                renderPassEnc.end();

            } else if (pass.type === PassType.Compute) {
                const computePassEnc = encoder.beginComputePass({ label: pass.name });
                pass.executeFn(computePassEnc, this);
                computePassEnc.end();
            } else {
                pass.executeFn(encoder, this);
            }

            // 3. Deallocate resources mapping them back to physical pool
            for (const handle of plan.deallocations[i]) {
                if (this.handleToPhysTex.has(handle)) {
                    this.releasePhysicalTexture(handle, this.handleToPhysTex.get(handle)!);
                } else if (this.handleToPhysBuf.has(handle)) {
                    this.releasePhysicalBuffer(handle, this.handleToPhysBuf.get(handle)!);
                }
            }
        }

        // LRU cleanup: Destroy objects idling > 5 frames
        for (const bucket of this.poolTextures.values()) {
            for (let i = bucket.length - 1; i >= 0; i--) {
                bucket[i].framesIdle++;
                if (bucket[i].framesIdle > 5) {
                    bucket[i].texture.destroy();
                    // O(1) swap pop since arbitrary order idle
                    const last = bucket.pop()!;
                    if (i < bucket.length) bucket[i] = last;
                }
            }
        }
        for (const bucket of this.poolBuffers.values()) {
            for (let i = bucket.length - 1; i >= 0; i--) {
                bucket[i].framesIdle++;
                if (bucket[i].framesIdle > 5) {
                    bucket[i].buffer.destroy();
                    const last = bucket.pop()!;
                    if (i < bucket.length) bucket[i] = last;
                }
            }
        }

        // Reset topological tracking state for strictly deterministic caching next frame
        this.passes = [];
        this.importedTextureViews.clear();
        this.importedBuffers.clear();
        this.textureDescs.clear();
        this.bufferDescs.clear();
        this.resourceNames.clear();
        this.nextHandle = 1; // Strict ID deterministic requirement fulfilled 
    }

    public clearPhysicalPool() {
        for (const bucket of this.poolTextures.values()) {
            for (const pt of bucket) pt.texture.destroy();
        }
        for (const bucket of this.poolBuffers.values()) {
            for (const pb of bucket) pb.buffer.destroy();
        }
        this.poolTextures.clear();
        this.poolBuffers.clear();
    }

    public getMermaidGraph(): string {
        return this.cachedPlan ? this.cachedPlan.mermaidString : "graph TD\n  Empty[No Plan Compiled]";
    }
}
