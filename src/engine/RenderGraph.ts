import { device, canvas } from '../renderer';

export type ResourceHandle = number;

export interface TextureDescriptor {
    format: GPUTextureFormat;
    usage?: GPUTextureUsageFlags; // Now optional! Graph will deduce it.
    width?: number; // if undefined, uses canvas width
    height?: number; // if undefined, uses canvas height
    scale?: number; // multiplier for canvas size
}

export interface SubresourceRange {
    baseMipLevel?: number;
    mipLevelCount?: number;
    baseArrayLayer?: number;
    arrayLayerCount?: number;
}

interface PhysicalTexture {
    texture: GPUTexture;
    view: GPUTextureView;
    desc: TextureDescriptor;
    width: number;
    height: number;
    framesIdle: number;
    subViews: Map<string, GPUTextureView>;
}

export type PassExecuteFn = (encoder: GPUCommandEncoder, pass: PassResolver) => void;

interface ResourceAccess {
    handle: ResourceHandle;
    usage: GPUTextureUsageFlags;
    range?: SubresourceRange;
}

class RenderPassData {
    name: string;
    reads: ResourceAccess[] = [];
    writes: ResourceAccess[] = [];
    executeFn!: PassExecuteFn;
    isRoot: boolean = false;
    
    constructor(name: string) {
        this.name = name;
    }
}

export interface PassResolver {
    getTextureView(handle: ResourceHandle, range?: SubresourceRange): GPUTextureView;
}

export class PassBuilder {
    constructor(private passParams: RenderPassData) {}

    markRoot(): PassBuilder {
        this.passParams.isRoot = true;
        return this;
    }

    readTexture(handle: ResourceHandle, usage: GPUTextureUsageFlags = 0, range?: SubresourceRange): PassBuilder {
        this.passParams.reads.push({ handle, usage, range });
        return this;
    }

    writeTexture(handle: ResourceHandle, usage: GPUTextureUsageFlags = 0, range?: SubresourceRange): PassBuilder {
        this.passParams.writes.push({ handle, usage, range });
        return this;
    }

    execute(fn: PassExecuteFn): void {
        this.passParams.executeFn = fn;
    }
}

export class RenderGraph implements PassResolver {
    private nextHandle = 1;
    
    // logical resources
    private resourceDescs = new Map<ResourceHandle, TextureDescriptor>();
    private resourceNames = new Map<ResourceHandle, string>();
    private importedViews = new Map<ResourceHandle, GPUTextureView>();
    
    // pass data
    private passes: RenderPassData[] = [];
    
    // physical textures managed by the graph
    private physicalPool: PhysicalTexture[] = [];
    private handleToPhysical = new Map<ResourceHandle, PhysicalTexture>();

    public createTexture(name: string, desc: TextureDescriptor): ResourceHandle {
        const handle = this.nextHandle++;
        // Start usage at 0 if not provided, accumulate later
        if (desc.usage === undefined) desc.usage = 0;
        this.resourceDescs.set(handle, desc);
        this.resourceNames.set(handle, name);
        return handle;
    }

    public importTexture(name: string, view: GPUTextureView): ResourceHandle {
        const handle = this.nextHandle++;
        this.importedViews.set(handle, view);
        this.resourceNames.set(handle, name);
        return handle;
    }

    public addPass(name: string): PassBuilder {
        const pass = new RenderPassData(name);
        this.passes.push(pass);
        return new PassBuilder(pass);
    }

    // PassResolver implementation
    public getTextureView(handle: ResourceHandle, range?: SubresourceRange): GPUTextureView {
        if (this.importedViews.has(handle)) {
            // Cannot easily fetch sub-ranges of externally provided views without their texture reference.
            // Assuming imported views are full views for now.
            return this.importedViews.get(handle)!;
        }
        
        const phys = this.handleToPhysical.get(handle);
        if (!phys) {
            throw new Error(`Texture handle ${handle} (${this.resourceNames.get(handle)}) has no physical texture bound.`);
        }
        
        if (!range) return phys.view;

        // Subresource key
        const key = `${range.baseMipLevel || 0}_${range.mipLevelCount || 1}_${range.baseArrayLayer || 0}_${range.arrayLayerCount || 1}`;
        if (!phys.subViews.has(key)) {
            phys.subViews.set(key, phys.texture.createView(range as GPUTextureViewDescriptor));
        }
        return phys.subViews.get(key)!;
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
        
        // Search pool for matching texture
        let bestCandidateIdx = -1;
        for (let i = 0; i < this.physicalPool.length; i++) {
            const pt = this.physicalPool[i];
            // Must have exactly same format and dimensions, and its usage capabilities must be a superset of requested
            if (pt.desc.format === desc.format && 
                pt.width === w && 
                pt.height === h &&
                (pt.desc.usage! & desc.usage!) === desc.usage!) {
                bestCandidateIdx = i;
                break;
            }
        }
        
        if (bestCandidateIdx >= 0) {
            const pt = this.physicalPool[bestCandidateIdx];
            this.physicalPool.splice(bestCandidateIdx, 1);
            pt.framesIdle = 0; // reset LRU
            // Overwrite desc to reflect new logical context, but usage retains the physical capabilities
            // Actually, keep the physical usage, as it's a superset.
            return pt;
        }
        
        // Create new if not found
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

    public compile() {
        // 1. Pass Culling (DAG Evaluation)
        const activePasses: RenderPassData[] = [];
        const requiredHandles = new Set<ResourceHandle>();
        
        // Imported handles are implicitly sinks (they are usually the swapchain canvas)
        for (const [handle] of this.importedViews) {
            requiredHandles.add(handle);
        }

        // Traverse backwards to find valid dependency chains
        for (let i = this.passes.length - 1; i >= 0; i--) {
            const pass = this.passes[i];
            
            let isNeeded = pass.isRoot;
            if (!isNeeded) {
                // If it writes to a required handle, it's needed
                for (const write of pass.writes) {
                    if (requiredHandles.has(write.handle)) {
                        isNeeded = true;
                        break;
                    }
                }
            }

            if (isNeeded) {
                // Add reads to required handles
                for (const read of pass.reads) {
                    requiredHandles.add(read.handle);
                }
                activePasses.push(pass);
            }
        }
        
        activePasses.reverse();

        // 2. Automatically deduce GPUTextureUsage
        for (const pass of activePasses) {
            for (const read of pass.reads) {
                const desc = this.resourceDescs.get(read.handle);
                if (desc) desc.usage! |= read.usage;
            }
            for (const write of pass.writes) {
                const desc = this.resourceDescs.get(write.handle);
                if (desc) desc.usage! |= write.usage;
            }
        }

        // 3. Compute Resource Lifetimes defined over active passes
        const resourceLifetimes = new Map<ResourceHandle, { start: number, end: number }>();
        
        for (let i = 0; i < activePasses.length; i++) {
            const pass = activePasses[i];
            const allHandles = new Set([
                ...pass.reads.map(r => r.handle), 
                ...pass.writes.map(w => w.handle)
            ]);
            
            for (const handle of allHandles) {
                if (!resourceLifetimes.has(handle)) {
                    resourceLifetimes.set(handle, { start: i, end: i });
                } else {
                    resourceLifetimes.get(handle)!.end = i;
                }
            }
        }

        return { activePasses, resourceLifetimes };
    }

    public execute(encoder: GPUCommandEncoder) {
        const { activePasses, resourceLifetimes } = this.compile();
        console.log("Active Passes:", activePasses.map(p => p.name));

        const activePhysicals = new Map<ResourceHandle, PhysicalTexture>();

        for (let i = 0; i < activePasses.length; i++) {
            const pass = activePasses[i];

            // Allocate resources that start in this pass
            for (const [handle, lifetime] of resourceLifetimes.entries()) {
                if (lifetime.start === i && !this.importedViews.has(handle)) {
                    const desc = this.resourceDescs.get(handle)!;
                    const name = this.resourceNames.get(handle)!;
                    const phys = this.getPhysicalTexture(desc, name);
                    activePhysicals.set(handle, phys);
                    this.handleToPhysical.set(handle, phys);
                }
            }

            // Execute the pass
            pass.executeFn(encoder, this);

            // Deallocate resources that end after this pass
            for (const [handle, lifetime] of resourceLifetimes.entries()) {
                if (lifetime.end === i && activePhysicals.has(handle)) {
                    const phys = activePhysicals.get(handle)!;
                    phys.framesIdle = 0; // Reset just in case
                    this.physicalPool.push(phys);
                    activePhysicals.delete(handle);
                    this.handleToPhysical.delete(handle); // no longer valid context
                }
            }
        }

        // LRU cleanup: Destroy physical textures not used for 5 frames
        for (let i = this.physicalPool.length - 1; i >= 0; i--) {
            this.physicalPool[i].framesIdle++;
            if (this.physicalPool[i].framesIdle > 5) {
                this.physicalPool[i].texture.destroy();
                this.physicalPool.splice(i, 1);
            }
        }

        // Clean up logical graph for next frame
        this.passes = [];
        this.importedViews.clear();
        this.resourceDescs.clear();
        this.resourceNames.clear();
    }
    
    public clearPhysicalPool() {
        for (const phys of this.physicalPool) {
            phys.texture.destroy();
        }
        this.physicalPool = [];
    }
}
