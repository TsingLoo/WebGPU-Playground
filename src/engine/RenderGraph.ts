import { device, canvas } from '../renderer';

export type ResourceHandle = number;

export interface TextureDescriptor {
    format: GPUTextureFormat;
    usage: GPUTextureUsageFlags;
    width?: number; // if undefined, uses canvas width
    height?: number; // if undefined, uses canvas height
    scale?: number; // multiplier for canvas size
}

interface PhysicalTexture {
    texture: GPUTexture;
    view: GPUTextureView;
    desc: TextureDescriptor;
    width: number;
    height: number;
}

export type PassExecuteFn = (encoder: GPUCommandEncoder, pass: PassResolver) => void;

class RenderPassData {
    name: string;
    reads: Map<ResourceHandle, boolean> = new Map();
    writes: Map<ResourceHandle, boolean> = new Map();
    executeFn!: PassExecuteFn;
    
    constructor(name: string) {
        this.name = name;
    }
}

export interface PassResolver {
    getTextureView(handle: ResourceHandle): GPUTextureView;
}

export class PassBuilder {
    constructor(private passParams: RenderPassData) {}

    readTexture(handle: ResourceHandle): PassBuilder {
        this.passParams.reads.set(handle, true);
        return this;
    }

    writeTexture(handle: ResourceHandle): PassBuilder {
        this.passParams.writes.set(handle, true);
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
    public getTextureView(handle: ResourceHandle): GPUTextureView {
        if (this.importedViews.has(handle)) {
            return this.importedViews.get(handle)!;
        }
        const phys = this.handleToPhysical.get(handle);
        if (!phys) {
            throw new Error(`Texture handle ${handle} (${this.resourceNames.get(handle)}) has no physical texture bound.`);
        }
        return phys.view;
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
        for (let i = 0; i < this.physicalPool.length; i++) {
            const pt = this.physicalPool[i];
            if (pt.desc.format === desc.format && 
                pt.desc.usage === desc.usage && 
                pt.width === w && 
                pt.height === h) {
                // Remove from free pool
                this.physicalPool.splice(i, 1);
                return pt;
            }
        }
        
        // Create new if not found
        const texture = device.createTexture({
            label: `RG_PhysTex_${name}`,
            size: [w, h],
            format: desc.format,
            usage: desc.usage
        });
        
        return {
            texture,
            view: texture.createView(),
            desc,
            width: w,
            height: h
        };
    }

    public compile() {
        // 1. Cull passes? (Skipping for now to ensure all logic runs)
        const activePasses = this.passes;

        // 2. Compute Resource Lifetimes (First pass to last pass)
        const resourceLifetimes = new Map<ResourceHandle, { start: number, end: number }>();
        
        for (let i = 0; i < activePasses.length; i++) {
            const pass = activePasses[i];
            const allHandles = new Set([...pass.reads.keys(), ...pass.writes.keys()]);
            
            for (const handle of allHandles) {
                if (!resourceLifetimes.has(handle)) {
                    resourceLifetimes.set(handle, { start: i, end: i });
                } else {
                    resourceLifetimes.get(handle)!.end = i;
                }
            }
        }

        // Return lifetimes for execution
        return { activePasses, resourceLifetimes };
    }

    public execute(encoder: GPUCommandEncoder) {
        const { activePasses, resourceLifetimes } = this.compile();

        // Execution and Memory Aliasing
        const activePhysicals = new Map<ResourceHandle, PhysicalTexture>();

        for (let i = 0; i < activePasses.length; i++) {
            const pass = activePasses[i];

            // Allocate resources that start in this pass
            for (const [handle, lifetime] of resourceLifetimes.entries()) {
                if (lifetime.start === i && !this.importedViews.has(handle)) {
                    // It's a new virtual texture born here
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
                    this.physicalPool.push(phys);
                    activePhysicals.delete(handle);
                    this.handleToPhysical.delete(handle); // no longer valid
                }
            }
        }

        // Clean up passes so the graph can be rebuilt next frame
        this.passes = [];
        this.importedViews.clear();
        this.resourceDescs.clear();
        this.resourceNames.clear();
    }
    
    // Clear pool if resolution changes or memory needs to be freed
    public clearPhysicalPool() {
        // GPUTextures will be garbage collected by JS once refs are removed
        this.physicalPool = [];
    }
}
