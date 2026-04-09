import { device, canvas } from '../renderer';

export enum RenderResource {
    SceneDepth = "SceneDepth",
    SceneColor = "SceneColor",
    GBufferAlbedo = "GBufferAlbedo",
    GBufferNormal = "GBufferNormal",
    GBufferPosition = "GBufferPosition",
    GBufferSpecular = "GBufferSpecular",
    ShadingOutput = "ShadingOutput",
    TransientFull_R16F_A = "TransientFull_R16F_A",
    TransientFull_R16F_B = "TransientFull_R16F_B",
    TransientFull_RGBA16F_A = "TransientFull_RGBA16F_A",
    SSRDummy1x1 = "SSRDummy1x1",
    VolumetricHalfRes = "VolumetricHalfRes",
}

export interface ResourceDescriptor {
    format: GPUTextureFormat;
    usage: GPUTextureUsageFlags;
    width?: number; // if undefined, uses canvas width
    height?: number; // if undefined, uses canvas height
    scale?: number; // multiplier for canvas size
}

export class RenderTexManager {
    private static textures = new Map<string, GPUTexture>();
    private static views = new Map<string, GPUTextureView>();
    private static formats = new Map<string, GPUTextureFormat>();

    public static getTextureView(name: RenderResource, desc: ResourceDescriptor): GPUTextureView {
        let w = desc.width !== undefined ? desc.width : canvas.width;
        let h = desc.height !== undefined ? desc.height : canvas.height;
        if (desc.scale !== undefined) {
            w = Math.max(1, Math.floor(w * desc.scale));
            h = Math.max(1, Math.floor(h * desc.scale));
        }

        const existingTex = this.textures.get(name);
        if (existingTex) {
            // Check if resize or format change is needed
            if (existingTex.width === w && existingTex.height === h && existingTex.format === desc.format && existingTex.usage === desc.usage) {
                return this.views.get(name)!;
            }
            // Configuration changed, destroy old texture
            existingTex.destroy();
            this.textures.delete(name);
            this.views.delete(name);
        }

        // Create new physical texture
        const tex = device.createTexture({
            label: `RG_Texture_${name}`,
            size: [w, h],
            format: desc.format,
            usage: desc.usage
        });

        const view = tex.createView();
        this.textures.set(name, tex);
        this.views.set(name, view);
        this.formats.set(name, desc.format);
        
        return view;
    }

    public static has(name: RenderResource): boolean {
        return this.textures.has(name);
    }

    public static getFormat(name: RenderResource): GPUTextureFormat {
        return this.formats.get(name)!;
    }

    public static clearAll() {
        for (const tex of this.textures.values()) {
            tex.destroy();
        }
        this.textures.clear();
        this.views.clear();
        this.formats.clear();
    }
}

let nextId = 1;
const objectIds = new WeakMap<any, number>();
function getObjectId(obj: any): number {
    if (typeof obj !== 'object' || obj === null) return 0;
    // For dictionaries like { buffer: GPUBuffer }
    if (!('buffer' in obj) && !('sampleType' in obj) && obj.constructor.name === 'Object') {
        const key = Object.keys(obj)[0];
        if (key && typeof obj[key] === 'object') return getObjectId(obj[key]);
    }
    if (!objectIds.has(obj)) objectIds.set(obj, nextId++);
    return objectIds.get(obj)!;
}

export class BindGroupCache {
    private static cache = new Map<string, GPUBindGroup>();

    /**
     * Retrieves or creates a GPUBindGroup. Hash is computed from the physical object references.
     * Perfect for RenderGraph transient dependencies.
     */
    public static get(desc: { label: string, layout: GPUBindGroupLayout, entries: Iterable<GPUBindGroupEntry> }): GPUBindGroup {
        const layoutId = getObjectId(desc.layout);
        let hash = `${desc.label}_${layoutId}`;
        const entriesArray = Array.from(desc.entries);
        for (const entry of entriesArray) {
            hash += `|${entry.binding}:${getObjectId(entry.resource)}`;
        }

        if (!this.cache.has(hash)) {
            const bg = device.createBindGroup(desc);
            this.cache.set(hash, bg);
        }
        return this.cache.get(hash)!;
    }

    public static clearAll() {
        this.cache.clear();
    }
}
