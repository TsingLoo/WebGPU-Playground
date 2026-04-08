import { Entity } from './Entity';
import { Component } from './Component';
import { BVHData } from '../stage/bvh_builder';
import { Material, Primitive } from './GLTFLoader';
import { MeshRenderer } from './components/MeshRenderer';

type Constructor<T> = new (...args: any[]) => T;

export class Scene {
    public root: Entity;
    
    // Global data assigned by GLTF Loader temporarily
    public bvhData!: BVHData;
    public voxelGrid!: GPUTexture;
    public voxelGridView!: GPUTextureView;
    public globalMaterialBuffer!: GPUBuffer;
    public baseColorTexArray!: GPUTexture;
    public baseColorTexArrayView!: GPUTextureView;

    // CPU Data required for appending
    public materialDataArray: Float32Array = new Float32Array(0);
    public materialCount: number = 0;
    public layerCount: number = 1; // Start at 1 to have at least a dummy layer

    // Fast lookup caches
    private componentsCache: Map<Function, Set<Component>> = new Map();

    constructor() {
        this.root = new Entity("SceneRoot");
        this.root.scene = this;
    }

    public addEntity(entity: Entity) {
        entity.setParent(this.root);
    }

    /**
     * Internal method called by Entity when a component is added.
     */
    public _registerComponent(component: Component) {
        const type = component.constructor;
        if (!this.componentsCache.has(type)) {
            this.componentsCache.set(type, new Set());
        }
        this.componentsCache.get(type)!.add(component);
    }

    /**
     * Internal method called by Entity when a component is removed / destroyed.
     */
    public _unregisterComponent(component: Component) {
        const type = component.constructor;
        if (this.componentsCache.has(type)) {
            this.componentsCache.get(type)!.delete(component);
        }
    }

    /**
     * Returns all components of a specific exact type.
     */
    public getComponents<T extends Component>(type: Constructor<T>): T[] {
        const set = this.componentsCache.get(type);
        if (set) {
            return Array.from(set) as T[];
        }
        return [];
    }

    /**
     * Updates all entities in the scene.
     * 1. Recompute world transforms (CPU-side mat4 hierarchy)
     * 2. Run component onUpdate (e.g. MeshRenderer uploads worldTransform to GPU)
     */
    public update(dt: number) {
        this.root.updateWorldTransform();
        this.root.update(dt);
    }

    /**
     * Compatibility layer: iterates the scene graph and calls callbacks per
     * MeshRenderer / Material / Primitive, matching the old Scene.iterate() API.
     * 
     * @param nodeFunction   Called once per entity that has a MeshRenderer. Receives the MeshRenderer.
     * @param materialFunction Called for each primitive's material.
     * @param primitiveFunction Called for each primitive to issue draw calls.
     * @param isOpaque If provided, filters primitives by opaque vs cutout type.
     *                 If undefined, iterates ALL primitives (used by shadow passes).
     */
    public iterate(
        nodeFunction: (mr: MeshRenderer) => void,
        materialFunction: (material: Material) => void,
        primitiveFunction: (primitive: Primitive) => void,
        isOpaque?: boolean
    ) {
        const meshRenderers = this.componentsCache.get(MeshRenderer);
        if (!meshRenderers) return;

        for (const comp of meshRenderers) {
            const mr = comp as MeshRenderer;
            if (mr.mesh && mr.modelBindGroup) {
                nodeFunction(mr);
                const primitives = mr.mesh.primitives;
                for (let i = 0; i < primitives.length; i++) {
                    const prim = primitives[i];
                    if (isOpaque === undefined || prim.material.isOpaque === isOpaque) {
                        materialFunction(prim.material);
                        primitiveFunction(prim);
                    }
                }
            }
        }
    }

    /**
     * Merges a new batch of CPU materials and new GPU images into the existing monolithic Scene resources.
     * WebGPU requires replacing ArrayTextures and StorageBuffers to increase their size.
     */
    public async mergeMaterialAndTextures(
        device: GPUDevice,
        newMaterialData: Float32Array,
        newCount: number,
        newImages: GPUTexture[], // The temporarily uploaded individual resized images 
        newLayerCount: number
    ) {
        // --- Merge Global Material Buffer ---
        const combinedMaterials = new Float32Array(this.materialCount * 12 + newCount * 12);
        combinedMaterials.set(this.materialDataArray);
        combinedMaterials.set(newMaterialData, this.materialCount * 12);

        this.materialDataArray = combinedMaterials;
        this.materialCount += newCount;

        const newGlobalMaterialBuffer = device.createBuffer({
            label: "Global Material Buffer (Merged)",
            size: this.materialDataArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(newGlobalMaterialBuffer, 0, this.materialDataArray as any);
        
        if (this.globalMaterialBuffer) {
            this.globalMaterialBuffer.destroy();
        }
        this.globalMaterialBuffer = newGlobalMaterialBuffer;

        // --- Merge BaseColor Texture Array ---
        const TEX_ARRAY_SIZE = 256;
        const totalLayers = this.layerCount + newLayerCount;

        const newBaseColorTexArray = device.createTexture({
            label: "BaseColor Texture Array (Merged)",
            size: [TEX_ARRAY_SIZE, TEX_ARRAY_SIZE, totalLayers],
            format: 'rgba8unorm',
            dimension: '2d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const commandEncoder = device.createCommandEncoder();

        // 1. Copy old layers if they exist
        if (this.baseColorTexArray && this.layerCount > 0) {
            commandEncoder.copyTextureToTexture(
                { texture: this.baseColorTexArray, aspect: 'all' },
                { texture: newBaseColorTexArray, aspect: 'all' },
                [TEX_ARRAY_SIZE, TEX_ARRAY_SIZE, this.layerCount]
            );
        }

        // 2. Copy new images 
        for (let i = 0; i < newImages.length; i++) {
            commandEncoder.copyTextureToTexture(
                { texture: newImages[i], aspect: 'all' },
                { texture: newBaseColorTexArray, origin: [0, 0, this.layerCount + i] },
                [TEX_ARRAY_SIZE, TEX_ARRAY_SIZE, 1]
            );
        }

        device.queue.submit([commandEncoder.finish()]);

        if (this.baseColorTexArray) {
            this.baseColorTexArray.destroy();
        }

        this.baseColorTexArray = newBaseColorTexArray;
        this.baseColorTexArrayView = newBaseColorTexArray.createView({ dimension: '2d-array' });
        this.layerCount = totalLayers;
    }
}
