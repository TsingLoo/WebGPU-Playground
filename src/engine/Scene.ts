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
    public normalMapTexArray!: GPUTexture;
    public normalMapTexArrayView!: GPUTextureView;
    public mrTexArray!: GPUTexture;
    public mrTexArrayView!: GPUTextureView;

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
        newBaseColorImages: GPUTexture[],
        newNormalMapImages: GPUTexture[],
        newMRImages: GPUTexture[],
        newLayerCount: number
    ) {
        const FLOATS_PER_MAT = 16;
        // --- Merge Global Material Buffer ---
        const combinedMaterials = new Float32Array(this.materialCount * FLOATS_PER_MAT + newCount * FLOATS_PER_MAT);
        
        // Use Int32Array view to precisely copy bit patterns without triggering JS NaN-normalization
        const combinedIntView = new Int32Array(combinedMaterials.buffer);
        const oldIntView = new Int32Array(this.materialDataArray.buffer);
        const newIntView = new Int32Array(newMaterialData.buffer);

        combinedIntView.set(oldIntView);
        combinedIntView.set(newIntView, this.materialCount * FLOATS_PER_MAT);

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

        // --- Helper: Build/Merge a Texture Array ---
        const TEX_ARRAY_SIZE = 1024;
        const totalLayers = this.layerCount + newLayerCount;

        const mergeTexArray = (label: string, oldArray: GPUTexture | null, oldLayers: number, newImages: GPUTexture[], format: GPUTextureFormat = 'rgba8unorm'): GPUTexture => {
            const tex = device.createTexture({
                label, size: [TEX_ARRAY_SIZE, TEX_ARRAY_SIZE, totalLayers],
                format: format, dimension: '2d',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            const enc = device.createCommandEncoder();
            if (oldArray && oldLayers > 0) {
                enc.copyTextureToTexture(
                    { texture: oldArray, aspect: 'all' },
                    { texture: tex, aspect: 'all' },
                    [TEX_ARRAY_SIZE, TEX_ARRAY_SIZE, oldLayers]
                );
            }
            for (let i = 0; i < newImages.length; i++) {
                enc.copyTextureToTexture(
                    { texture: newImages[i], aspect: 'all' },
                    { texture: tex, origin: [0, 0, oldLayers + i] },
                    [TEX_ARRAY_SIZE, TEX_ARRAY_SIZE, 1]
                );
            }
            device.queue.submit([enc.finish()]);
            if (oldArray) oldArray.destroy();
            return tex;
        };

        // --- Merge BaseColor Texture Array ---
        this.baseColorTexArray = mergeTexArray("BaseColor Array", this.baseColorTexArray, this.layerCount, newBaseColorImages);
        this.baseColorTexArrayView = this.baseColorTexArray.createView({ dimension: '2d-array' });

        // --- Merge Normal Map Texture Array (linear) ---
        this.normalMapTexArray = mergeTexArray("NormalMap Array", this.normalMapTexArray, this.layerCount, newNormalMapImages);
        this.normalMapTexArrayView = this.normalMapTexArray.createView({ dimension: '2d-array' });

        // --- Merge MR Texture Array (linear) ---
        this.mrTexArray = mergeTexArray("MR Array", this.mrTexArray, this.layerCount, newMRImages);
        this.mrTexArrayView = this.mrTexArray.createView({ dimension: '2d-array' });

        this.layerCount = totalLayers;
    }
}
