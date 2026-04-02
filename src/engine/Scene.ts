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
                    if (isOpaque === undefined || (prim.material.type === 'opaque') === isOpaque) {
                        materialFunction(prim.material);
                        primitiveFunction(prim);
                    }
                }
            }
        }
    }
}
