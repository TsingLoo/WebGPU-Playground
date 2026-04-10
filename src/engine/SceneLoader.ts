import { Entity } from './Entity';
import { Scene } from './Scene';
import { loadGltf } from './GLTFLoader';
import { DirectionalLightComponent, PointLightComponent } from './components/LightComponent';
import { mat4 } from 'wgpu-matrix';
import { device } from '../renderer';
import { buildBVHFromScene } from '../stage/bvh_builder';
import { buildVoxelGrid } from './GLTFLoader';

export interface SceneConfig {
    name: string;
    entities: EntityConfig[];
}

export interface EntityConfig {
    name?: string;
    transform?: {
        translation?: [number, number, number];
        rotation?: [number, number, number, number];
        scale?: [number, number, number];
    };
    components?: ComponentConfig[];
    children?: EntityConfig[];
}

export interface ComponentConfig {
    type: string;
    [key: string]: any;
}

export class SceneLoader {
    public static async loadFromJSON(url: string | SceneConfig): Promise<Scene> {
        let config: SceneConfig;
        
        if (typeof url === 'string') {
            const response = await fetch(url);
            config = await response.json();
        } else {
            config = url;
        }

        const scene = new Scene();
        
        for (const entityConfig of config.entities) {
            const entity = await this.createEntity(entityConfig, scene);
            scene.addEntity(entity);
        }

        return scene;
    }

    private static async createEntity(config: EntityConfig, scene: Scene): Promise<Entity> {
        const entity = new Entity(config.name);

        if (config.transform) {
            if (config.transform.translation) {
                entity.localTransform = mat4.mul(entity.localTransform, mat4.translation(config.transform.translation));
            }
            if (config.transform.rotation) {
                entity.localTransform = mat4.mul(entity.localTransform, mat4.fromQuat(config.transform.rotation));
            }
            if (config.transform.scale) {
                entity.localTransform = mat4.mul(entity.localTransform, mat4.scaling(config.transform.scale));
            }
        }

        if (config.components) {
            for (const compConfig of config.components) {
                if (compConfig.type === 'GLTFModel') {
                    // This is a temporary monolithic mapping. In a full engine, GLTF chunks are components.
                    const result = await loadGltf(compConfig.path);
                    entity.addChild(result.rootEntity);
                    await scene.mergeMaterialAndTextures(device, result.materialDataArray, result.materialCount, result.baseColorImages, result.normalMapImages, result.mrImages, result.emissiveImages, result.baseColorImages.length);
                    scene.bvhData = buildBVHFromScene(scene.root);
                    const voxelResult = buildVoxelGrid(scene.root);
                    scene.voxelGrid = voxelResult.voxelGrid;
                    scene.voxelGridView = voxelResult.voxelGridView;
                } else if (compConfig.type === 'DirectionalLightComponent') {
                    const light = entity.addComponent(new DirectionalLightComponent());
                    if (compConfig.direction) light.direction = compConfig.direction;
                    if (compConfig.color) light.color = compConfig.color;
                    if (compConfig.intensity !== undefined) light.intensity = compConfig.intensity;
                } else if (compConfig.type === 'PointLightComponent') {
                    const light = entity.addComponent(new PointLightComponent());
                    if (compConfig.color) light.color = compConfig.color;
                    if (compConfig.intensity !== undefined) light.intensity = compConfig.intensity;
                    if (compConfig.radius !== undefined) light.radius = compConfig.radius;
                }
            }
        }

        if (config.children) {
            for (const childConfig of config.children) {
                const child = await this.createEntity(childConfig, scene);
                entity.addChild(child);
            }
        }

        return entity;
    }
}
