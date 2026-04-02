import { Mat4, mat4 } from 'wgpu-matrix';
import { Component } from './Component';
import { Scene } from './Scene';

type Constructor<T> = new (...args: any[]) => T;

export class Entity {
    public name: string = 'Entity';
    public scene: Scene | null = null;
    public parent: Entity | null = null;
    public children: Set<Entity> = new Set();
    
    public localTransform: Mat4 = mat4.identity();
    public worldTransform: Mat4 = mat4.identity();
    public isTransformDirty: boolean = true;

    private components: Component[] = [];

    constructor(name?: string) {
        if (name) this.name = name;
    }

    public setParent(newParent: Entity | null) {
        if (this.parent === newParent) return;
        
        if (this.parent) {
            this.parent.children.delete(this);
        }

        this.parent = newParent;
        if (this.parent) {
            this.parent.children.add(this);
        }
        
        this.setScene(this.parent ? this.parent.scene : null);
        this.setTransformDirty();
    }

    public setScene(newScene: Scene | null) {
        if (this.scene === newScene) return;

        // Unregister from old
        if (this.scene) {
            for (const c of this.components) {
                this.scene._unregisterComponent(c);
            }
        }

        this.scene = newScene;

        // Register to new
        if (this.scene) {
            for (const c of this.components) {
                this.scene._registerComponent(c);
            }
        }

        for (const child of this.children) {
            child.setScene(newScene);
        }
    }

    public addChild(child: Entity) {
        child.setParent(this);
    }

    public setTransformDirty() {
        this.isTransformDirty = true;
        for (const child of this.children) {
            child.setTransformDirty();
        }
    }

    public wasTransformDirty: boolean = true;

    public updateWorldTransform() {
        if (this.isTransformDirty) {
            if (this.parent) {
                this.worldTransform = mat4.mul(this.parent.worldTransform, this.localTransform);
            } else {
                mat4.copy(this.localTransform, this.worldTransform);
            }
            this.isTransformDirty = false;
            this.wasTransformDirty = true;
        } else {
            this.wasTransformDirty = false;
        }

        // We only trigger updates on children if needed.
        for (const child of this.children) {
            child.updateWorldTransform();
        }
    }

    public addComponent<T extends Component>(component: T): T {
        component.entity = this;
        this.components.push(component);
        component.onAwake();
        if (this.scene) {
            this.scene._registerComponent(component);
        }
        return component;
    }

    public getComponent<T extends Component>(type: Constructor<T>): T | null {
        for (let i = 0; i < this.components.length; i++) {
            if (this.components[i] instanceof type) {
                return this.components[i] as T;
            }
        }
        return null;
    }

    public getComponents<T extends Component>(type: Constructor<T>): T[] {
        return this.components.filter(c => c instanceof type) as T[];
    }

    public getAllComponents(): Component[] {
        return this.components;
    }

    public update(dt: number) {
        for (const comp of this.components) {
            if (comp.enabled) comp.onUpdate(dt);
        }

        for (const child of this.children) {
            child.update(dt);
        }
    }

    public destroy() {
        for (const comp of this.components) {
            comp.onDestroy();
            if (this.scene) {
                this.scene._unregisterComponent(comp);
            }
        }
        this.components = [];

        const childrenCopy = Array.from(this.children);
        for (const child of childrenCopy) {
            child.destroy();
        }

        if (this.parent) {
            this.parent.children.delete(this);
        }
        this.parent = null;
        this.scene = null;
    }
}
