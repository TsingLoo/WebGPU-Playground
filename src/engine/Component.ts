import { Entity } from './Entity';

export abstract class Component {
    public entity!: Entity;
    public enabled: boolean = true;
    
    // Optional UI options mapping property name to an array of valid selections or object configs
    public getUIOptions?(): Record<string, any>;

    // Called when the component is added to the entity
    public onAwake(): void {}

    // Called every frame
    public onUpdate(_dt: number): void {}

    // Called when the component is removed or entity is destroyed
    public onDestroy(): void {}
}
