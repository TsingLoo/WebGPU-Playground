import { Entity } from './Entity';

export abstract class Component {
    public entity!: Entity;
    public enabled: boolean = true;

    // Called when the component is added to the entity
    public onAwake(): void {}

    // Called every frame
    public onUpdate(_dt: number): void {}

    // Called when the component is removed or entity is destroyed
    public onDestroy(): void {}
}
