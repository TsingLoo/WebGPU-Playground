import { Component } from '../Component';

export class CameraComponent extends Component {
    // Adapter/Reference to the existing Camera class
    // In a future pass, this will fully replace the standalone Camera class
    public camera: any = null;

    override onAwake(): void {
        // Init camera if not supplied externally
    }

    override onUpdate(_dt: number): void {
        // Sync the camera's internal position with our Entity's world transform
        if (this.camera && this.entity) {
            // Note: Camera currently handles its own input. We might just
            // sync entity transform -> camera, or camera -> entity depending on who controls it.
            // For now, if camera moves by mouse, we sync camera -> entity so Scene Graph matches.
            if (this.camera.cameraPos) {
                this.entity.localTransform[12] = this.camera.cameraPos[0];
                this.entity.localTransform[13] = this.camera.cameraPos[1];
                this.entity.localTransform[14] = this.camera.cameraPos[2];
                this.entity.setTransformDirty();
            }
        }
    }
}
