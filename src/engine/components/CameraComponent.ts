import { Component } from '../Component';

export class CameraComponent extends Component {
    // Adapter/Reference to the existing Camera class
    // In a future pass, this will fully replace the standalone Camera class
    public camera: any = null;
    
    // Internal cache to track if the SceneTreeUI modified the transform
    private lastKnownUiPos: [number, number, number] | null = null;

    override onAwake(): void {
        // Init camera if not supplied externally
    }

    override onUpdate(_dt: number): void {
        if (this.camera && this.entity) {
            const lt = this.entity.localTransform;
            
            if (this.lastKnownUiPos) {
                // Check if the UI modified the transform
                const uiChanged = Math.abs(lt[12] - this.lastKnownUiPos[0]) > 0.0001 ||
                                  Math.abs(lt[13] - this.lastKnownUiPos[1]) > 0.0001 ||
                                  Math.abs(lt[14] - this.lastKnownUiPos[2]) > 0.0001;
                
                if (uiChanged) {
                    // UI drove the camera! Push this back to the physical camera buffer
                    this.camera.cameraPos = new Float32Array([lt[12], lt[13], lt[14]]);
                } else {
                    // Pull from the camera physical controls
                    lt[12] = this.camera.cameraPos[0];
                    lt[13] = this.camera.cameraPos[1];
                    lt[14] = this.camera.cameraPos[2];
                    this.entity.setTransformDirty();
                }
            } else {
                // Initial sync from camera
                lt[12] = this.camera.cameraPos[0];
                lt[13] = this.camera.cameraPos[1];
                lt[14] = this.camera.cameraPos[2];
                this.entity.setTransformDirty();
            }
            
            // Save state for next frame
            this.lastKnownUiPos = [lt[12], lt[13], lt[14]];
        }
    }
}
