import { Component } from '../Component';

export class DirectionalLightComponent extends Component {
    public direction: [number, number, number] = [-0.17, 0.27, 0.05];
    public color: [number, number, number] = [1.0, 0.95, 0.85];
    public intensity: number = 10.0;
    
    // Volumetric generic options could go here or in a separate component.
    public volumetricEnabled: boolean = false;
    public volumetricIntensity: number = 0.001;
    public volumetricHeightFalloff: number = 0.66;
    public volumetricHeightScale: number = 2.0;
    public volumetricMaxDist: number = 82.0;
    public volumetricSteps: number = 16;
}

export class PointLightComponent extends Component {
    public color: [number, number, number] = [1.0, 1.0, 1.0];
    public intensity: number = 1.0;
    public radius: number = 5.0;

    // The transform provides the position.
    
    get position(): Float32Array {
        // worldTransform column 3 is translation
        const w = this.entity.worldTransform;
        return new Float32Array([w[12], w[13], w[14]]);
    }
}
