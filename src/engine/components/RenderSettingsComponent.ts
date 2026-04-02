import { Component } from '../Component';

export class VSMShadowComponent extends Component {
    public hideEnableInUI = true;
    public physAtlasSize: number = 4096;
    public pageSize: number = 128;
    public numClipmapLevels: number = 6;
    public pagesPerLevelAxis: number = 128;

    public virtualSizeMax: string = '16384';
    public maxPhysPagesInfo: string = '1024';

    public getUIOptions(): Record<string, any> {
        return {
            physAtlasSize: [1024, 2048, 4096, 8192],
            pageSize: [64, 128, 256],
            pagesPerLevelAxis: [32, 64, 128, 256],
            numClipmapLevels: { min: 1, step: 1 }
        };
    }
}

export class GIComponent extends Component {
    public hideEnableInUI = true;
    public mode: string = 'off';
    public showGIBounds: boolean = false;

    public getUIOptions(): Record<string, any[]> {
        return {
            mode: [
                { label: 'Off', value: 'off' },
                { label: 'DDGI', value: 'ddgi' },
                { label: 'Radiance Cascades', value: 'rc' }
            ]
        };
    }
}

export class DDGIComponent extends Component {
    public hideEnableInUI = true;
    public irradianceHysteresis: number = 0.97;
    public visibilityHysteresis: number = 0.98;
    public probeTraceAmbient: number = 0.1;
}

export class RadianceCascadesComponent extends Component {
    public hideEnableInUI = true;
    public intensity: number = 1.0;
    public ambient: number = 0.05;
    public hysteresis: number = 0.8;
    public debugMode: number = 0;
    
    public gridMin: [number, number, number] = [-15.0, -2.0, -8.0];
    public gridMax: [number, number, number] = [15.0, 10.0, 8.0];

    public getUIOptions(): Record<string, any[]> {
        return {
            debugMode: [
                { label: 'Off', value: 0 },
                { label: 'Show GI Only', value: 1 },
                { label: 'Show Probe Atlas', value: 2 }
            ]
        };
    }
}

export class PointLightSettingsComponent extends Component {
    public hideEnableInUI = true;
    public enabled: boolean = false;
    public count: number = 0;

    public getUIOptions(): Record<string, any> {
        return {
            count: { min: 0, step: 1 }
        };
    }
}

export class SSAOComponent extends Component {
    public radius: number = 1.0;
    public bias: number = 0.025;
    public power: number = 1.0;
}
