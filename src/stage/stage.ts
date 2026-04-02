import { device } from "../renderer";
import { Camera } from "./camera";
import { RadianceCascades } from "./radiance_cascades";
import { DDGI } from "./ddgi";
import { Environment } from "./environment";
import { Lights } from "./lights";
import { Scene } from "../engine/Scene";
import { VSM } from "./vsm";
import { SSAO } from "./ssao";

export class Stage {
    scene: Scene;
    lights: Lights;
    camera: Camera;
    stats: Stats;
    environment: Environment;
    ddgi: DDGI;
    radianceCascades: RadianceCascades;
    vsm: VSM;
    ssao: SSAO;

    // Sun light
    sunLightBuffer: GPUBuffer;
    sunDirection: [number, number, number] = [-0.17, 0.27, 0.05]; // direction TO light
    sunColor: [number, number, number] = [1.0, 0.95, 0.85];   // warm white
    sunIntensity: number = 10.0;
    sunVolumetricEnabled: boolean = false;
    sunVolumetricIntensity: number = 0.001;
    sunVolumetricHeightFalloff: number = 0.66;
    sunVolumetricHeightScale: number = 2.0;
    sunVolumetricMaxDist: number = 82.0;
    sunVolumetricSteps: number = 16;
    sunEnabled: boolean = true;

    showGIBounds: boolean = false;

    constructor(scene: Scene, lights: Lights, camera: Camera, stats: Stats, environment: Environment) {
        this.scene = scene;
        this.lights = lights;
        this.camera = camera;
        this.stats = stats;
        this.environment = environment;
        this.ddgi = new DDGI(this.camera, this.environment);
        this.radianceCascades = new RadianceCascades(this.camera, this.environment);
        this.vsm = new VSM(this.camera);
        this.ssao = new SSAO();

        // Sync sun direction into VSM
        this.vsm.sunDirection = this.sunDirection;

        // SunLight struct: direction(16) + color(16) + light_vp(64) + shadow_params(16) + volumetric_params(16) = 128 bytes
        this.sunLightBuffer = device.createBuffer({
            label: "Sun Light Uniform",
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.updateSunLight();
    }

    private sunLightData = new Float32Array(32); // Pre-allocated array for GC optimization

    updateSunLight() {
        // Sync sun direction to VSM
        this.vsm.sunDirection = this.sunDirection;

        const d = this.sunDirection;
        const len = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

        // Write to GPU buffer: direction(16) + color(16) + light_vp(64) + shadow_params(16) + volumetric_params(16) = 128 bytes
        const data = this.sunLightData;
        // direction.xyz, w=intensity
        data[0] = d[0] / len; data[1] = d[1] / len; data[2] = d[2] / len; data[3] = this.sunIntensity;
        // color.rgb, a=enabled
        data[4] = this.sunColor[0]; data[5] = this.sunColor[1]; data[6] = this.sunColor[2];
        data[7] = this.sunEnabled ? 1.0 : 0.0;
        // light_vp matrix (16 floats) — placeholder identity, VSM uses its own clipmap VPs
        data[8] = 1; data[13] = 1; data[18] = 1; data[23] = 1;
        // shadow_params: x = texel size, y = bias, z = steps, w = 0
        data[24] = 1.0 / this.vsm.physAtlasSize;
        data[25] = 0.05; // normal bias
        data[26] = this.sunVolumetricSteps;
        data[27] = 0;
        // volumetric_params: x = intensity, y = heightFalloff, z = heightScale, w = maxDist
        data[28] = this.sunVolumetricIntensity;
        data[29] = this.sunVolumetricHeightFalloff;
        data[30] = this.sunVolumetricHeightScale;
        data[31] = this.sunVolumetricMaxDist;

        device.queue.writeBuffer(this.sunLightBuffer, 0, data.buffer);
    }

    private camPosTuple: [number, number, number] = [0, 0, 0];

    renderShadowMap(encoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {
        if (!this.sunEnabled) return;

        const cp = this.camera.cameraPos;
        this.camPosTuple[0] = cp[0];
        this.camPosTuple[1] = cp[1];
        this.camPosTuple[2] = cp[2];

        this.vsm.update(
            encoder,
            depthTextureView,
            this.scene,
            this.camPosTuple,
        );
    }
}
