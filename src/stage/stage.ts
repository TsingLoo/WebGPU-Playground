import { device } from "../renderer";
import { Camera } from "./camera";
import { RadianceCascades } from "./radiance_cascades";
import { DDGI } from "./ddgi";
import { Environment } from "./environment";
import { Lights } from "./lights";
import { Scene } from "../engine/Scene";
import { VSM } from "./vsm";
import { SSAO } from "./ssao";
import { SSR } from "./ssr";
import { NRC } from "./nrc";
import { DirectionalLightComponent, VolumetricFogComponent } from "../engine/components/LightComponent";

export class Stage {
    private _scene: Scene;
    get scene(): Scene { return this._scene; }
    set scene(s: Scene) {
        this._scene = s;
        this.invalidateLightCache();
    }
    lights: Lights;
    camera: Camera;
    stats: Stats;
    environment: Environment;
    ddgi: DDGI;
    radianceCascades: RadianceCascades;
    vsm: VSM;
    ssao: SSAO;
    ssr: SSR;
    nrc: NRC;

    // Sun light GPU buffer — packed from the scene's DirectionalLightComponent each frame
    sunLightBuffer: GPUBuffer;

    showGIBounds: boolean = false;

    constructor(scene: Scene, lights: Lights, camera: Camera, stats: Stats, environment: Environment) {
        this._scene = scene;
        this.lights = lights;
        this.camera = camera;
        this.stats = stats;
        this.environment = environment;
        this.ddgi = new DDGI(this.camera, this.environment);
        this.radianceCascades = new RadianceCascades(this.camera, this.environment);
        this.vsm = new VSM(this.camera);
        this.ssao = new SSAO();
        this.ssr = new SSR();
        this.nrc = new NRC(this.camera, this.environment);

        // SunLight struct: direction(16) + color(16) + light_vp(64) + shadow_params(16) + volumetric_params(16) = 128 bytes
        this.sunLightBuffer = device.createBuffer({
            label: "Sun Light Uniform",
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.updateSunLight();
    }

    // ---- Direct component access (no proxies, no cycles) ----

    /** Cached reference — refreshed lazily when null. */
    private _sunLight: DirectionalLightComponent | null = null;
    private _volumetricFog: VolumetricFogComponent | null = null;

    get sunLight(): DirectionalLightComponent | null {
        if (!this._sunLight) {
            this._sunLight = this.scene.getDirectionalLight();
        }
        return this._sunLight;
    }

    get volumetricFog(): VolumetricFogComponent | null {
        if (!this._volumetricFog) {
            this._volumetricFog = this.scene.getVolumetricFog();
        }
        return this._volumetricFog;
    }

    /** Call when scene changes (e.g. new model loaded) to re-discover components. */
    invalidateLightCache() {
        this._sunLight = null;
        this._volumetricFog = null;
    }

    // ---- Property accessors that read/write the component directly ----
    // These are used by renderers, GUI, and RenderSchema bindings.
    // They read the component's OWN instance properties (not schema-overridden ones).

    get sunDirection(): [number, number, number] {
        return this.sunLight?.direction ?? [-0.17, 0.27, 0.05];
    }
    set sunDirection(v: [number, number, number]) {
        if (this.sunLight) this.sunLight.direction = v;
    }

    get sunColor(): [number, number, number] {
        return this.sunLight?.color ?? [1.0, 0.95, 0.85];
    }
    set sunColor(v: [number, number, number]) {
        if (this.sunLight) this.sunLight.color = v;
    }

    get sunIntensity(): number {
        return this.sunLight?.intensity ?? 10.0;
    }
    set sunIntensity(v: number) {
        if (this.sunLight) this.sunLight.intensity = v;
    }

    get sunEnabled(): boolean {
        return this.sunLight?.enabled ?? true;
    }
    set sunEnabled(v: boolean) {
        if (this.sunLight) this.sunLight.enabled = v;
    }

    get vsmEnabled(): boolean {
        return this.sunLight?.shadowEnabled ?? true;
    }
    set vsmEnabled(v: boolean) {
        if (this.sunLight) this.sunLight.shadowEnabled = v;
    }

    get sunVolumetricEnabled(): boolean {
        return this.volumetricFog?.enabled ?? false;
    }
    set sunVolumetricEnabled(v: boolean) {
        if (this.volumetricFog) this.volumetricFog.enabled = v;
    }

    get sunVolumetricIntensity(): number {
        return this.volumetricFog?.intensity ?? 0.001;
    }
    set sunVolumetricIntensity(v: number) {
        if (this.volumetricFog) this.volumetricFog.intensity = v;
    }

    get sunVolumetricHeightFalloff(): number {
        return this.volumetricFog?.heightFalloff ?? 0.66;
    }
    set sunVolumetricHeightFalloff(v: number) {
        if (this.volumetricFog) this.volumetricFog.heightFalloff = v;
    }

    get sunVolumetricHeightScale(): number {
        return this.volumetricFog?.heightScale ?? 2.0;
    }
    set sunVolumetricHeightScale(v: number) {
        if (this.volumetricFog) this.volumetricFog.heightScale = v;
    }

    get sunVolumetricMaxDist(): number {
        return this.volumetricFog?.maxDist ?? 82.0;
    }
    set sunVolumetricMaxDist(v: number) {
        if (this.volumetricFog) this.volumetricFog.maxDist = v;
    }

    get sunVolumetricSteps(): number {
        return this.volumetricFog?.steps ?? 16;
    }
    set sunVolumetricSteps(v: number) {
        if (this.volumetricFog) this.volumetricFog.steps = v;
    }

    // ---- GPU buffer packing (called every frame by renderers) ----

    private sunLightData = new Float32Array(32);

    updateSunLight() {
        const light = this.sunLight;
        const fog = this.volumetricFog;

        const dir = light?.direction ?? [-0.17, 0.27, 0.05];
        const col = light?.color ?? [1.0, 0.95, 0.85];
        const intensity = light?.intensity ?? 10.0;
        const enabled = light?.enabled ?? true;
        const shadowOn = light?.shadowEnabled ?? true;

        const fogEnabled = fog?.enabled ?? false;
        const fogIntensity = fog?.intensity ?? 0.001;
        const fogHeightFalloff = fog?.heightFalloff ?? 0.66;
        const fogHeightScale = fog?.heightScale ?? 2.0;
        const fogMaxDist = fog?.maxDist ?? 82.0;
        const fogSteps = fog?.steps ?? 16;

        // Sync sun direction to VSM
        this.vsm.sunDirection = dir as [number, number, number];

        const len = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) || 1;

        const data = this.sunLightData;
        // direction.xyz, w=intensity
        data[0] = dir[0] / len; data[1] = dir[1] / len; data[2] = dir[2] / len; data[3] = intensity;
        // color.rgb, a=enabled
        data[4] = col[0]; data[5] = col[1]; data[6] = col[2];
        data[7] = enabled ? 1.0 : 0.0;
        // light_vp matrix (16 floats) — placeholder identity, VSM uses its own clipmap VPs
        data.fill(0, 8, 24);
        data[8] = 1; data[13] = 1; data[18] = 1; data[23] = 1;
        // shadow_params: x = texel size, y = bias, z = steps, w = vsmEnabled
        data[24] = 1.0 / this.vsm.physAtlasSize;
        data[25] = 0.05;
        data[26] = fogSteps;
        data[27] = shadowOn ? 1.0 : 0.0;
        // volumetric_params: x = intensity, y = heightFalloff, z = heightScale, w = maxDist
        data[28] = fogEnabled ? fogIntensity : 0.0;
        data[29] = fogHeightFalloff;
        data[30] = fogHeightScale;
        data[31] = fogMaxDist;

        device.queue.writeBuffer(this.sunLightBuffer, 0, data.buffer);
    }

    // ---- Shadow Map ----

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
