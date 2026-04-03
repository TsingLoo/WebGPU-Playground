const isMobileDevice = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';

import { ForwardPlusRenderer } from './renderers/forward_plus';
import { ClusteredDeferredRenderer } from './renderers/clustered_deferred';

// @ts-ignore
import parseHdr from 'parse-hdr';
// @ts-ignore
import parseExr from 'parse-exr';

import { Scene } from './engine/Scene';
import { Entity } from './engine/Entity';
import { CameraComponent } from './engine/components/CameraComponent';
import { DirectionalLightComponent, PointLightComponent, VolumetricFogComponent } from './engine/components/LightComponent';
import { VSMShadowComponent, GIComponent, DDGIComponent, RadianceCascadesComponent, SSAOComponent, SSRComponent, PointLightSettingsComponent } from './engine/components/RenderSettingsComponent';
import { SceneTreeUI } from './engine/SceneTreeUI';
import { applyComponentSchema } from './engine/RenderSchema';
import { setupLoaders, loadGltf, loadGltfBuffer } from './engine/GLTFLoader';
import { Lights } from './stage/lights';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';
import { Environment } from './stage/environment';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const monitorTime = 8.0;
const restTime = 1000;

await initWebGPU();
setupLoaders();

let scene = new Scene();
const gltfResult = await loadGltf('./scenes/sponza/Sponza.gltf');
scene.root.addChild(gltfResult.rootEntity);
scene.bvhData = gltfResult.bvhData;
scene.voxelGrid = gltfResult.voxelGrid;
scene.voxelGridView = gltfResult.voxelGridView;
scene.globalMaterialBuffer = gltfResult.globalMaterialBuffer;
scene.baseColorTexArray = gltfResult.baseColorTexArray;
scene.baseColorTexArrayView = gltfResult.baseColorTexArrayView;
scene.root.updateWorldTransform();

const sceneTreeUI = new SceneTreeUI();
sceneTreeUI.setScene(scene);



const camera = new Camera();
const lights = new Lights(camera);
const environment = new Environment();

// --- Add helpers ---
function addHelpersToScene(targetScene: Scene, targetCamera: Camera, stageObj: Stage) {
    // We pass globals containing elements that might be undefined at initialization time (like setRenderer)
    // but the Schema will look them up dynamically when onUpdate is called.
    const globals = {
        get setRenderer() { return typeof setRenderer !== 'undefined' ? setRenderer : undefined; },
        get renderModeController() { return typeof renderModeController !== 'undefined' ? renderModeController : undefined; }
    };

    const cameraEntity = new Entity("Main Camera");
    const cameraComp = new CameraComponent();
    cameraComp.camera = targetCamera;
    cameraEntity.addComponent(cameraComp);
    targetScene.root.addChild(cameraEntity);

    const sunEntity = new Entity("Directional Light (Sun)");
    const sunComp = new DirectionalLightComponent();
    sunComp.direction = stageObj.sunDirection;
    sunComp.color = stageObj.sunColor;
    applyComponentSchema(sunComp, 'DirectionalLightComponent', stageObj, globals);
    sunEntity.addComponent(sunComp);

    // VSM Shadow Component appended to Sun Entity
    const vsmComp = new VSMShadowComponent();
    applyComponentSchema(vsmComp, 'VSMShadowComponent', stageObj, globals);
    // wire enabled to sunEnabled + trigger lighting sync
    Object.defineProperty(vsmComp, 'enabled', { 
        get: () => stageObj.sunEnabled, 
        set: (v) => { 
            stageObj.sunEnabled = v; 
            stageObj.updateSunLight();
        }, 
        enumerable: true 
    });
    Object.defineProperty(vsmComp, 'virtualSizeMax', { get: () => String(stageObj.vsm.virtualSize), set: () => {}, enumerable: true });
    Object.defineProperty(vsmComp, 'maxPhysPagesInfo', { get: () => String(stageObj.vsm.maxPhysPages), set: () => {}, enumerable: true });
    sunEntity.addComponent(vsmComp);
    targetScene.root.addChild(sunEntity);

    const volEntity = new Entity("Global Volume (Fog)");
    const volComp = new VolumetricFogComponent();
    applyComponentSchema(volComp, 'VolumetricFogComponent', stageObj, globals);
    volEntity.addComponent(volComp);
    targetScene.root.addChild(volEntity);

    const pointLightsEntity = new Entity("Point Lights Manager");
    const pointLightsComp = new PointLightComponent();
    pointLightsComp.intensity = Lights.lightIntensity;
    pointLightsEntity.addComponent(pointLightsComp);
    
    const plSettingsComp = new PointLightSettingsComponent();
    // Special bindings for point lights count logic
    Object.defineProperty(plSettingsComp, 'enabled', {
        get: () => lights.numLights > 0,
        set: (v) => {
            if (v) {
                lights.numLights = plSettingsComp.localSavedNumLights > 0 ? plSettingsComp.localSavedNumLights : 100;
            } else {
                plSettingsComp.localSavedNumLights = lights.numLights;
                lights.numLights = 0;
            }
            lights.updateLightSetUniformNumLights();
        },
        enumerable: true
    });
    Object.defineProperty(plSettingsComp, 'count', {
        get: () => lights.numLights,
        set: (v) => {
            if (v > 0) plSettingsComp.localSavedNumLights = v;
            lights.numLights = v <= Lights.maxNumLights ? v : Lights.maxNumLights;
            lights.updateLightSetUniformNumLights();
        },
        enumerable: true
    });
    pointLightsEntity.addComponent(plSettingsComp);
    targetScene.root.addChild(pointLightsEntity);

    // --- Global Illumination ---
    const giEntity = new Entity("Global Illumination");
    const giComp = new GIComponent();
    Object.defineProperty(giComp, 'mode', {
        get: () => {
            if (stageObj.ddgi.enabled) return 'ddgi';
            if (stageObj.radianceCascades.enabled) return 'rc';
            return 'off';
        },
        set: (v) => {
            stageObj.ddgi.enabled = (v === 'ddgi');
            stageObj.radianceCascades.enabled = (v === 'rc');
            stageObj.ddgi.updateUniforms();
            stageObj.radianceCascades.updateUniforms();
        },
        enumerable: true
    });
    applyComponentSchema(giComp, 'GIComponent', stageObj, globals);
    giEntity.addComponent(giComp);

    // Add sub-components
    const ddgiComp = new DDGIComponent();
    applyComponentSchema(ddgiComp, 'DDGIComponent', stageObj, globals);
    giEntity.addComponent(ddgiComp);

    const rcComp = new RadianceCascadesComponent();
    applyComponentSchema(rcComp, 'RadianceCascadesComponent', stageObj, globals);
    giEntity.addComponent(rcComp);
    targetScene.root.addChild(giEntity);

    // --- Post Processing ---
    const ppEntity = new Entity("Post Processing");
    const ssaoComp = new SSAOComponent();
    applyComponentSchema(ssaoComp, 'SSAOComponent', stageObj, globals);
    ppEntity.addComponent(ssaoComp);
    
    const ssrComp = new SSRComponent();
    applyComponentSchema(ssrComp, 'SSRComponent', stageObj, globals);
    ppEntity.addComponent(ssrComp);
    targetScene.root.addChild(ppEntity);
}

const stats = new Stats();

const originalStatsBegin = stats.begin.bind(stats);

stats.begin = () => {
    originalStatsBegin();

    const now = performance.now();

    if (avgStats.collecting) {
        const elapsedTime = (now - avgStats.startTime) / 1000; 

        if (elapsedTime < monitorTime) {
            avgStats.frameCount++;
        } else {
            avgStats.collecting = false;
            if (avgStats.frameCount > 0) {
                const avg = avgStats.frameCount / monitorTime;
                avgStats.avgFPS_20s = avg.toFixed(2);
            }else{
                avgStats.avgFPS_20s = 'N/A';
            }
        }
    }
    
    avgStats.lastFrameTime = now;
};

stats.showPanel(0);
document.body.appendChild(stats.dom);

const resultsElement = document.createElement('div');
resultsElement.style.cssText = `
    position: absolute;
    bottom: 10px;
    left: 10px;
    padding: 8px;
    background-color: rgba(0, 0, 0, 0.75);
    color: #00FF00;
    font-family: monospace;
    font-size: 14px;
    z-index: 100;
    max-height: 40vh;
    width: calc(100vw - 40px);
    overflow-y: auto;
    white-space: pre-wrap;
    display: none;
    box-sizing: border-box;
`;
document.body.appendChild(resultsElement);

const avgStats = {
    startTime: performance.now(),
    lastFrameTime: performance.now(),
    frameCount: 0,
    collecting: false,
    avgFPS_20s: 'Idle',

    reset: () => {
        avgStats.startTime = performance.now();
        avgStats.lastFrameTime = avgStats.startTime;
        avgStats.frameCount = 0;
        avgStats.collecting = true;
        avgStats.avgFPS_20s = 'Calculating...';
    }
};

const gui = new GUI();

// =========== Render Mode (top-level) ===========
const renderModes = { forwardPlus: 'forward+', clusteredDeferred: 'clustered deferred' };
let renderModeController = gui.add({ mode: renderModes.forwardPlus }, 'mode', renderModes);

gui.add(avgStats, 'avgFPS_20s').name('Avg FPS (8s)').listen();

// =========== Point Lights ===========
const desiredMobileOptions = [5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1450, 1500];
const desiredPCOptions = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 6000, 7000, 8000];
let desiredOptions = isMobileDevice ? desiredMobileOptions : desiredPCOptions;
const safeOptions = desiredOptions.filter(count => count <= Lights.maxNumLights);
safeOptions.push(desiredOptions[3]);
safeOptions.sort((a, b) => a - b);
// Toggle: saves/restores numLights
lights.numLights = 0;
lights.updateLightSetUniformNumLights();

// =========== Stage ===========
const stage = new Stage(scene, lights, camera, stats, environment);
addHelpersToScene(scene, camera, stage);

var renderer: Renderer | undefined;

function setRenderer(mode: string) {
    renderer?.stop();

    switch (mode) {
        case renderModes.forwardPlus:
            renderer = new ForwardPlusRenderer(stage);
            break;
        case renderModes.clusteredDeferred:
            renderer = new ClusteredDeferredRenderer(stage);
            break;
    }
}

renderModeController.onChange(setRenderer);

// =========== Helper functions (HDR / EXR parsing) ===========
function parseHdrFile(buffer: ArrayBuffer): { rgbaData: Float32Array, width: number, height: number } {
    const parsedLayout = parseHdr(buffer);

    let width: number, height: number;
    let data: Float32Array;

    if (parsedLayout.shape && parsedLayout.data) {
        width = parsedLayout.shape[0];
        height = parsedLayout.shape[1];
        data = parsedLayout.data;
    } else {
        throw new Error("Invalid HDR file format");
    }

    const rgbaData = new Float32Array(width * height * 4);
    if (data.length === width * height * 3) {
        for (let i = 0; i < width * height; i++) {
            rgbaData[i * 4 + 0] = data[i * 3 + 0];
            rgbaData[i * 4 + 1] = data[i * 3 + 1];
            rgbaData[i * 4 + 2] = data[i * 3 + 2];
            rgbaData[i * 4 + 3] = 1.0;
        }
    } else if (data.length === width * height * 4) {
        rgbaData.set(data);
    } else {
        throw new Error(`HDRI data length ${data.length} does not match dimensions ${width}x${height}`);
    }

    return { rgbaData, width, height };
}

function parseExrFile(buffer: ArrayBuffer): { rgbaData: Float32Array, width: number, height: number } {
    const FloatType = 1015;
    const RGBAFormat = 1023;
    const parsed = parseExr(buffer, FloatType);

    const { data, width, height, format } = parsed;
    const numPixels = width * height;

    let rgbaData: Float32Array;

    if (format === RGBAFormat) {
        if (data.length === numPixels * 4) {
            rgbaData = data as Float32Array;
        } else {
            const channels = data.length / numPixels;
            rgbaData = new Float32Array(numPixels * 4);
            if (channels === 3) {
                for (let i = 0; i < numPixels; i++) {
                    rgbaData[i * 4 + 0] = data[i * 3 + 0];
                    rgbaData[i * 4 + 1] = data[i * 3 + 1];
                    rgbaData[i * 4 + 2] = data[i * 3 + 2];
                    rgbaData[i * 4 + 3] = 1.0;
                }
            } else {
                throw new Error(`Unexpected EXR channel count: ${channels}`);
            }
        }
    } else {
        throw new Error(`Unsupported EXR format code: ${format}. Expected RGBA (1023).`);
    }

    const floatsPerRow = width * 4;
    const tempRow = new Float32Array(floatsPerRow);
    for (let y = 0; y < Math.floor(height / 2); y++) {
        const topOffset = y * floatsPerRow;
        const bottomOffset = (height - 1 - y) * floatsPerRow;
        tempRow.set(rgbaData.subarray(topOffset, topOffset + floatsPerRow));
        rgbaData.set(rgbaData.subarray(bottomOffset, bottomOffset + floatsPerRow), topOffset);
        rgbaData.set(tempRow, bottomOffset);
    }

    return { rgbaData, width, height };
}

// =========== Tools ===========
const toolsFolder = gui.addFolder('Tools');

// -- Benchmark --
const benchmarkController = {
    runBenchmark: async () => {
        resultsElement.innerHTML = '--- Benchmark Begin ---<br>';
        resultsElement.style.display = 'block';
        console.log("--- Benchmark Begin ---");
        const allResults: string[] = [];

        for (const lightCount of safeOptions) {
            lights.numLights = lightCount;
            lights.updateLightSetUniformNumLights();

            avgStats.avgFPS_20s = `Idling (${lightCount} lights)...`;
            await sleep(restTime);
            avgStats.reset();

            while (avgStats.collecting) {
                await sleep(200);
            }

            const resultString = `${lightCount} lights: ${avgStats.avgFPS_20s} FPS`;
            allResults.push(resultString);
            console.log(resultString);
            resultsElement.innerHTML += resultString + '<br>';
            resultsElement.scrollTop = resultsElement.scrollHeight;
            await sleep(500);
        }

        avgStats.avgFPS_20s = "Finished!";
        console.log("--- Benchmark End ---");
        console.log(allResults.join('\n'));
        resultsElement.innerHTML += '--- Benchmark End ---<br>';
        resultsElement.scrollTop = resultsElement.scrollHeight;
    }
};
toolsFolder.add(benchmarkController, 'runBenchmark').name('Run Full Benchmark');

// -- HDRI upload --
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.hdr,.exr';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

fileInput.addEventListener('change', async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
        const fileName = file.name.toLowerCase();
        const buffer = await file.arrayBuffer();

        let rgbaData: Float32Array;
        let width: number;
        let height: number;

        if (fileName.endsWith('.hdr')) {
            ({ rgbaData, width, height } = parseHdrFile(buffer));
        } else if (fileName.endsWith('.exr')) {
            ({ rgbaData, width, height } = parseExrFile(buffer));
        } else {
            alert('Unsupported file format. Please upload .hdr or .exr');
            return;
        }

        console.log(`[HDRI] Loaded ${file.name}: ${width}x${height}, data length: ${rgbaData.length}`);
        await stage.environment.loadHDRI(rgbaData, width, height);
    } catch (e) {
        console.error("Failed to load HDRI:", e);
        alert("Failed to load HDRI: " + String(e));
    }
});

const uploadController = {
    uploadHDRI: () => { fileInput.click(); },
    resetHDRI: () => { stage.environment.clearHDRI(); }
};
toolsFolder.add(uploadController, 'uploadHDRI').name('Upload HDRI (.hdr/.exr)');
toolsFolder.add(uploadController, 'resetHDRI').name('Clear HDRI');

// -- Model upload --
const modelFileInput = document.createElement('input');
modelFileInput.type = 'file';
modelFileInput.accept = '.gltf,.glb';
modelFileInput.style.display = 'none';
document.body.appendChild(modelFileInput);

modelFileInput.addEventListener('change', async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
        const fileName = file.name.toLowerCase();
        if (!fileName.endsWith('.gltf') && !fileName.endsWith('.glb')) {
            alert('Unsupported format. Please upload .gltf or .glb');
            return;
        }

        console.log(`[Model] Loading ${file.name}...`);
        const buffer = await file.arrayBuffer();

        const newScene = new Scene();
        const result = await loadGltfBuffer(buffer);
        newScene.root.addChild(result.rootEntity);
        newScene.bvhData = result.bvhData;
        newScene.voxelGrid = result.voxelGrid;
        newScene.voxelGridView = result.voxelGridView;
        newScene.globalMaterialBuffer = result.globalMaterialBuffer;
        newScene.baseColorTexArray = result.baseColorTexArray;
        newScene.baseColorTexArrayView = result.baseColorTexArrayView;
        newScene.root.updateWorldTransform();
        addHelpersToScene(newScene, camera, stage);
        stage.scene = newScene;
        sceneTreeUI.setScene(newScene);

        // Disable random point lights (designed for Sponza) to avoid color artifacts
        lights.numLights = 0;
        lights.updateLightSetUniformNumLights();

        if (renderModeController) {
            setRenderer(renderModeController.getValue());
        }

        console.log(`[Model] Successfully loaded ${file.name}`);
    } catch (e) {
        console.error("Failed to load model:", e);
        alert("Failed to load model: " + String(e));
    }

    modelFileInput.value = '';
});

const modelUploadController = {
    loadModel: () => { modelFileInput.click(); }
};
toolsFolder.add(modelUploadController, 'loadModel').name('Load Model (.gltf/.glb)');


setRenderer(renderModeController.getValue());

const loadingOverlay = document.getElementById('loading-overlay');
if (loadingOverlay) {
    loadingOverlay.classList.add('hidden');
    setTimeout(() => {
        loadingOverlay.remove();
    }, 500);
}
