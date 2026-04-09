import { Scene } from './engine/Scene';
import { Lights } from './stage/lights';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';

import { UniformPool } from './engine/UniformPool';

export var canvas: HTMLCanvasElement;
export var canvasFormat: GPUTextureFormat;
export var context: GPUCanvasContext;
export var device: GPUDevice;
export var canvasTextureView: GPUTextureView;

export var aspectRatio: number;
export const fovYDegrees = 45;

export var modelBindGroupLayout: GPUBindGroupLayout;
export var materialBindGroupLayout: GPUBindGroupLayout;

export var globalUniformPool: UniformPool;

export async function initWebGPU() {
    canvas = document.getElementById("mainCanvas") as HTMLCanvasElement;

    const devicePixelRatio = window.devicePixelRatio;
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;

    aspectRatio = canvas.width / canvas.height;

    if (!navigator.gpu)
    {
        let errorMessageElement = document.createElement("div");
        errorMessageElement.style.padding = '1em';
        let mainMessage = document.createElement("h1");
        mainMessage.textContent = "This browser doesn't support WebGPU!";
        errorMessageElement.appendChild(mainMessage);
        
        const isLinux = navigator.userAgent.toLowerCase().includes("linux");
        const isChrome = navigator.userAgent.toLowerCase().includes("chrome");
        if (isLinux && isChrome) {
            let linuxMessage = document.createElement("p");
            linuxMessage.innerHTML = "It looks like you are using Chrome on Linux. WebGPU is not enabled by default on Linux Chrome.<br>To enable it, please run Chrome with the following flags or enable them in <code>chrome://flags</code>:<br><br><code>--enable-unsafe-webgpu --enable-features=Vulkan</code>";
            errorMessageElement.appendChild(linuxMessage);
        } else {
            let otherMessage = document.createElement("p");
            otherMessage.textContent = "Try using a browser that supports WebGPU, like the latest Google Chrome on Windows/macOS.";
            errorMessageElement.appendChild(otherMessage);
        }

        document.body.innerHTML = '';
        document.body.appendChild(errorMessageElement);
        throw new Error("WebGPU not supported on this browser");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter)
    {
        let errorMessageElement = document.createElement("div");
        errorMessageElement.style.padding = '1em';
        let mainMessage = document.createElement("h1");
        mainMessage.textContent = "WebGPU is enabled, but no appropriate GPUAdapter was found!";
        errorMessageElement.appendChild(mainMessage);
        
        const isLinux = navigator.userAgent.toLowerCase().includes("linux");
        const isChrome = navigator.userAgent.toLowerCase().includes("chrome");
        
        let detailsMessage = document.createElement("p");
        if (isLinux && isChrome) {
            detailsMessage.innerHTML = "On Linux, this usually means Chrome could not initialize the Vulkan backend.<br><br><b>Troubleshooting steps:</b><br>1. Check <code>chrome://gpu</code> to see if Vulkan or WebGPU backend initialization failed.<br>2. Ensure your graphics drivers and Vulkan (e.g., install <code>mesa-vulkan-drivers</code> or run <code>vulkaninfo</code>) are correctly installed.<br>3. Sometimes Chrome's sandbox blocks Vulkan access on Linux. For testing purposes only, you can try launching Chrome with <code>--disable-gpu-sandbox</code>.";
        } else {
            detailsMessage.innerHTML = "This may mean your system's GPU doesn't support WebGPU, or your graphics drivers need to be updated. Check <code>chrome://gpu</code> for more information.";
        }
        errorMessageElement.appendChild(detailsMessage);

        document.body.innerHTML = '';
        document.body.appendChild(errorMessageElement);
        throw new Error("no appropriate GPUAdapter found");
    }

    device = await adapter.requestDevice();

    context = canvas.getContext("webgpu")!;
    canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    console.log("WebGPU init successsful");
    
    // Allocate a generous 16MB for the singleton uniform pool
    globalUniformPool = new UniformPool(device, 16 * 1024 * 1024);

    modelBindGroupLayout = device.createBindGroupLayout({
        label: "model bind group layout",
        entries: [
            { // modelMat
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "uniform" }
            }
        ]
    });

    materialBindGroupLayout = device.createBindGroupLayout({
        label: "material bind group layout",
        entries: [
            { // diffuseTex
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {}
            },
            { // diffuseTexSampler
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {}
            },
            { // PBR params uniform buffer (roughness, metallic, pad, pad, baseColorFactor)
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" }
            },
            { // metallicRoughnessTex
                binding: 3,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {}
            },
            { // metallicRoughnessTexSampler
                binding: 4,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {}
            },
            { // normalTex
                binding: 5,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {}
            },
            { // normalTexSampler
                binding: 6,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {}
            }
        ]
    });
}

export const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 48,
    attributes: [
        { // pos
            format: "float32x3",
            offset: 0,
            shaderLocation: 0
        },
        { // nor
            format: "float32x3",
            offset: 12,
            shaderLocation: 1
        },
        { // uv
            format: "float32x2",
            offset: 24,
            shaderLocation: 2
        },
        { // tangent
            format: "float32x4",
            offset: 32,
            shaderLocation: 3
        }
    ]
};

export abstract class Renderer {
    protected scene: Scene;
    protected lights: Lights;
    protected camera: Camera;
    protected stage: Stage;

    protected stats: Stats;

    private prevTime: number = 0;
    private frameRequestId: number;

    constructor(stage: Stage) {
        this.scene = stage.scene;
        this.lights = stage.lights;
        this.camera = stage.camera;
        this.stage = stage;
        this.stats = stage.stats;

        this.frameRequestId = requestAnimationFrame((t) => this.onFrame(t));
    }

    stop(): void {
        cancelAnimationFrame(this.frameRequestId);
    }

    protected abstract draw(): void;

    private onFrame(time: number) {
        if (this.prevTime == 0) {
            this.prevTime = time;
        }

        let deltaTime = time - this.prevTime;
        this.camera.onFrame(deltaTime);
        this.lights.onFrame(time);

        // Update scene: recompute transforms → upload MeshRenderer GPU buffers
        this.scene.update(deltaTime);

        // Sync pooled dirty regions to the GPU
        globalUniformPool.syncToGPU(device);

        this.stats.begin();

        this.draw();

        this.stats.end();

        this.prevTime = time;
        this.frameRequestId = requestAnimationFrame((t) => this.onFrame(t));
    }
}
