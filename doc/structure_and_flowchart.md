# Structure & Flowchart

This document details the code architecture, module interactions, and the rendering pipelines' structure for the WebGPU renderer project.

## Application Entry & Setup (`main.ts`)

The application starts in `main.ts`, which sets up the rendering environment, parses UI selections, and glues the systems together.

```mermaid
graph TD
Main[main ts] --> InitGPU[Initialize WebGPU Device]
Main --> LoadModel[Load GLTF Scene and BVH]
Main --> StageSetup[Create Stage Camera and Lights]
Main --> UI[Initialize GUI and Stats]
Main --> SetRenderer[Select Renderer]
SetRenderer --> FP[ForwardPlusRenderer]
SetRenderer --> CD[ClusteredDeferredRenderer]
SetRenderer --> WPT[WavefrontPathTracingRenderer]
```

## RenderGraph Execution Flow

All base renderers use a bespoke `RenderGraph` (`src/engine/RenderGraph.ts`) to orchestrate passes. This graph handles topological sorting, dependency tracking, and automatic resource memory aliasing.

```mermaid
graph TD
Draw[Renderer Draw] --> RGUpdate[Declare Virtual Resources and Passes]
RGUpdate --> RGCompile[RenderGraph compile]

subgraph RenderGraphAnalysis
RGCompile --> TopoSort[Topological Sort of Passes]
TopoSort --> Lifetime[Calculate Resource Lifetimes]
Lifetime --> Aliasing[Memory Aliasing]
end

RGCompile --> RGExecute[Execute Passes]

subgraph PassExecution
RGExecute --> StageUpdates[Stage Data Update]
StageUpdates --> Prepass[Z Prepass]
Prepass --> HiZ[Hi Z Depth Generation]
HiZ --> ShadowMap[VSM Shadow Map]
ShadowMap --> GBuffer[G Buffer Pass]
GBuffer --> GI[GI DDGI Cascades update]
GI --> LightCluster[Light Clustering]
LightCluster --> SSAO[SSAO Generation]
SSAO --> Shading[Shading]
Shading --> SkyboxDebug[Skybox and Debug Render]
SkyboxDebug --> PostProc[Final Blit]
PostProc --> Volumetric[Volumetric Fog]
end
```

## Wavefront Path Tracing Data Flow

The `WavefrontPathTracingRenderer` executes a distinct wavefront bounce loop, utilizing state-of-the-art compute shaders and ReSTIR for Direct Illumination (ReSTIR DI updates only occur on Bounce 0).

```mermaid
graph TD
PT[Start Frame] --> Init[Clear Accum Update NRC]
Init --> RayGen[Pass 0 Ray Gen]

subgraph BounceLoop
RayGen --> Intersect[Pass 1 BVH Intersect]
Intersect --> Shade[Pass 2 Shade and Material Eval]
Shade --> ReSTIR[ReSTIR DI Spatial Temporal]
ReSTIR --> ShadowTest[Pass 3 Shadow Test]
Shade --> Miss[Pass 4 Env Map Miss]
end

ShadowTest --> Accumulate[Accumulate to Persistent Buffer]
Miss --> Accumulate
Accumulate --> Tonemap[Tonemap Renderer Blit to Canvas]
```

## Virtual Shadow Maps (VSM)

The Virtual Shadow Map module (`src/stage/vsm.ts`) implements a clipmap-based virtualized shadow mapping algorithm. It uses GPU-driven allocation to render high-resolution shadows over vast distances without exhausting VRAM.

```mermaid
graph TD
subgraph VSM
VSMClear[Clear Page Request and State] --> VSMMark[Mark Pages via View Depth]
VSMMark --> VSMAlloc[Allocate Physical Pages]
VSMAlloc --> VSMRender[Render Cast Shadows]
VSMRender --> VSMAtlas[Physical Depth Atlas]
end
```

## Dynamic Diffuse Global Illumination (DDGI)

The DDGI module (`src/stage/ddgi.ts`) utilizes a grid of light probes spread across the scene. These probes trace rays into the BVH to gather lighting, which is temporally blended into irradiance and visibility atlases.

```mermaid
graph TD
subgraph DDGI
DDGI_Trace[Probe Trace] --> DDGI_Relocate[Probe Relocate]
DDGI_Relocate --> DDGI_UpdateIrr[Irradiance Update]
DDGI_Relocate --> DDGI_UpdateVis[Visibility Update]
DDGI_UpdateIrr --> DDGI_PingPong[Atlas Ping Pong]
DDGI_UpdateVis --> DDGI_PingPong
end
```
