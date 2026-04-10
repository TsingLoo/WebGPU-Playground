# Technical Report

This document serves as the ongoing technical report for the WebGPU Renderer project (Forward+, Clustered Deferred, and Wavefront Path Tracing).

## Codebase Architecture

### 1. `RenderGraph` (`src/engine/RenderGraph.ts`)
The project implements a bespoke Render Graph to efficiently dispatch GPU workloads. By separating logical resource requests from physical allocations, the RenderGraph computes resource lifetimes across topological passes and safely aliases GPU memory for transient textures and buffers (saving VRAM). 
- **Features:** 
  - Implicit load/store operation deduction.
  - Pass dependency tracking via read/write resource handles.
  - Seamless memory aliasing for transient buffers and textures.

### 2. Main Rendering Pipelines

#### Forward+ (`src/renderers/forward_plus.ts`)
Extends `BaseSceneRenderer` to add a shading pass that queries clustered point lights in forward execution. It utilizes a `Z-Prepass` (and partial G-Buffer when post-processing features like SSAO/SSR are enabled) to cull hidden pixels early, improving fill-rate efficiency for the heavy forward shading pass.

#### Clustered Deferred (`src/renderers/clustered_deferred.ts`)
Operates fully on a structured G-Buffer (Albedo, Normal, Position, Specular). Lighting calculations (combining clustered lights and directional shadows) happen as a fullscreen composite compute/fragment pass, decoupling geometry complexity from lighting complexity.

#### Wavefront Path Tracing & ReSTIR (`src/renderers/path_tracing_wavefront.ts`)
A highly optimized compute-based path tracer tracking rays across discrete kernels (RayGen, Intersect, Shade, ShadowTest, Miss). This "wavefront" approach maintains coherent SIMD execution avoiding the divergence common in megakernels.
- **ReSTIR DI:** Implements Reservoir-based Spatiotemporal Importance Resampling on the first bounce to rapidly converge soft shadows and direct illumination from emissive geometry and analytical lights.

### 3. Stage & Globals (`src/stage/`)
Manages rendering entities and data independently of the rendering pipelines themselves:
- `Stage` acts as the director, holding references to the camera, lights, and GI setups.
- `DDGI` (Dynamic Diffuse Global Illumination): Manages a 3D grid of light probes that iteratively trace rays against the scene's BVH. The traced results are temporally blended into octahedral-mapped irradiance and visibility atlases (using a memory-saving ping-pong approach). It also features a relocation pass to prevent probes from becoming stuck inside solid geometry.
- `Radiance Cascades`: Uses cascaded voxel grids for fast ambient light evaluation.
- `VSM` (Virtual Shadow Maps): Implements a clipmap-based virtual shadow mapping system for the directional sun light. Instead of rendering a massive single shadow map, it uses GPU-driven page allocation. A compute pass marks which virtual pages are visible based on the depth buffer, and the system dynamically assigns physical atlas pages to fulfill these requests, supporting incredibly high-resolution shadows over large distances.
- `BVH Builder` rapidly constructs acceleration structures from the GLTF scenes allowing compute shaders to perform fast ray intersections.
  - **Optimization Features**: The BVH pipeline forces a strict `maxLeafSize` of 4 during CPU construction to enable static branchless unrolling of triangle intersection loops in WGSL. Traversal operates on a highly constrained depth-24 short stack mapped explicitly to private thread scopes to avoid register thrashing. Intersection paths share an expanded `Ray` data structure that caches IEEE 754-safe inverse direction checks alongside FMA-fused (Fused Multiply-Add) AABB bounds testing to slash shader MS-per-frame execution time.

### 4. Scene Management (`src/engine/Scene.ts`, `Entity.ts`)
Standard Entity-Component hierarchy loading from `GLTF` formats in `main.ts`. Transforms are synced and flattened to a global material/mesh buffer consumed by either WebGPU raster pipelines or BVH compute traces.

## Project Rules & Documentation Sync

1. **Architecture Sync**: Any modifications to the rendering pipelines, shaders, RenderGraph, or architecture MUST be reflected directly in this file and the `structure_and_flowchart.md`.
2. **Mermaid Flowchart Syntax Requirement**: To guarantee 100% compatibility with the built-in VS Code markdown previewer, all Mermaid charts must strictly adhere to the following baseline syntax:
   - **No Subgraph Titles**: Use the format `subgraph PassExecution` (alphanumeric only). Heavily avoid the `subgraph ID [Title]` formatting.
   - **No Edge Labels**: Avoid `-->|Text|` labels as they crash older parsers when spaces or special characters are present. Use markdown text outside the chart to explain conditional routing.
   - **Standard Nodes Only**: Enforce the use of standard square brackets `[Text Node]` for all nodes. Do not use quotes `["Text"]`, diamond selectors `{Text}`, or database cylinders `[(Text)]`.
   - **No Special Characters**: Exclude characters such as `<`, `>`, `&`, `/`, `(`, `)`, and `"` inside node definitions.
   - **No Indentation**: Do not indent the node entries inside subgraph blocks.
