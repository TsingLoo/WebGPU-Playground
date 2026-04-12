WebGPU Render
==============================================

![](./img/chrome_Bm0daornYa.jpg)

## [Live Demo](https://tsingloo.github.io/WebGPU-Playground/)

* **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Tested on: 

  | Component\Platform | PC                                              | Mobile                                                       |
  | ------------------ | ----------------------------------------------- | ------------------------------------------------------------ |
  | OS                 | Google Chrome(141.0.7390.108) @ Windows 11 24H2 | Google Chrome(141.0.7390.111) @ Android 14                   |
  | CPU/SoC            | Intel 13600KF @ 3.5Ghz                          | [MediaTek Dimensity 8100](https://www.mediatek.com/products/smartphones/mediatek-dimensity-8100) |
  | GPU                | 4070 SUPER 12GB                                 | Arm Mali-G610 MC6                                            |
  | RAM                | 32GB RAM                                        | 12GB RAM                                                     |
  | Model              |                                                 | Redmi K50                                                    |



### Demo Video/GIF

https://github.com/user-attachments/assets/7da29776-83ef-4660-9702-f51ed1dad99b

# Features

## Render Graph Architecture

A custom Render Graph (Frame Graph) is implemented for automatic resource management, pass scheduling, and intelligent memory aliasing.

- **Dependency Tracking**: Tracks resource reads and writes across rendering/compute passes to build a Directed Acyclic Graph (DAG) and calculate topological sorting.
- **Resource Aliasing**: Allocates internal physical textures and buffers efficiently based on resource lifespans using interval scheduling, reducing peak memory load.
- **Intelligent Ops**: Automatically infers WebGPU `clear` and `discard` states based on the first/last usage index of each resource, saving memory bandwidth.
- **Caching**: Hashes pass topology to avoid recompilation costs for static frames.

```mermaid
graph TB
    subgraph PublicAPI["Public API (Called per-frame)"]
        A1[createTexture / importTexture]
        A2[createBuffer / importBuffer]
        A3[addRenderPass / addComputePass]
        A6[execute encoder]
    end

    subgraph CompilePipeline["Compile Pipeline (Topology Hash Miss)"]
        B1["(1) Topology Hash Check"]
        B2["(2) Dependency Analysis"]
        B3["(3) Root Node Recognition"]
        B4["(4) BFS Reverse Traversal"]
        B5["(5) Kahn Topological Sorting"]
        B6["(6) Lifetime & Usage Aggregation"]
        B7["(7) Memory Aliasing (Interval Scheduling)"]
        B9["(8) Precompile aliasBinds"]
        B10["(9) Infer Load/Store Operations"]
    end

    subgraph ExecutePipeline["Execution Pipeline (Per-frame)"]
        C1[Traverse Active Passes in Order]
        C2[Allocate Physical Resources]
        C3[Bind Alias Handles to Root]
        C4["Execute Passes (Render/Compute)"]
        C5[Release Resources to Pool]
        C6[LRU Pool Cleanup]
    end

    A6 --> B1
    B1 -->|Cache Miss| B2
    B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B9 --> B10
    B1 -->|Cache Hit| C1
    B10 --> C1
    C1 --> C2 --> C3 --> C4 --> C5 --> C6
```

### Single Frame Execution Sequence

This sequence highlights what happens dynamically per-frame, showing where the caching mechanism bridges logical pass declarations with physical GPU allocations and callback execution.

```mermaid
sequenceDiagram
    participant Caller as Renderer / Pipeline
    participant RG as RenderGraph
    participant PB as PassBuilder
    participant CP as compile()
    participant Pool as Physical Resource Pool
    participant GPU as WebGPU Device

    Note over Caller,GPU: ── Frame Start ──

    Caller->>RG: createTexture() / createBuffer()
    RG-->>Caller: Logical Handle

    Caller->>RG: addRenderPass() / addComputePass()
    RG->>PB: new Builder(passData)
    Caller->>PB: .readTexture() / .writeBuffer() / .execute(fn)

    Note over Caller,GPU: ── execute(encoder) ──

    Caller->>RG: execute(commandEncoder)
    
    RG->>CP: compile()
    Note right of CP: Topology Hash Check<br>Kahn Topological Sort<br>Alias Scheduling<br>Load/Store Inference
    CP-->>RG: CompiledPlan (O(1) if Cached)

    loop For each activePass (Topological Order)
        RG->>Pool: Allocate from pool or create Physical Resource
        Pool-->>RG: PhysicalTexture / PhysicalBuffer
        RG->>RG: Bind Alias Handle -> Root Physical Resource

        alt Render Pass
            RG->>GPU: encoder.beginRenderPass()
            RG->>Caller: executeFn(renderPassEnc, resolver)
            Caller->>RG: resolver.getTextureView(handle)
            RG-->>Caller: GPUTextureView
            RG->>GPU: renderPassEnc.end()
        else Compute Pass
            RG->>GPU: encoder.beginComputePass()
            RG->>Caller: executeFn(computePassEnc, resolver)
            RG->>GPU: computePassEnc.end()
        end

        RG->>Pool: Return unused physical resources to pool (deallocations)
    end

    RG->>Pool: LRU Scan (destroy if framesIdle > 5)
    RG->>RG: Reset passes and handles for next frame

    Note over Caller,GPU: ── Frame End ──
```

### Resource Lifetime & Memory Aliasing

To drastically reduce VRAM usage, physical resources from the pool are aliased dynamically. Logical resource handles that do not overlap in lifetime will automatically share the exact same physical WebGPU resource in sequential order, preventing unnecessary allocations pipeline-wide.

```mermaid
graph LR
    subgraph Logical["Logical Layer (Declared per-frame)"]
        L1["Handle = 1<br>GBuffer_albedo"]
        L2["Handle = 2<br>GBuffer_normal"]
        L3["Handle = 3<br>SSAO_Result"]
    end

    subgraph Physical["Physical Layer (Pooled across frames)"]
        P1["PhysicalTexture A<br>rgba8unorm 1920x1080"]
        P2["PhysicalTexture B<br>rgba8unorm 1920x1080"]
    end

    subgraph Timeline["Pass Execution Order & Aliasing Interval"]
        T1["Pass 0: GBuffer<br>Writes H1, H2"]
        T2["Pass 1: SSAO<br>Reads H1, H2<br>Writes H3"]
        T3["Pass 2: Lighting<br>Reads H3, H1"]
    end

    L1 -->|"firstUsage=Pass0<br>Allocate P1"| P1
    L2 -->|"firstUsage=Pass0<br>Allocate P2"| P2
    L3 -->|"firstUsage=Pass1<br>H2 lifetime ended → Alias H3 to P2"| P2
    
    T1 --> T2 --> T3
```

---

## Rendering Pipelines

### Forward+

The Forward+ pipeline performs clustered light culling before the main shading pass. It consists of three stages: Z-prepass, light culling, and the shading pass. This approach retains the simplicity of a single forward pass while supporting efficient per-fragment lighting from many sources. Compared to classic deferred, it naturally handles transparency and flexible material models.

### Clustered Deferred (Compute Pass)

The deferred path writes albedo, position, and normal into G-buffers, then performs light culling and shading in a subsequent pass. In this implementation, the traditional fragment-based lighting stage is replaced by a compute shader that dispatches one thread per screen pixel, combining the vertex and fragment stages into a single programmable step. The full pipeline is: Z-prepass, G-buffer pass, light culling, compute shading pass, and final blit.

---

## Z-Prepass

An early Z-prepass fills the depth buffer before any shading work begins. A simple vertex shader transforms geometry and a minimal fragment shader discards transparent fragments by alpha testing. The main shading pass then uses `depthCompare: equal`, so only the front-most fragments get shaded. This is particularly useful in scenes like Sponza where overlapping geometry would otherwise result in significant overdraw.

---

## Clustered Light Culling

The view frustum is divided into a 3D grid of clusters. A compute shader determines which lights overlap each cluster, and during shading each fragment retrieves the relevant light list by its cluster index.

### Single Global Buffer

Instead of per-cluster light lists, a single global buffer stores all light indices. Each cluster records an offset and count into this buffer. This is more memory-efficient, but the approach is bounded by WebGPU's maximum storage buffer binding size (134,217,728 bytes). At high grid resolutions with thousands of lights, the buffer can overflow, causing visible tiled artifacts or missing lights.

![scene with 6k lights](./img/tiledLooking.png)

### Logarithmic Z-Slicing

The Z-axis is sliced logarithmically rather than linearly in view space, which distributes clusters more evenly across the depth range and improves culling precision near the camera. Based on [A Primer On Efficient Rendering Algorithms & Clustered Shading](https://www.aortiz.me/2018/12/21/CG.html).

![log slice](https://www.aortiz.me/slides/ZSlices/zs2.png)

---

## PBR Shading (Cook-Torrance)

All lighting uses a physically-based Cook-Torrance BRDF with GGX normal distribution, Smith geometry, and Fresnel-Schlick approximation. The material pipeline reads glTF PBR parameters (base color, metallic, roughness) from textures and uniforms, and supports tangent-space normal mapping with Gram-Schmidt re-orthogonalization. A Reinhard tone mapper and gamma correction are applied as the final step.

---

## Image-Based Lighting (IBL)

Full split-sum IBL is computed entirely on the GPU at startup:

- **Environment Cubemap** -- A procedural sky cubemap (256x256 per face) is generated by a compute shader at initialization. Users can also upload custom `.hdr` or `.exr` environment maps, which are converted from equirectangular to cubemap via another compute pass.
- **Diffuse Irradiance** -- The environment cubemap is convolved into a 32x32 irradiance cubemap for diffuse ambient.
- **Prefiltered Specular** -- A roughness-stratified prefiltered cubemap (128x128, 5 mip levels, 1024 importance samples per mip) is generated for specular IBL.
- **BRDF LUT** -- A 256x256 LUT is precomputed for the split-sum integral's scale and bias terms.

The skybox is rendered as a fullscreen pass with reverse depth (`less-equal`), so it appears behind all geometry.

---

## Probe-Based Global Illumination (DDGI-style)

A probe-based diffuse GI system inspired by DDGI. The overall framework -- probe grid, irradiance/visibility octahedral atlases, hysteresis blending, Chebyshev visibility weighting -- follows the standard DDGI pipeline. Since WebGPU does not currently expose hardware ray tracing, the probe trace stage utilizes a **software ray tracer**. Rays are traced directly against a **Bounding Volume Hierarchy (BVH)** built from the scene's triangle geometry, rather than using RT cores.

The per-frame compute pipeline consists of:

1. **Probe Trace (Software Ray Tracing)** -- Each probe fires rays through the scene's BVH. On intersection with geometry, the hit surface's normal is evaluated for direct sun lighting (with VSM shadow lookup). Missed rays sample the environment cubemap.
2. **Irradiance Update** -- Ray radiance results are blended into the irradiance atlas using exponential hysteresis, with octahedral encoding per probe.
3. **Visibility Update** -- Mean distance and squared distance are accumulated for Chebyshev-based visibility testing.
4. **Ping-Pong Atlases** -- Double-buffered atlas textures prevent read-write hazards across frames.

During shading, each fragment performs trilinear interpolation across the 8 surrounding probes, weighted by Chebyshev visibility to suppress light leaking.

```mermaid
graph TD
    A[Probe Grid] --> B(Probe Trace Pass)
    B -->|Software Ray Tracing| C[BVH Intersection]
    C -->|Hit| D(Evaluate Direct Lighting & VSM)
    C -->|Miss| E(Sample Skybox)
    D --> F(Irradiance Update Pass)
    E --> F
    D --> G(Visibility Update Pass)
    F -->|Octahedral Encode| H[(Irradiance Atlas)]
    G -->|Chebyshev Moments| I[(Visibility Atlas)]
    H --> J(Shading Pass)
    I --> J
    J -->|Trilinear Interpolate| K[Final Indirect Diffuse]
```


## Virtual Shadow Maps (VSM)

Shadows from the directional (sun) light use a clipmap-based Virtual Shadow Map system. The pipeline runs entirely on the GPU:

1. **Clear Pass** -- Resets page request flags and allocation state.
2. **Mark Pass** -- A compute shader reprojects each screen pixel into light space to determine which virtual shadow pages are needed.
3. **Allocate Pass** -- Requested pages are assigned physical slots from an atlas pool.
4. **Render Pass** -- Scene geometry is rasterized into the physical atlas with per-level orthographic projections. Each clipmap level covers a progressively larger region (16 to 512 world units), centered on the camera with sub-texel snapping to eliminate shadow swimming.

The shadow lookup in the fragment shader walks the page table to find the physical atlas tile for the current fragment's light-space position. The default configuration is a 4096x4096 physical atlas with 128-texel pages and 6 clipmap levels.

```mermaid
graph TD
    A[Camera View] --> B(Mark Pass)
    B -->|Project to Light Space| C[Determine Required Pages]
    C --> D(Allocate Pass)
    D -->|Assign Physical Slots| E[(Physical Texture Atlas)]
    E --> F(Render Pass)
    F -->|Rasterize Geometry| E
    E --> G(Shading Pass)
    G -->|Lookup Page Table| H[Final Shadow Mask]
```

---

## Screen-Space Ambient Occlusion (SSAO)

A screen-space AO pass samples the G-buffer depth and normals to estimate local occlusion. The raw AO result is blurred with a box filter to reduce noise, then multiplied into the ambient term of the final shading. Radius, bias, and power are adjustable at runtime.

```mermaid
graph TD
    A[(G-Buffer Depth)] --> C(SSAO Pass)
    B[(G-Buffer Normal)] --> C
    C -->|Sample Hemisphere| D[Raw Ambient Occlusion]
    D --> E(Blur Pass)
    E -->|Box Filter| F[Smoothed AO]
    F --> G(Lighting Composite)
    G -->|Multiply Ambient Term| H[Final Shaded Fragment]
```

---


## Spectral Path Tracing & Dispersion

A spectral path tracing mode is implemented using the hero-wavelength approach (inspired by pbrt v4).
- **CIE Color Matching**: Uses Wyman Gaussian for accurate spectrum-to-RGB conversion, along with Smits RGB-to-Spectrum logic.
- **Dispersion**: Supports Cauchy's equation for modeling wavelength-dependent index of refraction (IOR), producing realistic chromatic aberration and prism effects.
- **Throughput & Light Transport**: Traces 4 hero wavelengths per pixel simultaneously to efficiently estimate full spectral transport.
- **Toggleable**: Users can seamlessly turn this on or off via the GUI.

### Spectral Rendering Comparison

![Spectral Rendering On](./img/chrome_WfIF9BLH1u.jpg)

---

# Performance Analysis

## Benchmark Setup

All test results are obtained with the cluster dimensions configured as **x = 16, y = 16, z = 16**, and a **maximum of 1024 lights per cluster**, rendering at a fixed resolution of **1080p**.

A benchmark script automatically varies the number of active lights in the scene and measures performance by counting how many frames are rendered within a given time period.

## Mobile & PC 

![](./img/benchmark_comparison_plot.svg)

The PC benchmark results performed as expected, initially demonstrating the clear advantage of powerful desktop hardware and high memory bandwidth. In the early stage, both Forward+ and Deferred rendering were so efficient they were simply limited by the 180 FPS cap. However, the distinction emerged as Forward+ began to degrade first at 2100 lights, indicating it hit a compute bottleneck as its fragment shader complexity scaled with the light count. Deferred rendering sustained peak performance until 2700 lights. 

Contrary to the common assumption that **deferred rendering** is problematic for mobile platforms due to bandwidth constraints, the benchmark results indicate **a performance advantage over Forward+ rendering in scenarios with high light counts**. While the deferred approach exhibited a minor initial overhead with fewer than 100 lights, its performance scaled more effectively as the scene complexity increased.

## Deferred Rendering by Single Compute Pass (Extra Credit)

![](./img/deferred_comparison_frametime.svg)

Both deferred rendering methods exhibit identical performance at lower light counts and when the light count exceeds 2,400, a consistent performance gap appears. The **Compute Deferred approach maintains a lower frame time compared to the Traditional Deferred** method, and this gap widens as the number of lights scales up. 

The performance advantage comes from the compute shader's ability to skip the GPU's fixed-function rasterization stage and run directly on compute units. Instead of running per-pixel like fragment shaders, compute shaders work in small groups allowing threads to share and reuse lighting data efficiently using fast shared memory. This reduces unnecessary global memory access and lowers overall bandwidth usage.

# Debug Images 

![](./img/chrome_THoVq9rDiw.png)

![](./img/chrome_NtVdUQ8qm5.png)

# Credits

- [A Primer On Efficient Rendering Algorithms & Clustered Shading.](https://www.aortiz.me/2018/12/21/CG.html)
- [Vite](https://vitejs.dev/)
- [loaders.gl](https://loaders.gl/)
- [dat.GUI](https://github.com/dataarts/dat.gui)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
