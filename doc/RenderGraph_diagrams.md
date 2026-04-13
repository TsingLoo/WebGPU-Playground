# RenderGraph — 架构与时序图

> 源文件: `src/engine/RenderGraph.ts`（879 行）

---

## 1. 整体架构（类图）

```mermaid
classDiagram
    direction TB

    class RenderGraph {
        -nextHandle: number
        -textureDescs: Map~Handle, TextureDescriptor~
        -bufferDescs: Map~Handle, BufferDescriptor~
        -resourceNames: Map~Handle, string~
        -importedTextureViews: Map~Handle, GPUTextureView~
        -importedBuffers: Map~Handle, GPUBuffer~
        -passes: RenderPassData[]
        -cachedPlan: CompiledPlan | null
        -handleToPhysTex: Map~Handle, PhysicalTexture~
        -handleToPhysBuf: Map~Handle, PhysicalBuffer~
        -poolTextures: Map~string, PhysicalTexture[]~
        -poolBuffers: Map~string, PhysicalBuffer[]~
        +createTexture(name, desc) ResourceHandle
        +importTexture(name, view) ResourceHandle
        +createBuffer(name, desc) ResourceHandle
        +importBuffer(name, buffer) ResourceHandle
        +addRenderPass(name) RenderPassBuilder
        +addComputePass(name) ComputePassBuilder
        +addGenericPass(name) GenericPassBuilder
        +getTextureView(handle, range?) GPUTextureView
        +getBuffer(handle) GPUBuffer
        +execute(encoder)
        +clearPhysicalPool()
        +getMermaidGraph() string
        -compile() CompiledPlan
        -buildTopologyHash() string
        -getPhysicalTexture(desc, name) PhysicalTexture
        -getPhysicalBuffer(desc, name) PhysicalBuffer
        -releasePhysicalTexture(handle, phys)
        -releasePhysicalBuffer(handle, phys)
        -resolveDimensions(desc)
    }

    class PassResolver {
        <<interface>>
        +getTextureView(handle, range?) GPUTextureView
        +getBuffer(handle) GPUBuffer
    }

    class PassBuilder {
        #passData: RenderPassData
        +markRoot() this
        +readTexture(handle, usage?, range?) this
        +writeTexture(handle, usage?, range?) this
        +readBuffer(handle, usage?) this
        +writeBuffer(handle, usage?) this
    }

    class RenderPassBuilder {
        +addColorAttachment(handle, options?) this
        +setDepthStencilAttachment(handle, options?) this
        +execute(fn)
    }

    class ComputePassBuilder {
        +execute(fn)
    }

    class GenericPassBuilder {
        +execute(fn)
    }

    class RenderPassData {
        +name: string
        +type: PassType
        +reads: ResourceAccess[]
        +writes: ResourceAccess[]
        +isRoot: boolean
        +executeFn: Function
        +colorAttachments: Map
        +depthStencilAttachment?
    }

    class CompiledPlan {
        +hash: string
        +activePassIndices: number[]
        +allocations: ResourceHandle[][]
        +deallocations: ResourceHandle[][]
        +textureUsages: Map
        +bufferUsages: Map
        +renderPassLoadStoreOps: Map
        +textureAliases: Map
        +bufferAliases: Map
        +aliasBinds: Array
        +mermaidString: string
    }

    class PhysicalTexture {
        +texture: GPUTexture
        +view: GPUTextureView
        +desc: TextureDescriptor
        +width: number
        +height: number
        +framesIdle: number
        +subViews: Map~number, GPUTextureView~
    }

    class PhysicalBuffer {
        +buffer: GPUBuffer
        +desc: BufferDescriptor
        +framesIdle: number
    }

    class PassType {
        <<enumeration>>
        Render
        Compute
        Generic
    }

    RenderGraph ..|> PassResolver
    PassBuilder <|-- RenderPassBuilder
    PassBuilder <|-- ComputePassBuilder
    PassBuilder <|-- GenericPassBuilder
    PassBuilder o-- RenderPassData
    RenderGraph o-- RenderPassData
    RenderGraph o-- CompiledPlan
    RenderGraph o-- PhysicalTexture
    RenderGraph o-- PhysicalBuffer
    RenderPassData --> PassType
```

---

## 2. 模块层次结构

```mermaid
graph TB
    subgraph Public_API["公共 API（每帧调用）"]
        A1[createTexture / importTexture]
        A2[createBuffer / importBuffer]
        A3[addRenderPass → RenderPassBuilder]
        A4[addComputePass → ComputePassBuilder]
        A5[addGenericPass → GenericPassBuilder]
        A6[execute encoder]
    end

    subgraph Compile_Pipeline["编译管线（拓扑变化时重编译）"]
        B1["① 拓扑哈希检查\nbuildTopologyHash()"]
        B2["② 依赖分析\nlastWriter / lastReaders"]
        B3["③ 根节点识别\nisRoot / writes-to-imported"]
        B4["④ BFS 逆向遍历\nvisitedRequired"]
        B5["⑤ Kahn 拓扑排序\n保持声明顺序优先级"]
        B6["⑥ 生命期 & Usage 聚合\nfirstUsageIdx / lastUsageIdx"]
        B7["⑦ 纹理内存别名\n区间调度算法"]
        B8["⑧ Buffer 内存别名\n区间调度算法"]
        B9["⑨ aliasBinds 预编译"]
        B10["⑩ load/store 操作推断"]
        B11["⑪ Mermaid 图生成"]
    end

    subgraph Execute_Pipeline["执行管线（每帧）"]
        C1[更新 desc.usage 为聚合值]
        C2[按序遍历 activePassIndices]
        C3["分配物理资源\ngetPhysicalTexture / Buffer"]
        C4[绑定别名句柄到根物理资源]
        C5["执行 pass\nRender / Compute / Generic"]
        C6["释放资源回池\nreleasePhysical*"]
        C7["LRU 清理\n>5帧 idle → destroy"]
        C8[重置逻辑状态 passes/descs/handles]
    end

    subgraph Resource_Pool["物理资源池（帧间复用）"]
        D1["poolTextures\nMap< key, PhysicalTexture[] >"]
        D2["poolBuffers\nMap< key, PhysicalBuffer[] >"]
    end

    A6 --> B1
    B1 -->|缓存命中| C1
    B1 -->|缓存失效| B2
    B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8 --> B9 --> B10 --> B11
    B11 --> C1
    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8
    C3 <-->|复用 / 归还| D1
    C3 <-->|复用 / 归还| D2
    C6 --> D1
    C6 --> D2
```

---

## 3. 帧执行时序图

```mermaid
sequenceDiagram
    participant Caller as 调用者（Renderer）
    participant RG as RenderGraph
    participant PB as PassBuilder
    participant CP as compile()
    participant Pool as 物理资源池
    participant GPU as WebGPU Device

    Note over Caller,GPU: ── 每帧开始 ──

    Caller->>RG: createTexture("GBuffer_albedo", desc)
    RG-->>Caller: handle = 1

    Caller->>RG: createBuffer("LightList", desc)
    RG-->>Caller: handle = 2

    Caller->>RG: importTexture("SwapChain", swapChainView)
    RG-->>Caller: handle = 3

    Caller->>RG: addRenderPass("GBuffer Pass")
    RG->>PB: new RenderPassBuilder(passData)
    RG-->>Caller: builder

    Caller->>PB: .addColorAttachment(handle=1)
    Caller->>PB: .writeBuffer(handle=2)
    Caller->>PB: .execute(fn)

    Caller->>RG: addComputePass("Light Culling")
    Caller->>PB: .readBuffer(handle=2)
    Caller->>PB: .markRoot()
    Caller->>PB: .execute(fn)

    Caller->>RG: addRenderPass("Lighting Pass")
    Caller->>PB: .readTexture(handle=1)
    Caller->>PB: .addColorAttachment(handle=3)
    Caller->>PB: .execute(fn)

    Note over Caller,GPU: ── execute(encoder) ──

    Caller->>RG: execute(commandEncoder)

    RG->>CP: compile()

    CP->>CP: buildTopologyHash() → 检查缓存
    CP->>CP: 依赖分析（read → lastWriter 边）
    CP->>CP: 识别根节点（markRoot / writes-imported）
    CP->>CP: BFS 逆向 → visitedRequired
    CP->>CP: Kahn 拓扑排序（priority = 声明顺序）
    CP->>CP: 聚合 textureUsages / bufferUsages
    CP->>CP: 纹理别名区间调度
    CP->>CP: Buffer 别名区间调度
    CP->>CP: 预编译 aliasBinds[]
    CP->>CP: 推断 load/store ops（首次=clear，末次非imported=discard）
    CP->>CP: 生成 mermaidString
    CP-->>RG: CompiledPlan（已缓存）

    loop 每个 activePass（按拓扑序）
        RG->>Pool: 从池中取 / 新建物理资源（allocations[i]）
        Pool-->>RG: PhysicalTexture / PhysicalBuffer
        RG->>RG: 绑定别名句柄 → 根物理资源（aliasBinds[i]）

        alt PassType.Render
            RG->>GPU: encoder.beginRenderPass({ colorAttachments, depthStencil })
            GPU-->>RG: GPURenderPassEncoder
            RG->>Caller: executeFn(renderPassEnc, resolver)
            Caller->>RG: getTextureView(handle)
            RG-->>Caller: GPUTextureView
            RG->>GPU: renderPassEnc.end()
        else PassType.Compute
            RG->>GPU: encoder.beginComputePass()
            GPU-->>RG: GPUComputePassEncoder
            RG->>Caller: executeFn(computePassEnc, resolver)
            RG->>GPU: computePassEnc.end()
        else PassType.Generic
            RG->>Caller: executeFn(encoder, resolver)
        end

        RG->>Pool: 归还物理资源（deallocations[i]）
    end

    RG->>Pool: LRU 扫描（framesIdle++ > 5 → destroy）
    RG->>RG: 重置 passes / descs / handles（为下帧准备）

    Note over Caller,GPU: ── 帧结束 ──
```

---

## 4. 资源生命期与内存别名

```mermaid
graph LR
    subgraph Logical["逻辑层（每帧声明，帧末清除）"]
        L1["Handle = 1\nGBuffer_albedo\nrgba8unorm"]
        L2["Handle = 2\nGBuffer_normal\nrgba8unorm"]
        L3["Handle = 3\nDepth\ndepth32float"]
    end

    subgraph Physical["物理层（跨帧池化）"]
        P1["PhysicalTexture A\nrgba8unorm 1920×1080"]
        P2["PhysicalTexture B\ndepth32float 1920×1080"]
    end

    subgraph Pool["资源池"]
        PL1["poolTextures\n'rgba8unorm_1920_1080_usage'"]
        PL2["poolBuffers\n'size_usage'"]
    end

    subgraph Timeline["Pass 执行顺序"]
        T1["Pass 0: GBuffer\n写 H1, H2, H3"]
        T2["Pass 1: SSAO\n读 H1, H2\n写 H4(别名→H1)"]
        T3["Pass 2: Lighting\n读 H4(→P1), H3"]
    end

    L1 -->|"firstUsage=Pass0\n从池取或新建"| P1
    L2 -->|"lastUsage=Pass1\n别名区间调度\n与H1不重叠→共享P1"| P1
    L3 --> P2

    P1 -->|"lastUsage结束\n归还池"| PL1
    P2 -->|"lastUsage结束\n归还池"| PL1

    T1 --> T2 --> T3
```

**别名规则（区间调度）：**

```
资源 H1:  [Pass0 ─────────── Pass1]
资源 H2:  [Pass0 ─── Pass0]          ← 生命期早于 H1 结束前结束，无法复用
资源 H4:              [Pass1 ── Pass2] ← H1 结束后开始 → 分配别名到 H1 的物理资源
```

---

## 5. 编译管线数据流（compile()）

```mermaid
flowchart TD
    A([开始 compile]) --> B{拓扑哈希\n命中缓存?}
    B -- 是 --> Z([返回 cachedPlan])
    B -- 否 --> C

    C["① 依赖分析\n遍历所有 pass 的 reads/writes\n建立 lastWriter / lastReaders 映射\n构造 deps: Pass→Set<Pass>"]

    C --> D["② 根节点识别\n• isRoot = true 的 pass\n• writes 到已导入 texture/buffer 的 pass"]

    D --> E["③ BFS 逆向遍历\n从根出发沿 deps 反向\n收集 visitedRequired"]

    E --> F["④ Kahn 拓扑排序\n计算入度 indegree\n按原声明顺序优先级入队\n输出 activePasses[]"]

    F --> G{检测环?}
    G -- activePasses ≠ visitedRequired --> ERR([抛出循环依赖错误])
    G -- 正常 --> H

    H["⑤ 生命期 & Usage 聚合\n• firstUsageIdx[handle] = 首次出现的 pass 序号\n• lastUsageIdx[handle] = 末次出现的 pass 序号\n• textureUsages / bufferUsages 按位 OR 累积"]

    H --> I["⑥ 纹理别名区间调度\n按格式/尺寸/usage 分组\n相同组内按 firstUsage 排序\n非重叠区间复用同一 root handle"]

    I --> J["⑦ Buffer 别名区间调度\n按 size/usage 分组\n同上逻辑"]

    J --> K["⑧ aliasBinds 预编译\n每个 child handle 映射到\n其 root handle 首次使用的 pass 槽位"]

    K --> L["⑨ Load/Store Op 推断\n• loadOp:  firstUsage == i → 'clear'  否则 'load'\n• storeOp: lastUsage == i && !imported → 'discard'  否则 'store'"]

    L --> M["⑩ Mermaid 图生成\n生成 graph TD 格式字符串\n写入 mermaidString"]

    M --> N[缓存 CompiledPlan]
    N --> Z
```

---

## 6. execute() 单帧执行流程

```mermaid
flowchart TD
    A([execute encoder]) --> B[compile → CompiledPlan]
    B --> C[将聚合 usage 写回 textureDescs / bufferDescs]
    C --> D{遍历 activePassIndices}

    D --> E["allocations[i]:\n按 desc 从池中取 / 新建物理资源\n→ handleToPhysTex / Buf"]

    E --> F["aliasBinds[i]:\n将 child handle\n指向 root 的物理对象"]

    F --> G{pass.type?}

    G -- Render --> H["组装 colorAttachments\n(view + load/storeOp + clearValue)\n组装 depthStencilAttachment"]
    H --> I[encoder.beginRenderPass]
    I --> J["executeFn(renderPassEnc, this)\n用户代码通过 resolver\n调用 getTextureView / getBuffer"]
    J --> K[renderPassEnc.end]

    G -- Compute --> L[encoder.beginComputePass]
    L --> M["executeFn(computePassEnc, this)"]
    M --> N[computePassEnc.end]

    G -- Generic --> O["executeFn(encoder, this)"]

    K --> P
    N --> P
    O --> P

    P["deallocations[i]:\n释放本 pass 最后使用的资源\n→ poolTextures / poolBuffers"]

    P --> Q{还有更多 pass?}
    Q -- 是 --> D
    Q -- 否 --> R

    R["LRU 清理:\n遍历所有池桶\nframesIdle++ > 5 → texture/buffer.destroy()"]
    R --> S["重置帧状态:\npasses=[], descs.clear()\nhandles reset to 1"]
    S --> T([帧结束])
```

---

## 7. 物理资源池（对象池模式）

```mermaid
stateDiagram-v2
    [*] --> 池中空闲 : 初始/归还

    池中空闲 --> 使用中 : getPhysicalTexture/Buffer\n（池命中，framesIdle=0）
    使用中 --> 池中空闲 : releasePhysical*\n（deallocations[i] 触发）
    [*] --> 使用中 : device.createTexture/Buffer\n（池未命中，新建）

    池中空闲 --> 已销毁 : framesIdle > 5\n（LRU 清理）
    已销毁 --> [*]

    note right of 池中空闲
        池键格式（纹理）:
        "format_width_height_usage"
        池键格式（Buffer）:
        "size_usage"
    end note
```

---

## 关键设计要点

| 机制 | 说明 |
|------|------|
| **拓扑缓存** | 每帧计算 hash，拓扑不变时跳过重编译，O(1) 路径直接执行 |
| **逆向可达性裁剪** | 未被根节点依赖的 pass 自动剔除，无需手动管理 |
| **Kahn 拓扑排序** | 优先级队列维持原声明顺序，保证确定性 |
| **内存别名（Aliasing）** | 生命期不重叠的同类资源复用同一物理对象，减少 GPU 内存分配 |
| **Load/Store 推断** | 首次使用自动 clear，末次非导入使用自动 discard，减少带宽 |
| **对象池 + LRU** | 物理资源跨帧复用，超过 5 帧空闲则销毁，平衡内存与分配开销 |
| **句柄重置** | 每帧末 nextHandle 重置为 1，保证跨帧拓扑哈希可比较 |
| **子资源视图缓存** | `subViews: Map<encodedRange, GPUTextureView>` 按需创建并缓存 mip/layer 视图 |
