import * as renderer from '../../renderer';
import * as shaders from '../../shaders/shaders';

export class HiZPass {
    private copyPipeline: GPUComputePipeline;
    private downsamplePipeline: GPUComputePipeline;
    private copyBindGroupLayout: GPUBindGroupLayout;
    private downsampleBindGroupLayout: GPUBindGroupLayout;
    
    public hizTexture!: GPUTexture;
    public hizTextureViews: GPUTextureView[] = [];
    public hizFullTextureView!: GPUTextureView;
    public mipLevelCount: number = 1;
    public hizSize: [number, number] = [0, 0];

    constructor() {
        this.copyBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "hiz copy bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } }
            ]
        });

        this.downsampleBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "hiz downsample bgl",
            entries: [
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } }
            ]
        });

        this.copyPipeline = renderer.device.createComputePipeline({
            label: "hiz copy pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.copyBindGroupLayout] }),
            compute: { module: renderer.device.createShaderModule({ code: shaders.hizDownsampleSrc }), entryPoint: "copy_main" }
        });

        this.downsamplePipeline = renderer.device.createComputePipeline({
            label: "hiz downsample pipeline",
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [this.downsampleBindGroupLayout] }),
            compute: { module: renderer.device.createShaderModule({ code: shaders.hizDownsampleSrc }), entryPoint: "downsample_main" }
        });
    }

    resize(width: number, height: number) {
        // Let garbage collection handle old textures natively
        

        this.hizSize = [width, height];
        this.mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

        this.hizTexture = renderer.device.createTexture({
            label: "MinMax Hi-Z Texture",
            size: this.hizSize,
            format: "rg32float",
            mipLevelCount: this.mipLevelCount,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
        });
        this.hizFullTextureView = this.hizTexture.createView();

        this.hizTextureViews = [];
        for (let i = 0; i < this.mipLevelCount; i++) {
            this.hizTextureViews.push(this.hizTexture.createView({
                baseMipLevel: i,
                mipLevelCount: 1
            }));
        }
    }

    execute(encoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {
        if (!this.hizTexture) return;

        const pass = encoder.beginComputePass({ label: "Hi-Z Pass" });

        // Phase 0: Copy Depth to Mip 0
        const copyBindGroup = renderer.device.createBindGroup({
            layout: this.copyBindGroupLayout,
            entries: [
                { binding: 0, resource: depthTextureView },
                { binding: 1, resource: this.hizTextureViews[0] }
            ]
        });

        pass.setPipeline(this.copyPipeline);
        pass.setBindGroup(0, copyBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.hizSize[0] / 8), Math.ceil(this.hizSize[1] / 8), 1);

        // Phase 1..N: Downsample
        pass.setPipeline(this.downsamplePipeline);
        let currWidth = this.hizSize[0];
        let currHeight = this.hizSize[1];

        for (let i = 1; i < this.mipLevelCount; i++) {
            currWidth = Math.max(1, Math.floor(currWidth / 2));
            currHeight = Math.max(1, Math.floor(currHeight / 2));

            const bindGroup = renderer.device.createBindGroup({
                layout: this.downsampleBindGroupLayout,
                entries: [
                    { binding: 2, resource: this.hizTextureViews[i - 1] },
                    { binding: 3, resource: this.hizTextureViews[i] }
                ]
            });

            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(currWidth / 8), Math.ceil(currHeight / 8), 1);
        }

        pass.end();
    }
}
