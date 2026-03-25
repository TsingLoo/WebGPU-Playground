import fs from 'fs';
import path from 'path';

const shadersDir = './src/shaders';
const shadersTsPath = path.join(shadersDir, 'shaders.ts');

const mapping = {
    // core
    'common.wgsl': 'core',
    'gi_evaluation.wgsl': 'core',
    'naive.vs.wgsl': 'core',
    'naive.fs.wgsl': 'core',
    'geometry.fs.wgsl': 'core',
    'zPrepass.fs.wgsl': 'core',

    // materials
    'standard_material.wgsl': 'materials',
    'unlit_material.wgsl': 'materials',

    // forward_plus
    'forward_plus.fs.wgsl': 'forward_plus',
    'move_lights.cs.wgsl': 'forward_plus',
    'clustering.cs.wgsl': 'forward_plus',

    // clustered_deferred
    'clustered_deferred.fs.wgsl': 'clustered_deferred',
    'clustered_deferred_fullscreen.vs.wgsl': 'clustered_deferred',
    'clustered_deferred_fullscreen.fs.wgsl': 'clustered_deferred',
    'clusteredDeferred.cs.wgsl': 'clustered_deferred',

    // ibl
    'generate_cubemap.cs.wgsl': 'ibl',
    'irradiance_convolution.cs.wgsl': 'ibl',
    'prefilter_envmap.cs.wgsl': 'ibl',
    'brdf_lut.cs.wgsl': 'ibl',
    'equirectangular_to_cubemap.cs.wgsl': 'ibl',

    // ddgi
    'ddgi_probe_trace.cs.wgsl': 'ddgi',
    'ddgi_irradiance_update.cs.wgsl': 'ddgi',
    'ddgi_visibility_update.cs.wgsl': 'ddgi',
    'ddgi_border_update.cs.wgsl': 'ddgi',

    // nrc
    'nrc_common.wgsl': 'nrc',
    'nrc_scatter_training.cs.wgsl': 'nrc',
    'nrc_train.cs.wgsl': 'nrc',
    'nrc_inference.cs.wgsl': 'nrc',

    // surfel
    'surfel_common.wgsl': 'surfel',
    'bvh.wgsl': 'surfel',
    'surfel_lifecycle.cs.wgsl': 'surfel',
    'surfel_grid.cs.wgsl': 'surfel',
    'surfel_integrator.cs.wgsl': 'surfel',
    'surfel_resolve.cs.wgsl': 'surfel',

    // shadows
    'shadow.vs.wgsl': 'shadows',
    'shadow.fs.wgsl': 'shadows',
    'vsm_clear.cs.wgsl': 'shadows',
    'vsm_mark_pages.cs.wgsl': 'shadows',
    'vsm_allocate_pages.cs.wgsl': 'shadows',

    // environment
    'skybox.vs.wgsl': 'environment',
    'skybox.fs.wgsl': 'environment',
    'volumetric_lighting.vs.wgsl': 'environment',
    'volumetric_lighting.fs.wgsl': 'environment',
    'volumetric_composite.fs.wgsl': 'environment',

    // postprocessing
    'ssao.fs.wgsl': 'postprocessing',
    'ssao_blur.fs.wgsl': 'postprocessing'
};

// 1. Read shaders.ts
let shadersTsContent = fs.readFileSync(shadersTsPath, 'utf-8');

// 2. Move files and update imports
for (const [filename, folder] of Object.entries(mapping)) {
    const oldPath = path.join(shadersDir, filename);
    const folderPath = path.join(shadersDir, folder);
    const newPath = path.join(folderPath, filename);

    if (fs.existsSync(oldPath)) {
        if (!fs.existsSync(folderPath)) {
            fs.mkdirSync(folderPath, { recursive: true });
        }
        
        fs.renameSync(oldPath, newPath);
        
        // Update shaders.ts import string
        // Matches exactly: from './filename?raw'
        const regex = new RegExp(`from\\s+['"]./${filename}\\?raw['"]`, 'g');
        shadersTsContent = shadersTsContent.replace(regex, `from './${folder}/${filename}?raw'`);
        
        console.log(`Moved ${filename} to ${folder}/`);
    } else {
        console.warn(`WARNING: File not found: ${filename}`);
    }
}

// 3. Save shaders.ts
fs.writeFileSync(shadersTsPath, shadersTsContent, 'utf-8');
console.log('Successfully updated shaders.ts imports!');
