export interface BindingConfig {
    compKey: string;
    stageKey?: string; // If omitted, defaults to compKey. Can be "path.to.prop" if needed.
    transformSet?: (val: any) => any;
    customGet?: (target: any) => any;
    customSet?: (target: any, val: any) => void;
}

export interface ComponentConfig {
    targetSystem: string; // The property key on stageObj (e.g. 'ddgi'). '' means stageObj itself.
    bindings: (string | BindingConfig)[];
    onUpdate?: (stageObj: any, globals: any) => void;
}

// 声明式的渲染参数绑定配置表
// AI 在此处添加新的 Component 绑定时，只需按 JSON 格式配置即可，无需再手写 Object.defineProperty
export const renderSchema: Record<string, ComponentConfig> = {
    'DirectionalLightComponent': {
        targetSystem: '',
        bindings: [
            { compKey: 'intensity', stageKey: 'sunIntensity' }
        ]
    },
    'PointLightSettingsComponent': {
        targetSystem: 'lights',
        bindings: [
            { compKey: 'enabled', stageKey: 'enabled' },
            { compKey: 'count', stageKey: 'numLights' }
        ],
        onUpdate: (stageObj) => stageObj.lights.updateLightSetUniformNumLights()
    },
    'VolumetricFogComponent': {
        targetSystem: '',
        bindings: [
            { compKey: 'enabled', stageKey: 'sunVolumetricEnabled' },
            { compKey: 'intensity', stageKey: 'sunVolumetricIntensity' },
            { compKey: 'heightFalloff', stageKey: 'sunVolumetricHeightFalloff' },
            { compKey: 'heightScale', stageKey: 'sunVolumetricHeightScale' },
            { compKey: 'maxDist', stageKey: 'sunVolumetricMaxDist' },
            { compKey: 'steps', stageKey: 'sunVolumetricSteps' }
        ]
    },
    'VSMShadowComponent': {
        targetSystem: 'vsm',
        bindings: ['physAtlasSize', 'pageSize', 'numClipmapLevels', 'pagesPerLevelAxis'],
        onUpdate: (stageObj, globals) => {
            stageObj.vsm.recreate();
            stageObj.updateSunLight();
            if (globals.setRenderer && globals.renderModeController) {
                globals.setRenderer(globals.renderModeController.getValue());
            }
        }
    },
    'GIComponent': {
        targetSystem: '',
        bindings: [
            { compKey: 'showGIBounds', stageKey: 'showGIBounds' }
        ]
    },
    'DDGIComponent': {
        targetSystem: 'ddgi',
        bindings: [
            'showProbes', 'irradianceHysteresis', 'visibilityHysteresis', 'probeTraceAmbient', 'debugMode',
            { compKey: 'gridMin', transformSet: (v) => Array.from(v) },
            { compKey: 'gridMax', transformSet: (v) => Array.from(v) }
        ],
        onUpdate: (stageObj) => { 
            stageObj.ddgi.updateUniforms();
            stageObj.ddgi.reset(); 
        }
    },
    'RadianceCascadesComponent': {
        targetSystem: 'radianceCascades',
        bindings: [
            'intensity', 'ambient', 'hysteresis', 'debugMode',
            { compKey: 'gridMin', transformSet: (v) => Array.from(v) },
            { compKey: 'gridMax', transformSet: (v) => Array.from(v) }
        ],
        onUpdate: (stageObj) => stageObj.radianceCascades.updateUniforms()
    },
    'SSAOComponent': {
        targetSystem: 'ssao',
        bindings: ['enabled', 'radius', 'bias', 'power'],
        onUpdate: (stageObj) => stageObj.ssao.updateUniforms()
    }
};

export function applyComponentSchema(comp: any, compName: string, stageObj: any, globals: any = {}) {
    const config = renderSchema[compName];
    if (!config) {
        console.warn(`[RenderSchema] No binding config found for ${compName}`);
        return;
    }

    const target = config.targetSystem ? stageObj[config.targetSystem] : stageObj;

    for (const b of config.bindings) {
        const compKey = typeof b === 'string' ? b : b.compKey;
        const stageKey = typeof b === 'string' ? b : (b.stageKey || b.compKey);
        const transformSet = typeof b === 'string' ? null : b.transformSet;
        const customGet = typeof b === 'string' ? null : b.customGet;
        const customSet = typeof b === 'string' ? null : b.customSet;

        Object.defineProperty(comp, compKey, {
            get: () => {
                if (customGet) return customGet(target);
                return target[stageKey];
            },
            set: (v) => {
                if (customSet) {
                    customSet(target, v);
                } else {
                    target[stageKey] = transformSet ? transformSet(v) : v;
                }
                if (config.onUpdate) {
                    config.onUpdate(stageObj, globals);
                }
            },
            enumerable: true
        });
    }
}
