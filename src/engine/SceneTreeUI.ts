import { Entity } from './Entity';
import { Scene } from './Scene';
import { Component } from './Component';
import { MeshRenderer } from './components/MeshRenderer';
import { CommandManager } from './CommandManager';
import { Pane } from 'tweakpane';
import './SceneTreeUI.css';

// ─── Icon SVGs ────────────────────────────────────────────────────────────────
const ICONS = {
    entity: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="2" y="2" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1.2"/></svg>`,
    mesh: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><polygon points="7,1 13,5 13,10 7,13 1,10 1,5" stroke="#4fc3f7" stroke-width="1.1" fill="none"/><line x1="7" y1="1" x2="7" y2="13" stroke="#4fc3f7" stroke-width="0.7"/><line x1="1" y1="5" x2="13" y2="5" stroke="#4fc3f7" stroke-width="0.7"/></svg>`,
    light: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="6" r="3.5" stroke="#ffd54f" stroke-width="1.1" fill="none"/><line x1="7" y1="11" x2="7" y2="13" stroke="#ffd54f" stroke-width="1"/><line x1="7" y1="0" x2="7" y2="2" stroke="#ffd54f" stroke-width="1"/><line x1="1" y1="6" x2="3" y2="6" stroke="#ffd54f" stroke-width="1"/><line x1="11" y1="6" x2="13" y2="6" stroke="#ffd54f" stroke-width="1"/></svg>`,
    camera: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="3" width="9" height="8" rx="1.5" stroke="#81c784" stroke-width="1.1" fill="none"/><polygon points="10,5 14,3 14,11 10,9" stroke="#81c784" stroke-width="1" fill="none"/></svg>`,
    component: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="5" stroke="#b0bec5" stroke-width="1.1" fill="none"/><circle cx="7" cy="7" r="1.5" fill="#b0bec5"/></svg>`,
    chevronRight: `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M3 1 L7 5 L3 9" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    chevronDown: `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1 3 L5 7 L9 3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    root: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="3" r="2" stroke="#ce93d8" stroke-width="1.1" fill="none"/><line x1="7" y1="5" x2="7" y2="9" stroke="#ce93d8" stroke-width="1"/><line x1="3" y1="9" x2="11" y2="9" stroke="#ce93d8" stroke-width="1"/><line x1="3" y1="9" x2="3" y2="12" stroke="#ce93d8" stroke-width="1"/><line x1="7" y1="9" x2="7" y2="12" stroke="#ce93d8" stroke-width="1"/><line x1="11" y1="9" x2="11" y2="12" stroke="#ce93d8" stroke-width="1"/></svg>`,
};

function getComponentIcon(comp: Component): string {
    const name = comp.constructor.name;
    if (name.includes('MeshRenderer')) return ICONS.mesh;
    if (name.includes('Light')) return ICONS.light;
    if (name.includes('Camera')) return ICONS.camera;
    return ICONS.component;
}



export class SceneTreeUI {
    private scene: Scene | null = null;
    private panel: HTMLDivElement;
    private toggleBtn: HTMLButtonElement;
    private treeContainer: HTMLDivElement;
    private inspector: HTMLDivElement;
    private searchInput: HTMLInputElement;
    private statsEl: HTMLSpanElement;
    private footerEl: HTMLDivElement;

    private selectedEntity: Entity | null = null;
    private expandedSet: WeakSet<Entity> = new WeakSet();
    private isOpen: boolean = true;
    private searchQuery: string = '';


    private inspectorPane: Pane | null = null;
    private updateRAF: number = 0;

    private totalEntities: number = 0;
    private totalComponents: number = 0;

    constructor() {
        CommandManager.initialize();

        this.toggleBtn = document.createElement('button');
        this.toggleBtn.id = 'scene-tree-toggle';
        this.toggleBtn.innerHTML = '☰';
        this.toggleBtn.title = 'Toggle Scene Hierarchy';
        this.toggleBtn.classList.add('panel-open');
        this.toggleBtn.addEventListener('click', () => this.toggle());
        document.body.appendChild(this.toggleBtn);

        this.panel = document.createElement('div');
        this.panel.id = 'scene-tree-panel';
        document.body.appendChild(this.panel);

        const header = document.createElement('div');
        header.className = 'stp-header';
        header.innerHTML = `
            <span class="stp-header-title">${ICONS.root} Scene Hierarchy</span>
            <span class="stp-stats"></span>
        `;
        this.panel.appendChild(header);
        this.statsEl = header.querySelector('.stp-stats') as HTMLSpanElement;

        const searchWrap = document.createElement('div');
        searchWrap.className = 'stp-search-wrap';
        this.searchInput = document.createElement('input');
        this.searchInput.className = 'stp-search';
        this.searchInput.type = 'text';
        this.searchInput.placeholder = 'Search...';
        this.searchInput.addEventListener('input', () => {
            this.searchQuery = this.searchInput.value.toLowerCase();
            this.rebuild();
        });
        searchWrap.appendChild(this.searchInput);
        this.panel.appendChild(searchWrap);

        this.treeContainer = document.createElement('div');
        this.treeContainer.className = 'stp-tree-container';
        this.panel.appendChild(this.treeContainer);

        // This will be the tweakpane container
        this.inspector = document.createElement('div');
        this.inspector.className = 'stp-inspector tweakpane-wrapper';
        this.panel.appendChild(this.inspector);

        this.footerEl = document.createElement('div');
        this.footerEl.className = 'stp-footer';
        this.panel.appendChild(this.footerEl);

        this.updateLoop = this.updateLoop.bind(this);
        this.updateRAF = requestAnimationFrame(this.updateLoop);
    }

    private updateLoop() {
        if (this.isOpen && this.inspectorPane && this.selectedEntity) {
            (this.inspectorPane as any).refresh();
        }
        this.updateRAF = requestAnimationFrame(this.updateLoop);
    }

    public setScene(scene: Scene) {
        this.scene = scene;
        this.expandedSet.add(scene.root);
        this.rebuild();
    }

    public toggle() {
        this.isOpen = !this.isOpen;
        this.panel.classList.toggle('collapsed', !this.isOpen);
        this.toggleBtn.classList.toggle('panel-open', this.isOpen);
        this.toggleBtn.innerHTML = this.isOpen ? '☰' : '▶';
    }

    public refresh() {
        this.rebuild();
    }

    private rebuild() {
        this.treeContainer.innerHTML = '';
        this.totalEntities = 0;
        this.totalComponents = 0;

        if (!this.scene) {
            this.treeContainer.innerHTML = '<div style="padding:20px;color:#4a5060;text-align:center;font-size:11px;">No scene loaded</div>';
            this.footerEl.innerHTML = '';
            this.statsEl.textContent = '';
            return;
        }

        this.countRecursive(this.scene.root);
        this.statsEl.textContent = `${this.totalEntities} entities`;

        const rootEl = this.buildEntityNode(this.scene.root, 0, true);
        if (rootEl) {
            this.treeContainer.appendChild(rootEl);
        }

        this.footerEl.innerHTML = `
            <span>Entities: ${this.totalEntities}</span>
            <span>Components: ${this.totalComponents}</span>
        `;
    }

    private countRecursive(entity: Entity) {
        this.totalEntities++;
        this.totalComponents += entity.getAllComponents().length;
        for (const child of entity.children) {
            this.countRecursive(child);
        }
    }

    private matchesSearch(entity: Entity): boolean {
        if (!this.searchQuery) return true;
        if (entity.name.toLowerCase().includes(this.searchQuery)) return true;
        for (const c of entity.getAllComponents()) {
            if (c.constructor.name.toLowerCase().includes(this.searchQuery)) return true;
        }
        for (const child of entity.children) {
            if (this.matchesSearch(child)) return true;
        }
        return false;
    }

    private buildEntityNode(entity: Entity, depth: number, isRoot: boolean = false): HTMLDivElement | null {
        if (this.searchQuery && !this.matchesSearch(entity)) return null;

        const node = document.createElement('div');
        node.className = 'stp-node';

        const hasChildren = entity.children.size > 0;
        const isExpanded = this.expandedSet.has(entity);

        if (this.searchQuery && hasChildren) {
            this.expandedSet.add(entity);
        }

        const row = document.createElement('div');
        row.className = 'stp-node-row';
        if (this.selectedEntity === entity) row.classList.add('selected');

        const indent = document.createElement('span');
        indent.style.width = `${depth * 16 + 4}px`;
        indent.style.flexShrink = '0';
        row.appendChild(indent);

        const chevron = document.createElement('span');
        chevron.className = 'stp-chevron' + (hasChildren ? '' : ' empty');
        if (hasChildren) {
            chevron.innerHTML = isExpanded ? ICONS.chevronDown : ICONS.chevronRight;
            chevron.addEventListener('click', (e) => {
                e.stopPropagation();
                if (this.expandedSet.has(entity)) {
                    this.expandedSet.delete(entity);
                } else {
                    this.expandedSet.add(entity);
                }
                this.rebuild();
            });
        }
        row.appendChild(chevron);

        const icon = document.createElement('span');
        icon.className = 'stp-node-icon';
        if (isRoot) {
            icon.innerHTML = ICONS.root;
        } else {
            const comps = entity.getAllComponents();
            if (comps.length > 0) {
                icon.innerHTML = getComponentIcon(comps[0]);
            } else {
                icon.innerHTML = ICONS.entity;
            }
        }
        row.appendChild(icon);

        const label = document.createElement('span');
        label.className = 'stp-node-label' + (isRoot ? ' root-label' : '');
        if (this.searchQuery && entity.name.toLowerCase().includes(this.searchQuery)) {
            label.innerHTML = this.highlightText(entity.name, this.searchQuery);
        } else {
            label.textContent = entity.name;
        }
        row.appendChild(label);

        const comps = entity.getAllComponents();
        if (comps.length > 0) {
            const badges = document.createElement('span');
            badges.className = 'stp-comp-badges';
            for (const comp of comps) {
                const badge = document.createElement('span');
                badge.className = 'stp-comp-badge';
                badge.innerHTML = getComponentIcon(comp);
                badge.title = comp.constructor.name;
                badges.appendChild(badge);
            }
            row.appendChild(badges);
        }

        if (hasChildren && !isExpanded) {
            const countBadge = document.createElement('span');
            countBadge.className = 'stp-children-count';
            countBadge.textContent = `(${entity.children.size})`;
            row.appendChild(countBadge);
        }

        row.addEventListener('click', () => {
            this.selectedEntity = entity;
            this.rebuild();
            this.showInspector(entity);
        });

        row.addEventListener('dblclick', (e) => {
            e.preventDefault();
            if (hasChildren) {
                if (this.expandedSet.has(entity)) {
                    this.expandedSet.delete(entity);
                } else {
                    this.expandedSet.add(entity);
                }
                this.rebuild();
            }
        });

        node.appendChild(row);

        if (hasChildren && (isExpanded || this.searchQuery)) {
            for (const child of entity.children) {
                const childEl = this.buildEntityNode(child, depth + 1);
                if (childEl) {
                    node.appendChild(childEl);
                }
            }
        }

        return node;
    }

    private highlightText(text: string, query: string): string {
        const idx = text.toLowerCase().indexOf(query);
        if (idx < 0) return text;
        const before = text.substring(0, idx);
        const match = text.substring(idx, idx + query.length);
        const after = text.substring(idx + query.length);
        return `${before}<span class="stp-highlight">${match}</span>${after}`;
    }

    // ─── Tweakpane Inspector ──────────────────────────────────────────────────
    private showInspector(entity: Entity) {
        if (this.inspectorPane) {
            this.inspectorPane.dispose();
        }

        this.inspectorPane = new Pane({ container: this.inspector, title: entity.name });

        // Helper to add a Tweakpane binding that supports Undo/Redo
        const addUndoableBinding = (folder: any, target: any, key: string, params: any = {}, onChangeCb?: () => void) => {
            let initialValue = structuredClone(target[key]);
            
            const binding = folder.addBinding(target, key, params);
            
            binding.on('change', (ev: any) => {
                if (onChangeCb) onChangeCb();
                
                // When dragging is finalized
                if (ev.last) {
                    const oldVal = structuredClone(initialValue);
                    const newVal = structuredClone(ev.value);
                    
                    if (JSON.stringify(oldVal) !== JSON.stringify(newVal)) {
                        CommandManager.push({
                            execute: () => {
                                target[key] = structuredClone(newVal);
                                if (onChangeCb) onChangeCb();
                                if (this.inspectorPane) (this.inspectorPane as any).refresh();
                            },
                            undo: () => {
                                target[key] = structuredClone(oldVal);
                                if (onChangeCb) onChangeCb();
                                if (this.inspectorPane) (this.inspectorPane as any).refresh();
                            }
                        });
                    }
                    initialValue = structuredClone(newVal);
                }
            });
            return binding;
        };

        // 1. Transform
        const transFolder = (this.inspectorPane as any).addFolder({ title: 'Transform' });
        
        const posProxy = {
            position: {
                get x() { return entity.localTransform[12]; },
                set x(v) { entity.localTransform[12] = v; entity.setTransformDirty(); },
                get y() { return entity.localTransform[13]; },
                set y(v) { entity.localTransform[13] = v; entity.setTransformDirty(); },
                get z() { return entity.localTransform[14]; },
                set z(v) { entity.localTransform[14] = v; entity.setTransformDirty(); }
            }
        };

        addUndoableBinding(transFolder, posProxy, 'position', {}, () => entity.setTransformDirty());

        // 2. Components
        const comps = entity.getAllComponents();
        for (const comp of comps) {
            const compFolder = (this.inspectorPane as any).addFolder({ title: comp.constructor.name });
            
            const props = this.getComponentProperties(comp);
            for (const prop of props) {
                if (prop.readonly) {
                    compFolder.addBinding(prop.obj, prop.key, { readonly: true, label: prop.key });
                } else if (prop.isColorArray) {
                    const colorProxy = {
                        color: {
                            get r() { return (comp as any)[prop.key][0]; },
                            set r(v) { (comp as any)[prop.key][0] = v; },
                            get g() { return (comp as any)[prop.key][1]; },
                            set g(v) { (comp as any)[prop.key][1] = v; },
                            get b() { return (comp as any)[prop.key][2]; },
                            set b(v) { (comp as any)[prop.key][2] = v; }
                        }
                    };
                    addUndoableBinding(compFolder, colorProxy, 'color', { color: { type: 'float' }, label: prop.key });
                } else if (prop.isVectorArray) {
                    const vecProxy = {
                        vector: {
                            get x() { return (comp as any)[prop.key][0]; },
                            set x(v) { (comp as any)[prop.key][0] = v; },
                            get y() { return (comp as any)[prop.key][1]; },
                            set y(v) { (comp as any)[prop.key][1] = v; },
                            get z() { return (comp as any)[prop.key][2]; },
                            set z(v) { (comp as any)[prop.key][2] = v; }
                        }
                    };
                    addUndoableBinding(compFolder, vecProxy, 'vector', { label: prop.key, ...prop.options });
                } else {
                    addUndoableBinding(compFolder, prop.obj, prop.key, { ...prop.options, label: prop.key });
                }
            }
        }
    }

    private getComponentProperties(comp: Component): {obj: any, key: string, readonly?: boolean, options?: any, isColorArray?: boolean, isVectorArray?: boolean}[] {
        const props: any[] = [];
        if (!(comp as any).hideEnableInUI) {
            props.push({ obj: comp, key: 'enabled' });
        }

        if (comp instanceof MeshRenderer) {
            if (comp.mesh) {
                props.push({ obj: { primitives: comp.mesh.primitives.length }, key: 'primitives', readonly: true });
                const totalIndices = comp.mesh.primitives.reduce((sum, p) => sum + p.numIndices, 0);
                props.push({ obj: { indices: totalIndices.toLocaleString() }, key: 'indices', readonly: true });
            } else {
                props.push({ obj: { mesh: 'null' }, key: 'mesh', readonly: true });
            }
            props.push({ obj: { bound: comp.modelBindGroup ? 'yes' : 'no' }, key: 'bound', readonly: true });
        } else {
            const descriptors = Object.getOwnPropertyDescriptors(comp);
            for (const [key, desc] of Object.entries(descriptors)) {
                if (key === 'entity' || key === 'enabled' || key === 'hideEnableInUI') continue;
                
                const val = desc.get ? desc.get.call(comp) : desc.value;
                if (typeof val === 'function') continue;

                let isColorArray = false;
                let isVectorArray = false;
                if (Array.isArray(val) && val.length === 3 && val.every(v => typeof v === 'number')) {
                    if (key.toLowerCase().includes('color')) {
                        isColorArray = true;
                    } else {
                        isVectorArray = true;
                    }
                }

                if (desc.get && !desc.set) {
                   props.push({ obj: comp, key, readonly: true });
                } else if (val === undefined || val === null) {
                   props.push({ obj: { [key]: 'null' }, key, readonly: true });
                } else if (Array.isArray(val) && !isColorArray && !isVectorArray) {
                   props.push({ obj: { [key]: `[${val.join(', ')}]` }, key, readonly: true });
                } else if (typeof desc.value === 'object' && !isColorArray && !isVectorArray) {
                   props.push({ obj: { [key]: desc.value.constructor?.name ?? 'Object' }, key, readonly: true });
                } else {
                   const uiOptions = (comp as any).getUIOptions ? (comp as any).getUIOptions()[key] : undefined;
                   const tpOpts: any = {};
                   if (uiOptions && typeof uiOptions === 'object' && !Array.isArray(uiOptions)) {
                       if (uiOptions.min !== undefined) tpOpts.min = uiOptions.min;
                       if (uiOptions.step !== undefined) tpOpts.step = uiOptions.step;
                       if (uiOptions.max !== undefined) tpOpts.max = uiOptions.max;
                   } else if (uiOptions && Array.isArray(uiOptions)) {
                       tpOpts.options = {};
                       // Normalize {label, value} vs raw string array
                       for (const o of uiOptions) {
                           if (typeof o === 'object') tpOpts.options[o.label] = o.value;
                           else tpOpts.options[String(o)] = o;
                       }
                   } else if (typeof val === 'number') {
                       if (key === 'intensity' || key.toLowerCase().includes('color')) {
                           tpOpts.min = 0;
                       }
                   }

                   props.push({ obj: comp, key, options: tpOpts, isColorArray, isVectorArray });
                }
            }
        }
        return props;
    }

    public destroy() {
        cancelAnimationFrame(this.updateRAF);
        this.panel.remove();
        this.toggleBtn.remove();
    }
}
