import { Entity } from './Entity';
import { Scene } from './Scene';
import { Component } from './Component';
import { MeshRenderer } from './components/MeshRenderer';
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

// ─── Component type → icon mapping ────────────────────────────────────────────
function getComponentIcon(comp: Component): string {
    const name = comp.constructor.name;
    if (name.includes('MeshRenderer')) return ICONS.mesh;
    if (name.includes('Light')) return ICONS.light;
    if (name.includes('Camera')) return ICONS.camera;
    return ICONS.component;
}

function getComponentColor(comp: Component): string {
    const name = comp.constructor.name;
    if (name.includes('MeshRenderer')) return '#4fc3f7';
    if (name.includes('Light')) return '#ffd54f';
    if (name.includes('Camera')) return '#81c784';
    return '#b0bec5';
}


// ─── Main Class ───────────────────────────────────────────────────────────────
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

    // Inspector dynamic elements
    private inspectorActiveEntity: Entity | null = null;
    private updatableElements: { el: HTMLElement | HTMLInputElement | HTMLSelectElement, getVal: () => any }[] = [];
    private updateRAF: number = 0;

    // Track total counts
    private totalEntities: number = 0;
    private totalComponents: number = 0;

    constructor() {
        // Toggle button
        this.toggleBtn = document.createElement('button');
        this.toggleBtn.id = 'scene-tree-toggle';
        this.toggleBtn.innerHTML = '☰';
        this.toggleBtn.title = 'Toggle Scene Hierarchy';
        this.toggleBtn.classList.add('panel-open');
        this.toggleBtn.addEventListener('click', () => this.toggle());
        document.body.appendChild(this.toggleBtn);

        // Panel
        this.panel = document.createElement('div');
        this.panel.id = 'scene-tree-panel';
        document.body.appendChild(this.panel);

        // Header
        const header = document.createElement('div');
        header.className = 'stp-header';
        header.innerHTML = `
            <span class="stp-header-title">${ICONS.root} Scene Hierarchy</span>
            <span class="stp-stats"></span>
        `;
        this.panel.appendChild(header);
        this.statsEl = header.querySelector('.stp-stats') as HTMLSpanElement;

        // Search
        const searchWrap = document.createElement('div');
        searchWrap.className = 'stp-search-wrap';
        this.searchInput = document.createElement('input');
        this.searchInput.className = 'stp-search';
        this.searchInput.type = 'text';
        this.searchInput.placeholder = 'Search entities...';
        this.searchInput.addEventListener('input', () => {
            this.searchQuery = this.searchInput.value.toLowerCase();
            this.rebuild();
        });
        searchWrap.appendChild(this.searchInput);
        this.panel.appendChild(searchWrap);

        // Tree container
        this.treeContainer = document.createElement('div');
        this.treeContainer.className = 'stp-tree-container';
        this.panel.appendChild(this.treeContainer);

        // Inspector (bottom)
        this.inspector = document.createElement('div');
        this.inspector.className = 'stp-inspector';
        this.inspector.style.display = 'none';
        this.panel.appendChild(this.inspector);

        // Footer
        this.footerEl = document.createElement('div');
        this.footerEl.className = 'stp-footer';
        this.panel.appendChild(this.footerEl);

        // Start real-time inspector update loop
        this.updateLoop = this.updateLoop.bind(this);
        this.updateRAF = requestAnimationFrame(this.updateLoop);
    }

    private updateLoop() {
        if (this.isOpen && this.inspectorActiveEntity) {
            for (const item of this.updatableElements) {
                // Skip updating if user is actively focused/typing
                if (document.activeElement === item.el) continue;

                const val = item.getVal();
                if (item.el instanceof HTMLInputElement) {
                    if (item.el.type === 'checkbox') {
                        item.el.checked = val as boolean;
                    } else if (item.el.type === 'number') {
                        // Only update if difference is noticeable
                        const numVal = val as number;
                        const isInt = item.el.step && parseFloat(item.el.step) % 1 === 0;
                        if (Math.abs(parseFloat(item.el.value) - numVal) > 0.0001) {
                            item.el.value = isInt ? Math.round(numVal).toString() : numVal.toFixed(3);
                        }
                    }
                } else if (item.el instanceof HTMLSelectElement) {
                    item.el.value = String(val);
                } else if (val instanceof Float32Array) {
                    item.el.textContent = `${val[0].toFixed(2)}, ${val[1].toFixed(2)}, ${val[2].toFixed(2)}`;
                } else {
                    item.el.textContent = String(val);
                }
            }
        }
        this.updateRAF = requestAnimationFrame(this.updateLoop);
    }

    public setScene(scene: Scene) {
        this.scene = scene;
        // Expand root by default
        this.expandedSet.add(scene.root);
        this.rebuild();
    }

    public toggle() {
        this.isOpen = !this.isOpen;
        this.panel.classList.toggle('collapsed', !this.isOpen);
        this.toggleBtn.classList.toggle('panel-open', this.isOpen);
        this.toggleBtn.innerHTML = this.isOpen ? '☰' : '▶';
    }

    /**
     * Call this when scene structure changes to rebuild the tree.
     */
    public refresh() {
        this.rebuild();
    }

    // ─── Build the tree ───────────────────────────────────────────────────────
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
        // Check component names
        for (const c of entity.getAllComponents()) {
            if (c.constructor.name.toLowerCase().includes(this.searchQuery)) return true;
        }
        // Check if any descendant matches
        for (const child of entity.children) {
            if (this.matchesSearch(child)) return true;
        }
        return false;
    }

    private buildEntityNode(entity: Entity, depth: number, isRoot: boolean = false): HTMLDivElement | null {
        // Filter by search
        if (this.searchQuery && !this.matchesSearch(entity)) {
            return null;
        }

        const node = document.createElement('div');
        node.className = 'stp-node';

        const hasChildren = entity.children.size > 0;
        const isExpanded = this.expandedSet.has(entity);

        // If searching, auto-expand
        if (this.searchQuery && hasChildren) {
            this.expandedSet.add(entity);
        }

        // ─── Row ──────────────────────────────────────────────────────────
        const row = document.createElement('div');
        row.className = 'stp-node-row';
        if (this.selectedEntity === entity) row.classList.add('selected');

        // Indent
        const indent = document.createElement('span');
        indent.style.width = `${depth * 16 + 4}px`;
        indent.style.flexShrink = '0';
        row.appendChild(indent);

        // Chevron
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

        // Entity icon
        const icon = document.createElement('span');
        icon.className = 'stp-node-icon';
        if (isRoot) {
            icon.innerHTML = ICONS.root;
        } else {
            // Use the primary component's icon, or generic entity icon
            const comps = entity.getAllComponents();
            if (comps.length > 0) {
                icon.innerHTML = getComponentIcon(comps[0]);
            } else {
                icon.innerHTML = ICONS.entity;
            }
        }
        row.appendChild(icon);

        // Label
        const label = document.createElement('span');
        label.className = 'stp-node-label' + (isRoot ? ' root-label' : '');
        if (this.searchQuery && entity.name.toLowerCase().includes(this.searchQuery)) {
            label.innerHTML = this.highlightText(entity.name, this.searchQuery);
        } else {
            label.textContent = entity.name;
        }
        row.appendChild(label);

        // Component badges
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

        // Child count (if collapsed with children)
        if (hasChildren && !isExpanded) {
            const countBadge = document.createElement('span');
            countBadge.className = 'stp-children-count';
            countBadge.textContent = `(${entity.children.size})`;
            row.appendChild(countBadge);
        }

        // Click to select
        row.addEventListener('click', () => {
            this.selectedEntity = entity;
            this.rebuild();
            this.showInspector(entity);
        });

        // Double click to toggle expand
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

        // ─── Children ─────────────────────────────────────────────────────
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

    // ─── Inspector Panel ──────────────────────────────────────────────────────
    private showInspector(entity: Entity) {
        this.inspectorActiveEntity = entity;
        this.updatableElements = [];
        this.inspector.style.display = 'block';
        this.inspector.innerHTML = '';

        // Title
        const title = document.createElement('div');
        title.className = 'stp-inspector-title';
        title.textContent = 'Inspector';
        this.inspector.appendChild(title);

        // Entity name
        const nameEl = document.createElement('div');
        nameEl.className = 'stp-inspector-entity-name';
        nameEl.textContent = entity.name;
        this.inspector.appendChild(nameEl);

        // Transform
        const transformSection = document.createElement('div');
        transformSection.className = 'stp-inspector-section';
        transformSection.innerHTML = `
            <div class="stp-inspector-section-title" style="color:#8e95a4;">Transform</div>
        `;
        
        const posRow = document.createElement('div');
        posRow.className = 'stp-inspector-row';
        posRow.innerHTML = `<span class="stp-inspector-key">Position</span>`;
        
        const posVal = document.createElement('span');
        posVal.className = 'stp-inspector-value';
        // Use localTransform for edits instead of worldTransform
        const lt = entity.localTransform;
        posRow.appendChild(posVal);
        transformSection.appendChild(posRow);
        
        this.updatableElements.push({
            el: posVal,
            getVal: () => new Float32Array([entity.worldTransform[12], entity.worldTransform[13], entity.worldTransform[14]])
        });

        // Add editable local position
        const editRow = document.createElement('div');
        editRow.className = 'stp-inspector-row';
        editRow.innerHTML = `<span class="stp-inspector-key">Local Pos</span>`;
        
        const xInput = this.createNumberInput(lt[12], v => { lt[12] = v; entity.setTransformDirty(); });
        const yInput = this.createNumberInput(lt[13], v => { lt[13] = v; entity.setTransformDirty(); });
        const zInput = this.createNumberInput(lt[14], v => { lt[14] = v; entity.setTransformDirty(); });
        
        editRow.appendChild(xInput);
        editRow.appendChild(yInput);
        editRow.appendChild(zInput);
        transformSection.appendChild(editRow);

        this.updatableElements.push({ el: xInput, getVal: () => entity.localTransform[12] });
        this.updatableElements.push({ el: yInput, getVal: () => entity.localTransform[13] });
        this.updatableElements.push({ el: zInput, getVal: () => entity.localTransform[14] });

        this.inspector.appendChild(transformSection);

        // Components
        const comps = entity.getAllComponents();
        if (comps.length > 0) {
            for (const comp of comps) {
                const section = document.createElement('div');
                section.className = 'stp-inspector-section';
                
                const color = getComponentColor(comp);
                section.innerHTML = `
                    <div class="stp-inspector-section-title" style="color:${color};">
                        ${getComponentIcon(comp)} ${comp.constructor.name}
                    </div>
                `;

                const props = this.getComponentProperties(comp);
                for (const prop of props) {
                    const row = document.createElement('div');
                    row.className = 'stp-inspector-row';
                    row.innerHTML = `<span class="stp-inspector-key">${prop.key}</span>`;
                    

                    if (prop.options && Array.isArray(prop.options)) {
                        const select = document.createElement('select');
                        select.className = 'stp-input';
                        for (const opt of prop.options) {
                            const option = document.createElement('option');
                            const val = typeof opt === 'object' ? opt.value : opt;
                            const label = typeof opt === 'object' ? opt.label : opt;
                            option.value = String(val);
                            option.textContent = String(label);
                            if (val === prop.initialValue) option.selected = true;
                            select.appendChild(option);
                        }
                        select.addEventListener('change', () => {
                            let val: any = select.value;
                            if (typeof prop.initialValue === 'number') val = parseFloat(val);
                            if (prop.setVal) prop.setVal(val);
                        });
                        row.appendChild(select);
                        if (prop.getVal) this.updatableElements.push({ el: select, getVal: prop.getVal });
                    } else if (prop.readonly || (typeof prop.initialValue !== 'number' && typeof prop.initialValue !== 'boolean' && !Array.isArray(prop.initialValue))) {
                        const valSpan = document.createElement('span');
                        valSpan.className = 'stp-inspector-value';
                        valSpan.textContent = String(prop.initialValue);
                        row.appendChild(valSpan);
                        if (prop.getVal) this.updatableElements.push({ el: valSpan, getVal: prop.getVal });
                    } else if (typeof prop.initialValue === 'boolean') {
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.className = 'stp-input';
                        checkbox.checked = prop.initialValue as boolean;
                        checkbox.addEventListener('change', () => { if (prop.setVal) prop.setVal(checkbox.checked); });
                        row.appendChild(checkbox);
                        if (prop.getVal) this.updatableElements.push({ el: checkbox, getVal: prop.getVal });
                    } else if (typeof prop.initialValue === 'number') {
                        const numInput = this.createNumberInput(prop.initialValue as number, v => { if (prop.setVal) prop.setVal(v); }, prop.min, prop.step);
                        row.appendChild(numInput);
                        if (prop.getVal) this.updatableElements.push({ el: numInput, getVal: prop.getVal });
                    } else if (Array.isArray(prop.initialValue)) {
                        const arrWrapper = document.createElement('div');
                        arrWrapper.style.display = 'flex';
                        arrWrapper.style.gap = '2px';
                        arrWrapper.style.flex = '1';
                        
                        const arr = prop.initialValue as any[];
                        for (let i = 0; i < arr.length; i++) {
                            if (typeof arr[i] === 'number') {
                                const numInput = this.createNumberInput(arr[i], v => { if (prop.setArrVal) prop.setArrVal(i, v); }, prop.min, prop.step);
                                numInput.style.minWidth = "0"; // allow shrinking
                                arrWrapper.appendChild(numInput);
                                if (prop.getArrVal) this.updatableElements.push({ el: numInput, getVal: () => prop.getArrVal!(i) });
                            } else {
                                const span = document.createElement('span');
                                span.className = 'stp-inspector-value';
                                span.textContent = String(arr[i]);
                                arrWrapper.appendChild(span);
                            }
                        }
                        row.appendChild(arrWrapper);
                    }
                    
                    section.appendChild(row);
                }

                this.inspector.appendChild(section);
            }
        }

        // Children info
        if (entity.children.size > 0) {
            const childSection = document.createElement('div');
            childSection.className = 'stp-inspector-section';
            childSection.innerHTML = `
                <div class="stp-inspector-section-title" style="color:#7b8394;">Children (${entity.children.size})</div>
            `;
            const names = Array.from(entity.children).slice(0, 8).map(c => c.name);
            if (entity.children.size > 8) names.push(`... +${entity.children.size - 8}`);
            childSection.innerHTML += `
                <div class="stp-inspector-row">
                    <span class="stp-inspector-value">${names.join(', ')}</span>
                </div>
            `;
            this.inspector.appendChild(childSection);
        }
    }

    private createNumberInput(val: number, onChange: (v: number) => void, min?: number, step: number = 0.1): HTMLInputElement {
        const input = document.createElement('input');
        input.type = 'number';
        input.step = String(step);
        if (min !== undefined) input.min = String(min);
        const formatVal = (v: number) => step % 1 === 0 ? Math.round(v).toString() : v.toFixed(3);
        input.value = formatVal(val);
        input.className = 'stp-input';
        
        // Hint that it can be dragged
        input.style.cursor = 'ew-resize';

        input.addEventListener('change', () => {
            let v = parseFloat(input.value);
            if (min !== undefined && v < min) v = min;
            if (step % 1 === 0) v = Math.round(v);
            input.value = formatVal(v);
            onChange(v);
        });
        
        // Prevent key events from bubbling up and moving the camera while typing
        input.addEventListener('keydown', (e) => e.stopPropagation());
        input.addEventListener('keyup', (e) => e.stopPropagation());

        let isDragging = false;
        let startX = 0;
        let startVal = 0;

        const onMouseMove = (e: MouseEvent) => {
            if (!isDragging) return;
            const deltaX = e.clientX - startX;
            if (Math.abs(deltaX) > 2) {
                // It is a real drag. Prevent selection.
                input.blur(); // Remove focus to prevent text cursor interfering
                document.body.style.userSelect = 'none';
                document.body.style.cursor = 'ew-resize';
                
                let multiplier = step % 1 === 0 ? 0.5 : 0.02; // Default drag speed
                if (e.shiftKey) multiplier *= 10;
                if (e.altKey) multiplier *= 0.1;

                let newVal = startVal + deltaX * multiplier;
                if (min !== undefined && newVal < min) newVal = min;
                if (step % 1 === 0) newVal = Math.round(newVal);
                input.value = formatVal(newVal);
                onChange(newVal);
            }
        };

        const onMouseUp = () => {
            if (isDragging) {
                isDragging = false;
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                document.body.style.cursor = ''; // Restore global cursor
                document.body.style.userSelect = ''; // Restore text selection
            }
        };

        input.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return; // Only left click
            isDragging = true;
            startX = e.clientX;
            startVal = parseFloat(input.value) || 0;
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        return input;
    }

    private getComponentProperties(comp: Component): {key: string, initialValue: any, readonly?: boolean, min?: number, step?: number, options?: any[], getVal?: () => any, setVal?: (v: any) => void, getArrVal?: (i: number) => any, setArrVal?: (i: number, v: any) => void}[] {
        const props: any[] = [];
        if (!(comp as any).hideEnableInUI) {
            props.push({ 
                key: 'enabled', 
                initialValue: comp.enabled, 
                getVal: () => comp.enabled,
                setVal: (v: boolean) => comp.enabled = v
            });
        }

        if (comp instanceof MeshRenderer) {
            const mr = comp as MeshRenderer;
            if (mr.mesh) {
                props.push({ key: 'primitives', initialValue: mr.mesh.primitives.length, readonly: true });
                const totalIndices = mr.mesh.primitives.reduce((sum, p) => sum + p.numIndices, 0);
                props.push({ key: 'indices', initialValue: totalIndices.toLocaleString(), readonly: true });
            } else {
                props.push({ key: 'mesh', initialValue: 'null', readonly: true });
            }
            props.push({ key: 'gpu bound', initialValue: mr.modelBindGroup ? 'yes' : 'no', readonly: true });
        } else {
            // Generic property extraction for other components
            const descriptors = Object.getOwnPropertyDescriptors(comp);
            for (const [key, desc] of Object.entries(descriptors)) {
                if (key === 'entity' || key === 'enabled' || key === 'hideEnableInUI') continue;
                
                const val = desc.get ? desc.get.call(comp) : desc.value;
                if (typeof val === 'function') continue;
                

                const uiOptions = comp.getUIOptions ? comp.getUIOptions() : {};
                const options = uiOptions[key];

                if (options && Array.isArray(options)) {
                    props.push({
                        key,
                        initialValue: val,
                        options,
                        getVal: () => (comp as any)[key],
                        setVal: (v: any) => (comp as any)[key] = v
                    });
                } else if (typeof val === 'number' || typeof val === 'boolean') {
                    let min = (key === 'intensity' || key.toLowerCase().includes('color')) ? 0 : undefined;
                    let step = undefined;
                    if (options && typeof options === 'object' && !Array.isArray(options)) {
                        if (options.min !== undefined) min = options.min;
                        if (options.step !== undefined) step = options.step;
                    }
                    props.push({
                        key,
                        initialValue: val,
                        min,
                        step,
                        getVal: () => (comp as any)[key],
                        setVal: (v: any) => (comp as any)[key] = v
                    });
                } else if (val === undefined || val === null) {
                    props.push({ key, initialValue: 'null', readonly: true });
                } else if (Array.isArray(val) && val.every(v => typeof v === 'number')) {
                    props.push({
                        key,
                        initialValue: val,
                        readonly: false,
                        min: (key === 'intensity' || key.toLowerCase().includes('color')) ? 0 : undefined,
                        getArrVal: (i: number) => (comp as any)[key][i],
                        setArrVal: (i: number, v: any) => (comp as any)[key][i] = v
                    });
                } else if (Array.isArray(val)) {
                    props.push({ key, initialValue: `[${val.map((v: any) => String(v)).join(', ')}]`, readonly: true });
                } else if (typeof desc.value === 'string') {
                    props.push({ key, initialValue: desc.value, readonly: true });
                } else if (typeof desc.value === 'object') {
                    props.push({ key, initialValue: desc.value.constructor?.name ?? 'Object', readonly: true });
                }
            }
        }

        return props;
    }

    public destroy() {
        cancelAnimationFrame(this.updateRAF);
        this.panel.remove();
        this.toggleBtn.remove();
        const style = document.getElementById('scene-tree-styles');
        if (style) style.remove();
    }
}
