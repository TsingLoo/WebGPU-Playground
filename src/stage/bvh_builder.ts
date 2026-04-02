import { BufferAttribute, BufferGeometry } from 'three';
import { MeshBVH, MeshBVHOptions } from 'three-mesh-bvh';
import { Entity } from '../engine/Entity';
import { MeshRenderer } from '../engine/components/MeshRenderer';
import { device } from '../renderer';

export class BVHData {
    nodeBuffer!: GPUBuffer;
    positionBuffer!: GPUBuffer;
    indexBuffer!: GPUBuffer;
    uvBuffer!: GPUBuffer;  // per-vertex UV (vec4f: xy = UV, zw = pad)
    triangleMaterialBuffer!: GPUBuffer; // per-triangle material index
    triangleCount: number = 0;
}

export function buildBVHFromScene(sceneRoot: Entity): BVHData {
    console.log("Building BVH on CPU...");
    
    // 1. Gather all triangles into flat arrays
    let totalTris = 0;
    let totalVerts = 0;
    
    // Pre-calculate totals
    let nodes = [sceneRoot];
    while (nodes.length > 0) {
        let node = nodes.pop()!;
        const mr = node.getComponent(MeshRenderer);
        if (mr && mr.mesh) {
            for (let prim of mr.mesh.primitives) {
                if (prim.cpuPositions && prim.cpuIndices) {
                    totalTris += prim.numIndices / 3;
                    totalVerts += prim.cpuPositions.length / 3;
                }
            }
        }
        for (let child of node.children) nodes.push(child);
    }
    
    console.log(`Total Triangles for BVH: ${totalTris}`);
    
    const positions = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const indices = new Uint32Array(totalTris * 3);
    const vertexMaterials = new Uint32Array(totalVerts); // Map vertex -> material
    
    let triOffset = 0;
    let vertOffset = 0;
    let uvOffset = 0;
    
    nodes = [sceneRoot];
    while (nodes.length > 0) {
        let node = nodes.pop()!;
        const mr = node.getComponent(MeshRenderer);
        if (mr && mr.mesh) {
            for (let prim of mr.mesh.primitives) {
                if (!prim.cpuPositions || !prim.cpuIndices) continue;
                
                const pos = prim.cpuPositions;
                const ind = prim.cpuIndices;
                const primUVs = prim.cpuUVs;
                const mat = node.worldTransform;
                const materialId = prim.material.id;
                
                const baseVertIndex = vertOffset / 3;
                const numPrimVerts = pos.length / 3;
                
                // Add transformed vertices + UVs
                for (let i = 0; i < pos.length; i += 3) {
                    const x = pos[i], y = pos[i+1], z = pos[i+2];
                    positions[vertOffset++] = x*mat[0] + y*mat[4] + z*mat[8] + mat[12];
                    positions[vertOffset++] = x*mat[1] + y*mat[5] + z*mat[9] + mat[13];
                    positions[vertOffset++] = x*mat[2] + y*mat[6] + z*mat[10] + mat[14];
                    vertexMaterials[baseVertIndex + i/3] = materialId;
                }
                
                // Copy UVs (no transform needed)
                if (primUVs) {
                    for (let i = 0; i < numPrimVerts; i++) {
                        uvs[uvOffset++] = primUVs[i * 2];
                        uvs[uvOffset++] = primUVs[i * 2 + 1];
                    }
                } else {
                    for (let i = 0; i < numPrimVerts; i++) {
                        uvs[uvOffset++] = 0;
                        uvs[uvOffset++] = 0;
                    }
                }
                
                // Add indices
                for (let i = 0; i < prim.numIndices; i++) {
                    indices[triOffset * 3 + i] = baseVertIndex + ind[i];
                }
                
                triOffset += prim.numIndices / 3;
            }
        }
        for (let child of node.children) nodes.push(child);
    }
    
    // 2. Generate BVH
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    geometry.setIndex(new BufferAttribute(indices, 1));
    
    const bvhOptions: MeshBVHOptions = { verbose: false, maxLeafSize: 10 };
    const bvh = new MeshBVH(geometry, bvhOptions);
    const bvhRoots = (bvh as any)._roots; 
    const bvhNodeBufferArray: ArrayBuffer = bvhRoots[0]; // the 32-byte structs array
    const reorderedIndices = geometry.getIndex()!.array;
    
    // 3. Pack for WebGPU
    // array<vec4f> stride 16. Expand positions to vec4f.
    const wgpuPositions = new Float32Array(totalVerts * 4);
    for (let i = 0; i < totalVerts; i++) {
        wgpuPositions[i*4 + 0] = positions[i*3 + 0];
        wgpuPositions[i*4 + 1] = positions[i*3 + 1];
        wgpuPositions[i*4 + 2] = positions[i*3 + 2];
        wgpuPositions[i*4 + 3] = 0; 
    }
    
    // Pack UVs to vec4f (xy = UV, zw = 0)
    const wgpuUVs = new Float32Array(totalVerts * 4);
    for (let i = 0; i < totalVerts; i++) {
        wgpuUVs[i*4 + 0] = uvs[i*2 + 0];
        wgpuUVs[i*4 + 1] = uvs[i*2 + 1];
        wgpuUVs[i*4 + 2] = 0;
        wgpuUVs[i*4 + 3] = 0;
    }
    
    // array<vec4u> stride 16. Pack `(i0, i1, i2, matId)`.
    const wgpuIndices = new Uint32Array(totalTris * 4);
    for (let i = 0; i < totalTris; i++) {
        let idx0 = reorderedIndices[i*3 + 0];
        let idx1 = reorderedIndices[i*3 + 1];
        let idx2 = reorderedIndices[i*3 + 2];
        
        // Recover material from the first vertex of the triangle
        let matId = vertexMaterials[idx0];
        
        wgpuIndices[i*4 + 0] = idx0;
        wgpuIndices[i*4 + 1] = idx1;
        wgpuIndices[i*4 + 2] = idx2;
        wgpuIndices[i*4 + 3] = matId; // Pass matId implicitly here!
    }
    
    // 4. Create WebGPU Buffers
    const nodeGpuBuffer = device.createBuffer({
        label: "BVH Node Buffer",
        size: bvhNodeBufferArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(nodeGpuBuffer, 0, bvhNodeBufferArray);
    
    const posGpuBuffer = device.createBuffer({
        label: "BVH Position Buffer",
        size: wgpuPositions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(posGpuBuffer, 0, wgpuPositions);
    
    const uvGpuBuffer = device.createBuffer({
        label: "BVH UV Buffer",
        size: wgpuUVs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(uvGpuBuffer, 0, wgpuUVs);
    
    const indexGpuBuffer = device.createBuffer({
        label: "BVH Index Buffer",
        size: wgpuIndices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(indexGpuBuffer, 0, wgpuIndices);

    const bvhData = new BVHData();
    bvhData.nodeBuffer = nodeGpuBuffer;
    bvhData.positionBuffer = posGpuBuffer;
    bvhData.uvBuffer = uvGpuBuffer;
    bvhData.indexBuffer = indexGpuBuffer;
    bvhData.triangleCount = totalTris;
    console.log("BVH Generation Complete.");
    
    return bvhData;
}
