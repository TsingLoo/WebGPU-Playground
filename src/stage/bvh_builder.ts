import { BufferAttribute, BufferGeometry, Box3 } from 'three';
import { MeshBVH, MeshBVHOptions } from 'three-mesh-bvh';
import { Entity } from '../engine/Entity';
import { MeshRenderer } from '../engine/components/MeshRenderer';
import { device } from '../renderer';

export class BVHData {
    nodeBuffer!: GPUBuffer;
    positionBuffer!: GPUBuffer;
    indexBuffer!: GPUBuffer;
    uvBuffer!: GPUBuffer;      // per-vertex UV (vec4f: xy = UV, zw = pad)
    normalBuffer!: GPUBuffer;  // per-vertex smooth normal (vec4f: xyz = normal, w = 0)
    tangentBuffer!: GPUBuffer; // per-vertex tangent (vec4f: xyz = tangent, w = handedness)
    triangleMaterialBuffer!: GPUBuffer; // per-triangle material index
    triangleCount: number = 0;
    boundingBoxMin: [number, number, number] = [-1, -1, -1];
    boundingBoxMax: [number, number, number] = [1, 1, 1];
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
    const normals = new Float32Array(totalVerts * 3);
    const tangents = new Float32Array(totalVerts * 4);
    const uvs = new Float32Array(totalVerts * 2);
    const indices = new Uint32Array(totalTris * 3);
    const vertexMaterials = new Uint32Array(totalVerts); // Map vertex -> material
    
    let triOffset = 0;
    let vertOffset = 0;
    let normalOffset = 0;
    let tangentOffset = 0;
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
                const primNormals = prim.cpuNormals;
                const primTangents = prim.cpuTangents;
                const mat = node.worldTransform;
                const materialId = prim.material.id;
                
                const baseVertIndex = vertOffset / 3;
                const numPrimVerts = pos.length / 3;
                
                // Compute normal matrix (inverse transpose of upper-left 3x3)
                // For Sponza with only uniform-ish transforms, transpose of inverse ~= mat itself
                // Proper inverse-transpose of upper-left 3x3:
                const m00 = mat[0], m01 = mat[1], m02 = mat[2];
                const m10 = mat[4], m11 = mat[5], m12 = mat[6];
                const m20 = mat[8], m21 = mat[9], m22 = mat[10];
                
                // Add transformed vertices + normals + tangents + UVs
                for (let i = 0; i < pos.length; i += 3) {
                    const x = pos[i], y = pos[i+1], z = pos[i+2];
                    // Transform position by world matrix
                    positions[vertOffset++] = x*mat[0] + y*mat[4] + z*mat[8] + mat[12];
                    positions[vertOffset++] = x*mat[1] + y*mat[5] + z*mat[9] + mat[13];
                    positions[vertOffset++] = x*mat[2] + y*mat[6] + z*mat[10] + mat[14];
                    vertexMaterials[baseVertIndex + i/3] = materialId;
                    
                    // Transform normal by upper-left 3x3 (for uniform scale, same as world matrix)
                    if (primNormals) {
                        const nx = primNormals[i], ny = primNormals[i+1], nz = primNormals[i+2];
                        let tnx = nx*m00 + ny*m10 + nz*m20;
                        let tny = nx*m01 + ny*m11 + nz*m21;
                        let tnz = nx*m02 + ny*m12 + nz*m22;
                        const len = Math.sqrt(tnx*tnx + tny*tny + tnz*tnz);
                        if (len > 0.0001) { tnx /= len; tny /= len; tnz /= len; }
                        normals[normalOffset++] = tnx;
                        normals[normalOffset++] = tny;
                        normals[normalOffset++] = tnz;
                    } else {
                        normals[normalOffset++] = 0;
                        normals[normalOffset++] = 1;
                        normals[normalOffset++] = 0;
                    }
                }
                
                // Transform tangents by world matrix (direction only, preserve handedness)
                if (primTangents) {
                    for (let i = 0; i < numPrimVerts; i++) {
                        const tx = primTangents[i*4], ty = primTangents[i*4+1], tz = primTangents[i*4+2];
                        const tw = primTangents[i*4+3]; // handedness
                        let ttx = tx*m00 + ty*m10 + tz*m20;
                        let tty = tx*m01 + ty*m11 + tz*m21;
                        let ttz = tx*m02 + ty*m12 + tz*m22;
                        const len = Math.sqrt(ttx*ttx + tty*tty + ttz*ttz);
                        if (len > 0.0001) { ttx /= len; tty /= len; ttz /= len; }
                        tangents[tangentOffset++] = ttx;
                        tangents[tangentOffset++] = tty;
                        tangents[tangentOffset++] = ttz;
                        tangents[tangentOffset++] = tw;
                    }
                } else {
                    for (let i = 0; i < numPrimVerts; i++) {
                        tangents[tangentOffset++] = 1;
                        tangents[tangentOffset++] = 0;
                        tangents[tangentOffset++] = 0;
                        tangents[tangentOffset++] = 1;
                    }
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
    
    // Pack normals to vec4f (xyz = normal, w = 0)
    const wgpuNormals = new Float32Array(totalVerts * 4);
    for (let i = 0; i < totalVerts; i++) {
        wgpuNormals[i*4 + 0] = normals[i*3 + 0];
        wgpuNormals[i*4 + 1] = normals[i*3 + 1];
        wgpuNormals[i*4 + 2] = normals[i*3 + 2];
        wgpuNormals[i*4 + 3] = 0;
    }
    
    // Tangents are already vec4f (xyz = tangent, w = handedness)
    const wgpuTangents = new Float32Array(totalVerts * 4);
    wgpuTangents.set(tangents);
    
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
    const createAndUpload = (label: string, data: ArrayBuffer): GPUBuffer => {
        const buf = device.createBuffer({
            label, size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(buf, 0, data);
        return buf;
    };

    const bvhData = new BVHData();
    const boundingBox = bvh.getBoundingBox(new Box3());
    bvhData.boundingBoxMin = [boundingBox.min.x, boundingBox.min.y, boundingBox.min.z];
    bvhData.boundingBoxMax = [boundingBox.max.x, boundingBox.max.y, boundingBox.max.z];
    bvhData.nodeBuffer     = createAndUpload("BVH Node Buffer", bvhNodeBufferArray);
    bvhData.positionBuffer = createAndUpload("BVH Position Buffer", wgpuPositions.buffer);
    bvhData.uvBuffer       = createAndUpload("BVH UV Buffer", wgpuUVs.buffer);
    bvhData.normalBuffer   = createAndUpload("BVH Normal Buffer", wgpuNormals.buffer);
    bvhData.tangentBuffer  = createAndUpload("BVH Tangent Buffer", wgpuTangents.buffer);
    bvhData.indexBuffer    = createAndUpload("BVH Index Buffer", wgpuIndices.buffer);
    bvhData.triangleCount  = totalTris;
    console.log("BVH Generation Complete.");
    
    return bvhData;
}
