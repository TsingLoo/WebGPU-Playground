// bvh.wgsl
// WGSL implementation of Quadtree BVH (BVH4) traversal

const BVH_STACK_DEPTH = 24u;
const INFINITY = 1e20;
// Reduced from 1e-5 to 1e-8 to allow intersection with very small triangles 
// (e.g. detailed glTF models like the chess pieces) without returning false-negative misses.
const TRI_INTERSECT_EPSILON = 1e-8;

struct Ray {
    origin: vec3f,
    direction: vec3f,
    invDirection: vec3f,
    dirSign: vec3u,
};

// 128-byte BVH4 Node (8 x vec4f)
struct BVH4Node {
    minX: vec4f, // 4 children's Min X
    minY: vec4f, // 4 children's Min Y
    minZ: vec4f, // 4 children's Min Z
    maxX: vec4f, // 4 children's Max X
    maxY: vec4f, // 4 children's Max Y
    maxZ: vec4f, // 4 children's Max Z
    data0: vec4u, // children indices or triangle offsets
    data1: vec4u, // triangle counts (0 means internal node, 0xFFFFFFFF means empty)
};

struct IntersectionResult {
    didHit: bool,
    indices: vec4u, // xyz = vertex indices, w = matId
    normal: vec3f,
    barycoord: vec3f,
    side: f32,
    dist: f32,
};

// Returns distances to 4 bounds. Distances are INFINITY if missed.
fn intersectsBounds4(ray: Ray, node: BVH4Node) -> vec4f {
    let invDir = ray.invDirection;
    let negOriginInvDir = -ray.origin * invDir;
    
    let tMinPlaneX = node.minX * invDir.x + vec4f(negOriginInvDir.x);
    let tMaxPlaneX = node.maxX * invDir.x + vec4f(negOriginInvDir.x);
    
    let tMinPlaneY = node.minY * invDir.y + vec4f(negOriginInvDir.y);
    let tMaxPlaneY = node.maxY * invDir.y + vec4f(negOriginInvDir.y);
    
    let tMinPlaneZ = node.minZ * invDir.z + vec4f(negOriginInvDir.z);
    let tMaxPlaneZ = node.maxZ * invDir.z + vec4f(negOriginInvDir.z);
    
    let tMinHitX = min(tMinPlaneX, tMaxPlaneX);
    let tMaxHitX = max(tMinPlaneX, tMaxPlaneX);
    
    let tMinHitY = min(tMinPlaneY, tMaxPlaneY);
    let tMaxHitY = max(tMinPlaneY, tMaxPlaneY);
    
    let tMinHitZ = min(tMinPlaneZ, tMaxPlaneZ);
    let tMaxHitZ = max(tMinPlaneZ, tMaxPlaneZ);
    
    let t0 = max(max(tMinHitX, tMinHitY), tMinHitZ);
    let t1 = min(min(tMaxHitX, tMaxHitY), tMaxHitZ);
    
    let hitDist = max(t0, vec4f(0.0));
    let validHit = (t1 >= hitDist);
    
    return select(vec4f(INFINITY), hitDist, validHit);
}

fn intersectsTriangle(ray: Ray, a: vec3f, b: vec3f, c: vec3f) -> IntersectionResult {
    var result: IntersectionResult;
    result.didHit = false;

    let edge1 = b - a;
    let edge2 = c - a;
    let n = cross(edge1, edge2);

    let det = -dot(ray.direction, n);
    if (abs(det) < TRI_INTERSECT_EPSILON) { return result; }

    let invdet = 1.0 / det;
    let AO = ray.origin - a;
    let DAO = cross(AO, ray.direction);

    let u = dot(edge2, DAO) * invdet;
    let v = -dot(edge1, DAO) * invdet;
    let t = dot(AO, n) * invdet;
    let w = 1.0 - u - v;

    if (u < -TRI_INTERSECT_EPSILON || v < -TRI_INTERSECT_EPSILON || w < -TRI_INTERSECT_EPSILON || t < TRI_INTERSECT_EPSILON) {
        return result;
    }

    result.didHit = true;
    result.barycoord = vec3f(w, u, v);
    result.dist = t;
    result.side = sign(det);
    result.normal = result.side * normalize(n);

    return result;
}

fn intersectTriangles(
    bvh_position: ptr<storage, array<vec4f>, read>,
    bvh_index: ptr<storage, array<vec4u>, read>,
    offset: u32, count: u32, ray: Ray
) -> IntersectionResult {
    var closestResult: IntersectionResult;
    closestResult.didHit = false;
    closestResult.dist = INFINITY;

    // Up to 4 triangles are tested per leaf loop usually, but count could be any number.
    // In three-mesh-bvh maxLeafSize is configurable, here we loop through all in the leaf.
    for (var i = 0u; i < count; i++) {
        let indices = bvh_index[offset + i];
        let triResult = intersectsTriangle(ray, bvh_position[indices.x].xyz, bvh_position[indices.y].xyz, bvh_position[indices.z].xyz);
        if (triResult.didHit && triResult.dist < closestResult.dist) {
            closestResult = triResult;
            closestResult.indices = vec4u(indices.xyz, indices.w);
        }
    }
    return closestResult;
}

var<private> bvh_stack: array<u32, BVH_STACK_DEPTH>;

fn bvhIntersectFirstHit(
    bvh: ptr<storage, array<BVH4Node>, read>,
    bvh_position: ptr<storage, array<vec4f>, read>,
    bvh_index: ptr<storage, array<vec4u>, read>,
    ray: Ray
) -> IntersectionResult {
    var pointer = 0;
    bvh_stack[0] = 0u;

    var bestHit: IntersectionResult;
    bestHit.didHit = false;
    bestHit.dist = INFINITY;

    loop {
        if (pointer < 0 || pointer >= i32(BVH_STACK_DEPTH)) { break; }

        let currNodeIndex = bvh_stack[pointer];
        pointer = pointer - 1;

        let node = bvh[currNodeIndex];
        let hitDists = intersectsBounds4(ray, node);

        var d_arr = array<f32, 4>(hitDists.x, hitDists.y, hitDists.z, hitDists.w);
        var idx_arr = array<u32, 4>(0u, 1u, 2u, 3u);

        // Sorting network for 4 elements (descending order).
        // Smaller distances are moved to the right.
        if (d_arr[0] < d_arr[1]) { let tmp_d = d_arr[0]; d_arr[0] = d_arr[1]; d_arr[1] = tmp_d; let tmp_i = idx_arr[0]; idx_arr[0] = idx_arr[1]; idx_arr[1] = tmp_i; }
        if (d_arr[2] < d_arr[3]) { let tmp_d = d_arr[2]; d_arr[2] = d_arr[3]; d_arr[3] = tmp_d; let tmp_i = idx_arr[2]; idx_arr[2] = idx_arr[3]; idx_arr[3] = tmp_i; }
        if (d_arr[0] < d_arr[2]) { let tmp_d = d_arr[0]; d_arr[0] = d_arr[2]; d_arr[2] = tmp_d; let tmp_i = idx_arr[0]; idx_arr[0] = idx_arr[2]; idx_arr[2] = tmp_i; }
        if (d_arr[1] < d_arr[3]) { let tmp_d = d_arr[1]; d_arr[1] = d_arr[3]; d_arr[3] = tmp_d; let tmp_i = idx_arr[1]; idx_arr[1] = idx_arr[3]; idx_arr[3] = tmp_i; }
        if (d_arr[1] < d_arr[2]) { let tmp_d = d_arr[1]; d_arr[1] = d_arr[2]; d_arr[2] = tmp_d; let tmp_i = idx_arr[1]; idx_arr[1] = idx_arr[2]; idx_arr[2] = tmp_i; }

        let data0_arr = array<u32, 4>(node.data0.x, node.data0.y, node.data0.z, node.data0.w);
        let data1_arr = array<u32, 4>(node.data1.x, node.data1.y, node.data1.z, node.data1.w);

        for (var k = 0u; k < 4u; k++) {
            let childIdx = idx_arr[k];
            let dist = d_arr[k];

            if (dist >= bestHit.dist) { continue; } // Missed or further than bestHit
            
            let data1 = data1_arr[childIdx];
            if (data1 == 0xFFFFFFFFu) { continue; } // Empty child

            let data0 = data0_arr[childIdx];
            
            if (data1 == 0u) {
                // Internal Node: push to stack
                if (pointer >= i32(BVH_STACK_DEPTH) - 1) { continue; }
                pointer++;
                bvh_stack[pointer] = data0;
            } else {
                // Leaf Node: intersect triangles
                let count = data1;
                let offset = data0;
                let localHit = intersectTriangles(bvh_position, bvh_index, offset, count, ray);
                
                if (localHit.didHit && localHit.dist < bestHit.dist) {
                    bestHit = localHit;
                }
            }
        }
    }

    return bestHit;
}
