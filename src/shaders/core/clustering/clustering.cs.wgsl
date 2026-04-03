// TODO-2: implement the light clustering compute shader
@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read_write> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read_write> globalLightIndices: LightIndexList;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;
@group(${bindGroup_scene}) @binding(5) var hizTexture: texture_2d<f32>;

const MAX_LIGHTS_PER_CLUSTER = ${maxLightsPerCluster}u;
var<workgroup> local_light_indices: array<u32, MAX_LIGHTS_PER_CLUSTER>;
// ------------------------------------
// Calculating cluster bounds:
// ------------------------------------
// For each cluster (X, Y, Z):
//     - Calculate the screen-space bounds for this cluster in 2D (XY).
//     - Calculate the depth bounds for this cluster in Z (near and far planes).
//     - Convert these screen and depth bounds into view-space coordinates.
//     - Store the computed bounding box (AABB) for the cluster.

var<workgroup> local_light_count: atomic<u32>;
var<workgroup> cluster_aabb_min: vec3f;
var<workgroup> cluster_aabb_max: vec3f;
var<workgroup> cluster_has_geometry: u32;

// ------------------------------------
// Assigning lights to clusters:
// ------------------------------------
// For each cluster:
//     - Initialize a counter for the number of lights in this cluster.

//     For each light:
//         - Check if the light intersects with the cluster’s bounding box (AABB).
//         - If it does, add the light to the cluster's light list.
//         - Stop adding lights if the maximum number of lights is reached.

//     - Store the number of lights assigned to this cluster.

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    if (local_idx == 0u) {
        atomicStore(&local_light_count, 0u);

        let num_clusters_X = clusterSet.num_clusters_X;
        let num_clusters_Y = clusterSet.num_clusters_Y;
        let num_clusters_Z = clusterSet.num_clusters_Z;
        let num_clusters = num_clusters_X * num_clusters_Y * num_clusters_Z;

        let num_lights = lightSet.numLights;

        // size of cluster in NDC space
        let tileNDCX = 2.0 / f32(num_clusters_X);
        let tileNDCY = 2.0 / f32(num_clusters_Y);

        // aabb x y bounds in NDC space
        let xMin = -1.0 + f32(group_id.x) * tileNDCX;
        let xMax = -1.0 + f32(group_id.x + 1u) * tileNDCX;
        let yMin = -1.0 + f32(group_id.y) * tileNDCY;
        let yMax = -1.0 + f32(group_id.y + 1u) * tileNDCY;

        let near = f32(camera.near_plane);
        let far = f32(camera.far_plane);
    

        let exponent_min = f32(group_id.z) / f32(num_clusters_Z);
        let z_view_min = - near * pow(far / near, exponent_min);

        let exponent_max = f32(group_id.z + 1u) / f32(num_clusters_Z);
        let z_view_max = - near * pow(far / near, exponent_max);

        let P = camera.proj_mat;
        let p22 = P[2][2];
        let p23 = P[2][3];
        let p32 = P[3][2];
        let p33 = P[3][3];
        let zNDCMin = ((p22 * z_view_min) + p32) / ((p23 * z_view_min) + p33);
        let zNDCMax = ((p22 * z_view_max) + p32) / ((p23 * z_view_max) + p33);

        let invP = camera.inv_proj_mat;
        var ndcCorners = array<vec4f, 8>(
            vec4f(xMin, yMin, zNDCMin, 1.0), vec4f(xMax, yMin, zNDCMin, 1.0),
            vec4f(xMin, yMax, zNDCMin, 1.0), vec4f(xMax, yMax, zNDCMin, 1.0),
            vec4f(xMin, yMin, zNDCMax, 1.0), vec4f(xMax, yMin, zNDCMax, 1.0),
            vec4f(xMin, yMax, zNDCMax, 1.0), vec4f(xMax, yMax, zNDCMax, 1.0)
        );
           
        // P_view = invP * P_clip
        // projectMatrix * P_view = P_clip
        // p_clip.xyzw / p_clip.w = ndcPos
        // projectMatrix * P_view = ndcPos * p_clip.w
        // P_view = invP * (ndcPos * p_clip.w)

        // P_view = invP * ndcPos * p_clip.w
        // P_view.w = (invP * ndcPos).w * p_clip.w
        // P_clip.w = P_view.w / (invP * ndcPos).w

        // P_view = invP * (ndcPos * (1.0 / (invP * ndcPos).w))

        var viewCorners: array<vec3f, 8>;
        for (var i = 0u; i < 8u; i++) {
            var c = invP * ndcCorners[i];
            c = c / c.w;
            viewCorners[i] = c.xyz;
        }

        var aabbMin = viewCorners[0];
        var aabbMax = viewCorners[0];
        for (var i = 1; i < 8; i++) {
            aabbMin = min(aabbMin, viewCorners[i]);
            aabbMax = max(aabbMax, viewCorners[i]);
        }
        
        cluster_aabb_min = aabbMin;
        cluster_aabb_max = aabbMax;

        // --- Hi-Z Cluster Culling ---
        let dim = vec2f(textureDimensions(hizTexture, 0));
        let uvMin = clamp(vec2f(xMin, -yMax) * 0.5 + vec2f(0.5, 0.5), vec2f(0.0), vec2f(1.0));
        let uvMax = clamp(vec2f(xMax, -yMin) * 0.5 + vec2f(0.5, 0.5), vec2f(0.0), vec2f(1.0)); 
        
        let pxMin = min(uvMin, uvMax) * dim;
        let pxMax = max(uvMin, uvMax) * dim;
        let size = pxMax - pxMin;
        
        let max_size = max(max(size.x, size.y), 1.0);
        let level = u32(clamp(log2(max_size), 0.0, 10.0));
        
        let p0 = vec2i(pxMin) >> vec2u(level);
        let p1 = vec2i(pxMax) >> vec2u(level);
        let sDims = vec2i(textureDimensions(hizTexture, level));
        
        let d00 = textureLoad(hizTexture, clamp(vec2i(p0.x, p0.y), vec2i(0), sDims - 1), level).rg;
        let d10 = textureLoad(hizTexture, clamp(vec2i(p1.x, p0.y), vec2i(0), sDims - 1), level).rg;
        let d01 = textureLoad(hizTexture, clamp(vec2i(p0.x, p1.y), vec2i(0), sDims - 1), level).rg;
        let d11 = textureLoad(hizTexture, clamp(vec2i(p1.x, p1.y), vec2i(0), sDims - 1), level).rg;
        
        let tile_ndc_far = min(min(d00.r, d10.r), min(d01.r, d11.r));
        let tile_ndc_near = max(max(d00.g, d10.g), max(d01.g, d11.g));
        
        var tile_view_far = -p32 / (p22 + tile_ndc_far);
        if (tile_ndc_far <= 0.000001) {
            tile_view_far = -1000000.0;
        }
        let tile_view_near = -p32 / (p22 + tile_ndc_near);
        
        let has_geom = (z_view_max >= tile_view_far) && (z_view_min <= tile_view_near);
        cluster_has_geometry = select(0u, 1u, has_geom);
    }

    workgroupBarrier();

    if (cluster_has_geometry == 1u) {
        let radius = f32(${lightRadius});
        let V = camera.view_mat;

        let num_lights = lightSet.numLights;
        for (var li = local_idx; li < num_lights; li += 64u) {
            
            let light_view_pos = (V * vec4f(lightSet.lights[li].pos, 1.0)).xyz;

            let sMin = light_view_pos - vec3f(radius);
            let sMax = light_view_pos + vec3f(radius);
            let intersectMin = max(cluster_aabb_min, sMin);
            let intersectMax = min(cluster_aabb_max, sMax);
            let overlaps = all(intersectMin <= intersectMax);
            
            if (overlaps) {
                let list_idx = atomicAdd(&local_light_count, 1u);
                if (list_idx < MAX_LIGHTS_PER_CLUSTER) {
                    local_light_indices[list_idx] = li;
                }
            }
        }
    }

    workgroupBarrier();

    if (local_idx == 0u) {

        let num_clusters_X = clusterSet.num_clusters_X;
        let num_clusters_Y = clusterSet.num_clusters_Y;
        let num_clusters_Z = clusterSet.num_clusters_Z;
        let num_clusters = num_clusters_X * num_clusters_Y * num_clusters_Z;

        let cluster_index = group_id.z * (num_clusters_X * num_clusters_Y) +
                              group_id.y * num_clusters_X +
                              group_id.x;

        if (cluster_has_geometry == 1u) {
            let final_count = atomicLoad(&local_light_count);
            let count_to_write = min(final_count, MAX_LIGHTS_PER_CLUSTER);

            let global_offset = atomicAdd(&globalLightIndices.counter, count_to_write);

            for (var i = 0u; i < count_to_write; i += 1u) {
                globalLightIndices.indices[global_offset + i] = local_light_indices[i];
            }

            tileOffsets[cluster_index].offset = global_offset;
            tileOffsets[cluster_index].count = count_to_write;
        } else {
            tileOffsets[cluster_index].offset = 0u;
            tileOffsets[cluster_index].count = 0u;
        }
    }
}