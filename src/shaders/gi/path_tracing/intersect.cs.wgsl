// intersect.cs.wgsl
// Wavefront Path Tracing — Pass 2: BVH Closest-Hit Intersection
// Now interpolates per-vertex normals and tangents for smooth shading + normal mapping.

@group(0) @binding(0) var<uniform>             pt:           PTUniforms;
@group(0) @binding(1) var<storage, read_write>  ray_buffer:   array<PTRay>;
@group(0) @binding(2) var<storage, read_write>  hit_buffer:   array<HitRecord>;
@group(0) @binding(3) var<storage, read>        bvh_nodes:    array<BVHNode>;
@group(0) @binding(4) var<storage, read>        bvh_pos:      array<vec4f>;
@group(0) @binding(5) var<storage, read>        bvh_indices:  array<vec4u>;
@group(0) @binding(6) var<storage, read>        bvh_uvs:      array<vec4f>;
@group(0) @binding(7) var<storage, read>        bvh_normals:  array<vec4f>;
@group(0) @binding(8) var<storage, read>        bvh_tangents: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    var rec: HitRecord;
    rec.did_hit = 0u;

    let pt_ray = ray_buffer[pixel_id];
    if (pt_ray.ray_active == 0u) {
        hit_buffer[pixel_id] = rec;
        return;
    }

    var bvh_ray: Ray;
    bvh_ray.origin    = pt_ray.origin;
    bvh_ray.direction = pt_ray.direction;

    let result = bvhIntersectFirstHit(&bvh_nodes, &bvh_pos, &bvh_indices, bvh_ray);

    if (result.didHit) {
        rec.did_hit = 1u;
        rec.dist    = result.dist;
        rec.mat_id  = result.indices.w;
        rec.side    = result.side;

        let idx  = result.indices;
        let bary = result.barycoord;

        // Interpolate position
        let p0 = bvh_pos[idx.x].xyz;
        let p1 = bvh_pos[idx.y].xyz;
        let p2 = bvh_pos[idx.z].xyz;
        rec.pos = bary.x * p0 + bary.y * p1 + bary.z * p2;

        // Interpolate UV
        let uv0 = bvh_uvs[idx.x].xy;
        let uv1 = bvh_uvs[idx.y].xy;
        let uv2 = bvh_uvs[idx.z].xy;
        rec.uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

        // Store geometric normal (from cross product, already flipped by BVH for face direction)
        rec.geom_normal = result.normal;

        // Interpolate smooth shading normal from per-vertex normals
        let n0 = bvh_normals[idx.x].xyz;
        let n1 = bvh_normals[idx.y].xyz;
        let n2 = bvh_normals[idx.z].xyz;
        var smooth_normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
        // Flip smooth normal to match geometric normal direction (ensure it faces the ray)
        if (dot(smooth_normal, pt_ray.direction) > 0.0) {
            smooth_normal = -smooth_normal;
        }
        rec.normal = smooth_normal;

        // Interpolate tangent from per-vertex tangents (xyz=tangent dir, w=handedness)
        let t0 = bvh_tangents[idx.x];
        let t1 = bvh_tangents[idx.y];
        let t2 = bvh_tangents[idx.z];
        let interp_tangent_dir = normalize(bary.x * t0.xyz + bary.y * t1.xyz + bary.z * t2.xyz);
        // Average handedness (should be the same for all verts of a triangle)
        let handedness = t0.w;
        rec.tangent = vec4f(interp_tangent_dir, handedness);
    }

    hit_buffer[pixel_id] = rec;
}
