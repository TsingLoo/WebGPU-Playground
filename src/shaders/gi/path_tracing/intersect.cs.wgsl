// intersect.cs.wgsl
// Wavefront Path Tracing — Pass 2: BVH Closest-Hit Intersection

@group(0) @binding(0) var<uniform>             pt:          PTUniforms;
@group(0) @binding(1) var<storage, read_write>  ray_buffer:  array<PTRay>;
@group(0) @binding(2) var<storage, read_write>  hit_buffer:  array<HitRecord>;
@group(0) @binding(3) var<storage, read>        bvh_nodes:   array<BVHNode>;
@group(0) @binding(4) var<storage, read>        bvh_pos:     array<vec4f>;
@group(0) @binding(5) var<storage, read>        bvh_indices: array<vec4u>;
@group(0) @binding(6) var<storage, read>        bvh_uvs:     array<vec4f>;

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
        rec.normal  = result.normal;
        rec.side    = result.side;

        let idx  = result.indices;
        let bary = result.barycoord;

        let p0 = bvh_pos[idx.x].xyz;
        let p1 = bvh_pos[idx.y].xyz;
        let p2 = bvh_pos[idx.z].xyz;
        rec.pos = bary.x * p0 + bary.y * p1 + bary.z * p2;

        let uv0 = bvh_uvs[idx.x].xy;
        let uv1 = bvh_uvs[idx.y].xy;
        let uv2 = bvh_uvs[idx.z].xy;
        rec.uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    }

    hit_buffer[pixel_id] = rec;
}
