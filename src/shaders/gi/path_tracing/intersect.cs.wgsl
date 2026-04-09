// intersect.cs.wgsl
// Wavefront Path Tracing — Pass 2: BVH Closest-Hit Intersection
// Stores compact hit record (vertex indices + bary); shade reconstructs full data.

@group(0) @binding(0) var<uniform>             pt:           PTUniforms;
@group(0) @binding(1) var<storage, read_write>  ray_buffer:   array<PTRay>;
@group(0) @binding(2) var<storage, read_write>  hit_buffer:   array<HitRecord>;
@group(0) @binding(3) var<storage, read>        bvh_nodes:    array<BVHNode>;
@group(0) @binding(4) var<storage, read>        bvh_pos:      array<vec4f>;
@group(0) @binding(5) var<storage, read>        bvh_indices:  array<vec4u>;

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
        rec.bary    = result.barycoord.xy;  // z = 1 - x - y
        rec.idx0    = result.indices.x;
        rec.idx1    = result.indices.y;
        rec.idx2    = result.indices.z;
    }

    hit_buffer[pixel_id] = rec;
}
