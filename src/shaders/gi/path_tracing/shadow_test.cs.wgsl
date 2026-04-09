// shadow_test.cs.wgsl
// Wavefront Path Tracing — Pass 4: Shadow Ray Any-Hit Test

@group(0) @binding(0) var<uniform>             pt:            PTUniforms;
@group(0) @binding(1) var<storage, read_write>  shadow_buffer: array<ShadowRay>;
@group(0) @binding(2) var<storage, read_write>  accum_buffer:  array<vec4f>;
@group(0) @binding(3) var<storage, read>        bvh_nodes:     array<BVHNode>;
@group(0) @binding(4) var<storage, read>        bvh_pos:       array<vec4f>;
@group(0) @binding(5) var<storage, read>        bvh_indices:   array<vec4u>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let shadow = shadow_buffer[pixel_id];
    if (shadow.shadow_active == 0u) { return; }

    var bvh_ray: Ray;
    bvh_ray.origin    = shadow.origin;
    bvh_ray.direction = shadow.direction;

    let result   = bvhIntersectFirstHit(&bvh_nodes, &bvh_pos, &bvh_indices, bvh_ray);
    let occluded = result.didHit && result.dist < (shadow.max_dist - 0.01);

    if (!occluded) {
        let prev = accum_buffer[pixel_id];
        accum_buffer[pixel_id] = vec4f(prev.xyz + shadow.Li, prev.w);
    }

    shadow_buffer[pixel_id].shadow_active = 0u;
}
