// intersect.cs.wgsl
// Wavefront Path Tracing — Pass 2: BVH Closest-Hit Intersection
// With alpha-cutout transparency: skips transparent surfaces internally
// so that shade always receives opaque (or no) hits.

@group(0) @binding(0) var<uniform>             pt:           PTUniforms;
@group(0) @binding(1) var<storage, read_write>  ray_buffer:   array<PTRay>;
@group(0) @binding(2) var<storage, read_write>  hit_buffer:   array<HitRecord>;
@group(0) @binding(3) var<storage, read>        bvh_nodes:    array<BVHNode>;
@group(0) @binding(4) var<storage, read>        bvh_pos:      array<vec4f>;
@group(0) @binding(5) var<storage, read>        bvh_indices:  array<vec4u>;
@group(0) @binding(6) var<storage, read>        materials:    array<vec4f>;
@group(0) @binding(7) var                       base_color_tex: texture_2d_array<f32>;
@group(0) @binding(8) var                       tex_sampler:  sampler;
@group(0) @binding(9) var<storage, read>        bvh_uvs:      array<vec4f>;

const MAX_ALPHA_SKIP = 16u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = u32(f32(pt.width)  * pt.pixel_scale);
    let render_height = u32(f32(pt.height) * pt.pixel_scale);
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    var rec: HitRecord;
    rec.did_hit = 0u;

    var pt_ray = ray_buffer[pixel_id];
    if (pt_ray.ray_active == 0u) {
        hit_buffer[pixel_id] = rec;
        return;
    }

    var bvh_ray: Ray;
    bvh_ray.origin    = pt_ray.origin;
    bvh_ray.direction = pt_ray.direction;

    // Loop: find closest hit, check alpha, skip transparent surfaces
    for (var skip = 0u; skip < MAX_ALPHA_SKIP; skip++) {
        let result = bvhIntersectFirstHit(&bvh_nodes, &bvh_pos, &bvh_indices, bvh_ray);

        if (!result.didHit) {
            // No hit at all — miss
            break;
        }

        // Check if this hit is alpha-transparent
        let mat_id = result.indices.w;
        let mat = unpackPTMaterial(&materials, mat_id);

        var final_alpha = mat.alpha;
        if (mat.tex_layer >= 0) {
            // Reconstruct UV from barycentric
            let bw = result.barycoord; // (w, u, v) from intersectsTriangle
            let uv0 = bvh_uvs[result.indices.x].xy;
            let uv1 = bvh_uvs[result.indices.y].xy;
            let uv2 = bvh_uvs[result.indices.z].xy;
            let hit_uv = bw.x * uv0 + bw.y * uv1 + bw.z * uv2;

            let tex_col = textureSampleLevel(base_color_tex, tex_sampler, hit_uv, mat.tex_layer, 0.0);
            final_alpha *= tex_col.w;
        }

        if (final_alpha >= 0.5) {
            // Opaque hit — accept it
            rec.did_hit = 1u;
            rec.dist    = result.dist;
            rec.mat_id  = result.indices.w;
            rec.side    = result.side;
            rec.bary    = result.barycoord.xy;
            rec.idx0    = result.indices.x;
            rec.idx1    = result.indices.y;
            rec.idx2    = result.indices.z;
            break;
        }

        // Transparent hit — advance ray past this surface and re-traverse
        let hit_pos = bvh_ray.origin + bvh_ray.direction * result.dist;
        bvh_ray.origin = hit_pos + bvh_ray.direction * 0.001;

        // Also update the ray buffer origin so shade uses the advanced position
        pt_ray.origin = bvh_ray.origin;
    }

    // Write back the (potentially advanced) ray origin
    ray_buffer[pixel_id] = pt_ray;
    hit_buffer[pixel_id] = rec;
}
