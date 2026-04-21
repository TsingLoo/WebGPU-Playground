// shadow_test.cs.wgsl
// Wavefront Path Tracing — Pass 4: Shadow Ray Any-Hit Test
// With alpha-cutout transparency support (e.g. leaf textures).

@group(0) @binding(0) var<uniform>             pt:            PTUniforms;
@group(0) @binding(1) var<storage, read_write>  shadow_buffer: array<ShadowRay>;
@group(0) @binding(2) var<storage, read_write>  accum_buffer:  array<vec4f>;
@group(0) @binding(3) var<storage, read>        bvh_nodes:     array<BVH4Node>;
@group(0) @binding(4) var<storage, read>        bvh_pos:       array<vec4f>;
@group(0) @binding(5) var<storage, read>        bvh_indices:   array<vec4u>;
@group(0) @binding(6) var<storage, read>        materials:     array<vec4f>;
@group(0) @binding(7) var                       base_color_tex: texture_2d_array<f32>;
@group(0) @binding(8) var                       tex_sampler:   sampler;
@group(0) @binding(9) var<storage, read>        bvh_uvs:       array<vec4f>;

const MAX_ALPHA_SKIP = 8u; // max transparent layers to traverse

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let render_width  = pt.width;
    let render_height = pt.height;
    let total_pixels  = render_width * render_height;
    if (gid.x >= total_pixels) { return; }
    let pixel_id = gid.x;

    let shadow = shadow_buffer[pixel_id];
    if (shadow.shadow_active == 0u) { return; }

    var bvh_ray: Ray;
    bvh_ray.origin    = shadow.origin;
    bvh_ray.direction = shadow.direction;
    let safeDir = select(bvh_ray.direction, vec3f(1e-7), abs(bvh_ray.direction) < vec3f(1e-7));
    bvh_ray.invDirection = 1.0 / safeDir;
    bvh_ray.dirSign = vec3u(select(0u, 1u, safeDir.x < 0.0), select(0u, 1u, safeDir.y < 0.0), select(0u, 1u, safeDir.z < 0.0));

    var occluded = false;

    // Loop: intersect, check alpha, skip transparent hits
    for (var skip = 0u; skip < MAX_ALPHA_SKIP; skip++) {
        let result = bvhIntersectFirstHit(&bvh_nodes, &bvh_pos, &bvh_indices, bvh_ray);

        if (!result.didHit || result.dist >= (shadow.max_dist - 0.01)) {
            // No hit, or hit is beyond the light — NOT occluded
            break;
        }

        // Check alpha at this hit point
        let mat_id = result.indices.w;
        let mat = unpackPTMaterial(&materials, mat_id);

        var final_alpha = mat.alpha;
        if (mat.tex_layer >= 0) {
            // Reconstruct UV from barycentric coordinates
            let bw = result.barycoord; // (w, u, v) → matches intersectsTriangle output
            let uv0 = bvh_uvs[result.indices.x].xy;
            let uv1 = bvh_uvs[result.indices.y].xy;
            let uv2 = bvh_uvs[result.indices.z].xy;
            let hit_uv = bw.x * uv0 + bw.y * uv1 + bw.z * uv2;

            let tex_col = textureSampleLevel(base_color_tex, tex_sampler, hit_uv, mat.tex_layer, 0.0);
            final_alpha *= tex_col.w;
        }

        if (mat.alpha_mode == 0u || final_alpha >= mat.alpha_cutoff) {
            // Opaque hit — shadow is blocked
            occluded = true;
            break;
        }

        // Transparent hit — advance ray past this surface and try again
        let hit_pos = bvh_ray.origin + bvh_ray.direction * result.dist;
        bvh_ray.origin = hit_pos + bvh_ray.direction * 0.001;
    }

    if (!occluded) {
        let prev = accum_buffer[pixel_id];
        accum_buffer[pixel_id] = vec4f(prev.xyz + shadow.Li, prev.w);
    }

    shadow_buffer[pixel_id].shadow_active = 0u;
}
