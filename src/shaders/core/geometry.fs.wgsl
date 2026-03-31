

struct FragmentInput
{
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent_world: vec4f
}

struct GBufferOutput {
    @location(0) albedo : vec4f,
    @location(1) normal : vec4f,
    @location(2) position : vec4f,
    @location(3) specular_material : vec4f,
}

@fragment
fn main(in: FragmentInput) ->  GBufferOutput
{
    let surf = evaluateMaterial(in.uv, in.nor, in.tangent_world);
    if (surf.alpha < 0.5f) {
        discard;
    }

    let ao = 1.0;

    var output: GBufferOutput;
    output.albedo = vec4f(surf.albedo, surf.alpha);
    output.normal = vec4f(surf.N, 1.0); 
    output.position = vec4f(in.pos, 1.0);

    // Store PBR params: R=roughness, G=metallic, B=ao, A=shadingModelId
    output.specular_material = vec4f(surf.roughness, surf.metallic, ao, surf.shadingModelId);
    
    return output;
}
