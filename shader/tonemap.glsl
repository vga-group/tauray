layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 2, rgba32f) uniform writeonly image2DArray out_color;

#ifdef MSAA_SAMPLES
layout(binding = 1, rgba32f) uniform readonly image2DMSArray in_color;
vec4 read_tonemap(ivec3 p)
{
    vec4 sum_col = vec4(0);
    for(int i = 0; i < MSAA_SAMPLES; ++i)
    {
        vec4 src_col = imageLoad(in_color, p, i);
#ifndef POST_RESOLVE
        src_col = tonemap(src_col);
#endif
        sum_col += src_col;
    }
    sum_col *= (1.0f/float(MSAA_SAMPLES));
#ifdef POST_RESOLVE
    return tonemap(sum_col);
#else
    return sum_col;
#endif
}
#else
layout(binding = 1, rgba32f) uniform readonly image2DArray in_color;
vec4 read_tonemap(ivec3 p) { return tonemap(imageLoad(in_color, p)); }
#endif

layout(binding = 3, scalar) buffer output_reorder_buffer
{
    int indices[];
} output_reorder;

void main()
{
    ivec3 p = ivec3(gl_GlobalInvocationID.xyz);
    p.z += info.base_layer;
    if(all(lessThan(p.xy, info.size)))
    {
        vec4 col = read_tonemap(p);

        // pow() breaks with negative values whether gamma is 1 or not.
        if(info.gamma != 1.0f)
            col.rgb = pow(col.rgb, vec3(1.0f/info.gamma));

        if(info.alpha_grid_background != 0)
        {
            ivec2 grid = (p.xy/info.alpha_grid_background)&1;
            vec3 alpha_color = (grid.x^grid.y) == 0 ? vec3(0.4) : vec3(0.6);
            col.rgb = mix(alpha_color, col.rgb, col.a);
        }
        imageStore(out_color, ivec3(p.xy, output_reorder.indices[p.z]), col);
    }
}
