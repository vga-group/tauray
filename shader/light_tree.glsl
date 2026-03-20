#ifndef LIGHT_TREE_GLSL
#define LIGHT_TREE_GLSL

#include "light_tree_data.glsl"

#ifndef LIGHT_TREE_SET
#define LIGHT_TREE_SET 3
#endif

#ifndef printf
#define printf 
#endif

layout(binding = 0, set = LIGHT_TREE_SET) readonly buffer light_tree_links_buffer
{
    light_link array[];
} light_tree_links;

layout(binding = 1, set = LIGHT_TREE_SET) readonly buffer light_tree_buffer
{
    light_tree_node array[];
} light_tree;

layout(binding = 2, set = LIGHT_TREE_SET) readonly buffer light_tree_trail_buffer
{
    uint array[];
} light_tree_trail;

layout(binding = 3, set = LIGHT_TREE_SET) readonly buffer light_tree_param_buffer
{
    float tree_weight;
    float envmap_weight;
    float directional_light_weight;
    uint leaf_count;
    uint tree_start;
    uint tree_width;
    uint tree_height;
    uint branch_bits;
    uint branch_mask;
    uint layer_offsets[32];
} light_tree_params;

float light_tree_weight(
    vec3 pos,
    vec3 normal,
    float transmission,
    light_tree_node node
){
    vec3 radius = node.radius_brightness.xyz;
    float brightness = node.radius_brightness.w;
    vec3 center = node.center_radius.xyz;
    float radius1 = node.center_radius.w;

    vec3 delta = center - pos;
    float delta2 = dot(delta, delta);
    float dn = dot(normal, delta);
    if(dn < 0 && transmission > 0) dn = -dn;
    float a = clamp((dn + radius1)*inversesqrt((dn * 2.0f + radius1) * radius1 + delta2), 0.0f, 1.0f);
    if(transmission >= 0) brightness *= a;

    vec3 min_delta_dist = max(abs(delta) - radius, vec3(0.0f));
    float min_dist2 = dot(min_delta_dist, min_delta_dist);
    return brightness / max(min_dist2, radius1);
}

vec3 light_type_weights()
{
    return vec3(
        light_tree_params.tree_weight,
        light_tree_params.directional_light_weight,
        light_tree_params.envmap_weight
    );
}

// Only does triangle and point lights!
void light_tree_sample_light(
    inout uint seed,
    vec3 pos,
    vec3 normal,
    float transmission,
    out light_link ll,
    out float pmf
){
    uint layer_start = 0;
    uint entry_index = 0;
    pmf = 1.0f;

    for(int layer = 0; layer < light_tree_params.tree_height; ++layer)
    {
        entry_index *= LIGHT_TREE_WIDTH;
        float sum_weight = 0;
        float selected_weight = 0;
        uint selected_branch = 0;

        for(uint branch = 0; branch < LIGHT_TREE_WIDTH; ++branch)
        {
            light_tree_node node = light_tree.array[light_tree_params.tree_start+layer_start+entry_index+branch];

            float weight = light_tree_weight(pos, normal, transmission, node);
            sum_weight += weight;

            float sw = sum_weight == 0 ? branch+1 : sum_weight;
            float w = sum_weight == 0 ? 1.0f : weight;

            float u = generate_uniform_random_lq(seed);
            if(u * sw <= w && node.radius_brightness.w > 0)
            {
                selected_branch = branch;
                selected_weight = weight;
            }
        }

        pmf *= sum_weight == 0 ? 1.0f / LIGHT_TREE_WIDTH : selected_weight / sum_weight;

        entry_index = entry_index + selected_branch;
        layer_start += light_tree_params.layer_offsets[layer];
    }

    ll = light_tree_links.array[entry_index];
}

// Only does triangle and point lights!
float light_tree_light_pmf(vec3 pos, vec3 normal, float transmission, light_link ll)
{
    uint light_index = 0;
    if(ll.kind == POINT_LIGHT_KIND)
    {
        light_index = ll.primitive_id;
    }
    else
    { // Tri light
        light_index = scene_metadata.point_light_count + ll.primitive_id;
    }
    uint trail = light_tree_trail.array[light_index];

    uint layer_start = 0;
    uint entry_index = 0;
    float pmf = 1.0f;

    for(int layer = 0; layer < light_tree_params.tree_height; ++layer)
    {
        entry_index *= LIGHT_TREE_WIDTH;
        float sum_weight = 0;
        float selected_weight = 0;
        uint selected_branch = trail & light_tree_params.branch_mask;
        trail >>= light_tree_params.branch_bits;

        for(uint branch = 0; branch < LIGHT_TREE_WIDTH; ++branch)
        {
            light_tree_node node = light_tree.array[light_tree_params.tree_start+layer_start+entry_index+branch];

            float weight = light_tree_weight(pos, normal, transmission, node);

            sum_weight += weight;
            if(branch == selected_branch)
                selected_weight = weight;
        }

        pmf *= sum_weight == 0 ? 1.0f / LIGHT_TREE_WIDTH : selected_weight / sum_weight;

        entry_index = entry_index + selected_branch;
        layer_start += light_tree_params.layer_offsets[layer];
    }
    return pmf;
}

struct light_sample
{
    bool infinitesimal;
    vec3 color;
    vec3 dir;
    float dist;
    float pdf;

#ifdef LIGHT_SAMPLE_HIT_INFO
    uint instance_id;
    uint primitive_id;
    vec2 hit_info;
    vec3 normal;
#endif
};

vec3 sample_triangle_light(
    tri_light tl,
    vec2 u,
    vec3 pos,
    out vec3 dir,
    out float dist,
    out vec3 color,
    out float pdf
){
    vec3 A = tl.pos[0]-pos;
    vec3 B = tl.pos[1]-pos;
    vec3 C = tl.pos[2]-pos;
    dir = sample_triangle_light(u, A, B, C, pdf);
    dist = ray_plane_intersection_dist(dir, A, B, C);

    vec3 bary = get_barycentric_coords(dir * dist, A, B, C);
    color = r9g9b9e5_to_rgb(tl.emission_factor);

    if(tl.emission_tex_id >= 0)
    {
        vec2 uv =
            bary.x * unpackHalf2x16(tl.uv[0]) +
            bary.y * unpackHalf2x16(tl.uv[1]) +
            bary.z * unpackHalf2x16(tl.uv[2]);
        color *= textureLod(textures[nonuniformEXT(tl.emission_tex_id)], uv, 0.0f).rgb;
    }
    return bary;
}

// Warning: does NOT update the seed! You need to do that yourself.
light_sample sample_light(
    uvec4 rand32,
    vec3 pos,
    vec3 normal,
    float transmission,
    float min_dist,
    float max_dist
){
    vec3 prob = light_type_weights();
    light_sample ls;
    ls.color = vec3(0);

    vec4 u = ldexp(vec4(rand32), ivec4(-32));

#ifdef LIGHT_SAMPLE_HIT_INFO
    ls.instance_id = NULL_INSTANCE_ID;
    ls.normal = vec3(0);
#endif

    ls.dist = max_dist;

    float local_pdf = 0.0f;
    if((u.x -= prob.x) < 0)
    { // Point or triangle light
        light_link link;
        light_tree_sample_light(rand32.y, pos, normal, transmission, link, ls.pdf);

        ls.pdf *= prob.x;
#ifdef LIGHT_SAMPLE_HIT_INFO
        ls.primitive_id = link.primitive_id;
#endif

        if(link.kind == POINT_LIGHT_KIND)
        { // Point light
            point_light pl = point_lights.lights[link.primitive_id];
            sample_point_light(pl, u.zw, pos, ls.dir, ls.dist, ls.color, local_pdf);

#ifdef LIGHT_SAMPLE_HIT_INFO
            ls.instance_id = POINT_LIGHT_INSTANCE_ID;
            vec3 p = pos + ls.dir * ls.dist;
            ls.normal = normalize(p - vec3(pl.pos_x, pl.pos_y, pl.pos_z));
            ls.hit_info = octahedral_pack(ls.normal) * 0.5f + 0.5f;
            if(local_pdf <= 0.0f) ls.normal = vec3(0);
#else
            // If there's no hit info, the caller cannot apply inverse-square
            // law, so it must be done here.
            //if(local_pdf < 0.0f) ls.color /= ls.dist * ls.dist;
#endif
        }
        else
        { // Tri light
            tri_light tl = tri_lights.lights[link.primitive_id];
            vec2 hit_info = sample_triangle_light(tl, u.zw, pos, ls.dir, ls.dist, ls.color, local_pdf).yz;
            ls.dist -= min_dist;

#ifdef LIGHT_SAMPLE_HIT_INFO
            ls.instance_id = tl.instance_id;
            ls.primitive_id = tl.primitive_id;
            ls.normal = normalize(cross(
                tl.pos[0] - tl.pos[1],
                tl.pos[0] - tl.pos[2]
            ));
#endif

            // TODO: Check this condition if ReSTIR is doing NaNs with triangle
            // lights! May still need the flat normal check or something else
            // that is equivalent! ls.normal?
            if(
                isinf(local_pdf) || local_pdf <= 0 || any(isnan(ls.dir)) ||
                ls.dist < min_dist // || abs(dot(ls.dir, d.flat_normal)) < 1e-4f
            ){
#ifdef LIGHT_SAMPLE_HIT_INFO
                ls.instance_id = NULL_INSTANCE_ID;
#else
                ls.color = vec3(0);
                ls.dist = 0;
                local_pdf = 1.0f;
                ls.dir = normal;
#endif
            }
        }
    }
    else if((u.x -= prob.y) < 0)
    { // Directional light
        int selected_index = clamp(
            int(u.y * scene_metadata.directional_light_count),
            0, int(scene_metadata.directional_light_count)-1
        );
        directional_light dl = directional_lights.lights[selected_index];

        sample_directional_light(dl, u.zw, ls.dir, ls.color, local_pdf);
#ifdef LIGHT_SAMPLE_HIT_INFO
        ls.instance_id = DIRECTIONAL_LIGHT_INSTANCE_ID;
        ls.primitive_id = floatBitsToUint(local_pdf * prob.y);
        ls.hit_info = octahedral_encode(ls.dir) * 0.5f + 0.5f;
#endif

        ls.pdf = prob.y / scene_metadata.directional_light_count;
    }
    else if((u.x -= prob.z) < 0)
    { // Envmap
        rand32 += uvec4(12); // Make 'values' not correlate with future RNG samples
        pcg4d(rand32);
        ls.color = sample_environment_map(rand32.xyz, ls.dir, ls.dist, local_pdf);

#ifdef LIGHT_SAMPLE_HIT_INFO
        ls.instance_id = ENVMAP_INSTANCE_ID;
        ls.primitive_id = floatBitsToUint(local_pdf * envmap_prob);
        ls.hit_info = octahedral_pack(ls.dir) * 0.5f + 0.5f;
#endif

        ls.pdf = prob.z;
    }

    ls.infinitesimal = local_pdf <= 0;
#ifdef LIGHT_SAMPLE_HIT_INFO
    if(!ls.infinitesimal) ls.pdf *= local_pdf;
#else
    ls.pdf *= local_pdf;
#endif

    return ls;
}

float calculate_light_pdf(
    uint instance_id,
    uint primitive_id,
    float local_pdf,
    float envmap_pdf,
    vec3 pos,
    vec3 normal,
    float transmission
){
    vec3 prob = light_type_weights();

    if(
        instance_id == DIRECTIONAL_LIGHT_INSTANCE_ID ||
        instance_id == ENVMAP_INSTANCE_ID ||
        instance_id == MISS_INSTANCE_ID
    ){
        return local_pdf * prob.y / max(scene_metadata.directional_light_count, 1u) + envmap_pdf * prob.z;
    }
    else if(instance_id == NULL_INSTANCE_ID || local_pdf == 0)
        return 0;
    else
    {
        light_link link;
        if (instance_id == POINT_LIGHT_INSTANCE_ID)
        {
            link.kind = POINT_LIGHT_KIND;
            link.primitive_id = primitive_id;
        }
        else
        {
            link.kind = TRI_LIGHT_KIND;
            link.primitive_id = instances.o[instance_id].light_base_id + primitive_id;
        }
        return local_pdf * prob.x * light_tree_light_pmf(pos, normal, transmission, link);
    }
}

#endif
