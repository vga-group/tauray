#ifndef RT_COMMON_GLSL
#define RT_COMMON_GLSL
#include "math.glsl"
#include "rt.glsl"
#include "random_sampler.glsl"

struct hit_info
{
    // Negative if the ray escaped the scene or otherwise died. Otherwise, it's
    // the index of the mesh instance, and primitive_id and barycentrics are
    // valid as well.
    int instance_id;
    // If instance_id is non-negative, this is the triangle index. Otherwise,
    // if primitive_id is non-negative, it is the light index. If it is negative,
    // that means that the ray escaped the scene and hit the environment map
    // instead.
    int primitive_id;

    // Barycentric coordinates to the triangle that was hit.
    vec2 barycentrics;
};

float get_mesh_hit_coverage(uint instance_id, uint primitive_index, vec2 barycentrics)
{
    vec2 uv;
    get_interpolated_vertex_light(vec3(0), barycentrics, int(instance_id), int(primitive_index), uv);
    material mat = instances.o[instance_id].mat;
    float alpha = mat.albedo_factor.a;
    if(mat.albedo_tex_id >= 0)
        alpha *= texture(textures[nonuniformEXT(mat.albedo_tex_id)], uv).a;
    return alpha;
}

// Adapted from
// https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
float sphere_intersection(
    vec3 sphere_pos, float sphere_radius, vec3 ray_origin, vec3 ray_direction
){
    vec3 oc = ray_origin - sphere_pos;
    float a = dot(ray_direction, ray_direction);
    float b = 2.0f * dot(oc, ray_direction);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - 4.0f * a * c;
    if(discriminant < 0) return -1.0f;
    else return (-b - sqrt(discriminant)) / (2.0f * a);
}

float get_point_light_hit_t(point_light pl, vec3 origin, vec3 dir)
{
    return sphere_intersection(pl.pos, pl.radius, origin, dir);
}

#ifdef USE_RAY_QUERIES

float trace_ray_query_visibility(rayQueryEXT rq)
{
    const vec3 origin = rayQueryGetWorldRayOriginEXT(rq);
    const vec3 dir = rayQueryGetWorldRayDirectionEXT(rq);

    float visibility = 1.0f;
    while(rayQueryProceedEXT(rq))
    {
        uint type = rayQueryGetIntersectionTypeEXT(rq, false);
        if(type == gl_RayQueryCandidateIntersectionTriangleEXT)
        {
            visibility *= 1.0f - get_mesh_hit_coverage(
                rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) +
                rayQueryGetIntersectionGeometryIndexEXT(rq, false),
                rayQueryGetIntersectionPrimitiveIndexEXT(rq, false),
                rayQueryGetIntersectionBarycentricsEXT(rq, false)
            );
            if(visibility == 0)
                rayQueryConfirmIntersectionEXT(rq);
        }
        // Point lights are basically always excluded from visibility rays for
        // practical reasons.
        /*
        else if(type == gl_RayQueryCandidateIntersectionAABBEXT)
        { // Point lights
            point_light pl = point_lights.lights[rayQueryGetIntersectionPrimitiveIndexEXT(rq, false)];
            float hit = get_point_light_hit_t(pl, origin, dir);
            if(hit > 0) rayQueryGenerateIntersectionEXT(rq, hit);
        }
        */
    }
    uint type = rayQueryGetIntersectionTypeEXT(rq, true);
    if(type == gl_RayQueryCommittedIntersectionNoneEXT)
        return visibility;
    return 0;
}

hit_info trace_ray_query(rayQueryEXT rq, inout uint seed)
{
    const vec3 origin = rayQueryGetWorldRayOriginEXT(rq);
    const vec3 dir = rayQueryGetWorldRayDirectionEXT(rq);

    hit_info hi;
    while(rayQueryProceedEXT(rq))
    {
        uint type = rayQueryGetIntersectionTypeEXT(rq, false);
        if(type == gl_RayQueryCandidateIntersectionTriangleEXT)
        {
            float alpha_cutoff = generate_single_uniform_random(seed);
            float alpha = get_mesh_hit_coverage(
                rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) +
                rayQueryGetIntersectionGeometryIndexEXT(rq, false),
                rayQueryGetIntersectionPrimitiveIndexEXT(rq, false),
                rayQueryGetIntersectionBarycentricsEXT(rq, false)
            );
            if(alpha > alpha_cutoff)
                rayQueryConfirmIntersectionEXT(rq);
        }
        else if(type == gl_RayQueryCandidateIntersectionAABBEXT)
        { // Point lights
            point_light pl = point_lights.lights[rayQueryGetIntersectionPrimitiveIndexEXT(rq, false)];
            float hit = get_point_light_hit_t(pl, origin, dir);
            if(hit > 0) rayQueryGenerateIntersectionEXT(rq, hit);
        }
    }

    uint type = rayQueryGetIntersectionTypeEXT(rq, true);
    if(type == gl_RayQueryCommittedIntersectionTriangleEXT)
        return hit_info(
            rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true) + rayQueryGetIntersectionGeometryIndexEXT(rq, true),
            rayQueryGetIntersectionPrimitiveIndexEXT(rq, true),
            rayQueryGetIntersectionBarycentricsEXT(rq, true)
        );
    else if(type == gl_RayQueryCommittedIntersectionGeneratedEXT)
        return hit_info(-1, rayQueryGetIntersectionPrimitiveIndexEXT(rq, true), vec2(rayQueryGetIntersectionTEXT(rq, true)));
    else
        return hit_info(-1, -1, vec2(0));
}

#ifdef TEMPORAL_TABLE_SET

// Assumes IDs are outdated.
float trace_ray_query_visibility_prev(rayQueryEXT rq)
{
    const vec3 origin = rayQueryGetWorldRayOriginEXT(rq);
    const vec3 dir = rayQueryGetWorldRayDirectionEXT(rq);

    float visibility = 1.0f;
    while(rayQueryProceedEXT(rq))
    {
        uint type = rayQueryGetIntersectionTypeEXT(rq, false);
        if(type == gl_RayQueryCandidateIntersectionTriangleEXT)
        {
            int instance_id =
                rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) +
                rayQueryGetIntersectionGeometryIndexEXT(rq, false);
            uint new_id = instance_forward_map.array[instance_id];
            if(new_id != 0xFFFFFFFFu)
            {
                visibility *= 1.0f - get_mesh_hit_coverage(
                    new_id,
                    rayQueryGetIntersectionPrimitiveIndexEXT(rq, false),
                    rayQueryGetIntersectionBarycentricsEXT(rq, false)
                );
                if(visibility == 0)
                    rayQueryConfirmIntersectionEXT(rq);
            }
        }
        // Point lights are basically always excluded from visibility rays for
        // practical reasons.
        /*
        else if(type == gl_RayQueryCandidateIntersectionAABBEXT)
        { // Point lights
            point_light pl = prev_point_lights.lights[rayQueryGetIntersectionPrimitiveIndexEXT(rq, false)];
            float hit = get_point_light_hit_t(pl, origin, dir);
            if(hit > 0) rayQueryGenerateIntersectionEXT(rq, hit);
        }
        */
    }
    uint type = rayQueryGetIntersectionTypeEXT(rq, true);
    if(type == gl_RayQueryCommittedIntersectionNoneEXT)
        return visibility;
    return 0;
}

// Assumes IDs are outdated.
hit_info trace_ray_query_prev(rayQueryEXT rq, inout uint seed)
{
    const vec3 origin = rayQueryGetWorldRayOriginEXT(rq);
    const vec3 dir = rayQueryGetWorldRayDirectionEXT(rq);
    hit_info hi;
    while(rayQueryProceedEXT(rq))
    {
        uint type = rayQueryGetIntersectionTypeEXT(rq, false);
        if(type == gl_RayQueryCandidateIntersectionTriangleEXT)
        {
            int instance_id =
                rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) +
                rayQueryGetIntersectionGeometryIndexEXT(rq, false);
            uint new_id = instance_forward_map.array[instance_id];
            if(new_id != 0xFFFFFFFFu)
            {
                float alpha_cutoff = generate_single_uniform_random(seed);
                float alpha = get_mesh_hit_coverage(
                    new_id,
                    rayQueryGetIntersectionPrimitiveIndexEXT(rq, false),
                    rayQueryGetIntersectionBarycentricsEXT(rq, false)
                );
                if(alpha > alpha_cutoff)
                    rayQueryConfirmIntersectionEXT(rq);
            }
        }
        else if(type == gl_RayQueryCandidateIntersectionAABBEXT)
        { // Point lights
            point_light pl = prev_point_lights.lights[rayQueryGetIntersectionPrimitiveIndexEXT(rq, false)];
            float hit = get_point_light_hit_t(pl, origin, dir);
            if(hit > 0) rayQueryGenerateIntersectionEXT(rq, hit);
        }
    }

    uint type = rayQueryGetIntersectionTypeEXT(rq, true);
    if(type == gl_RayQueryCommittedIntersectionTriangleEXT)
        return hit_info(
            rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true) + rayQueryGetIntersectionGeometryIndexEXT(rq, true),
            rayQueryGetIntersectionPrimitiveIndexEXT(rq, true),
            rayQueryGetIntersectionBarycentricsEXT(rq, true)
        );
    else if(type == gl_RayQueryCommittedIntersectionGeneratedEXT)
        return hit_info(-1, rayQueryGetIntersectionPrimitiveIndexEXT(rq, true), vec2(rayQueryGetIntersectionTEXT(rq, true)));
    else
        return hit_info(-1, -1, vec2(0));
}
#endif

#endif
#endif
