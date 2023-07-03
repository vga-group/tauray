#include "sh_grid.hh"

namespace tr
{

sh_grid::sh_grid(
    uvec3 resolution,
    int order,
    transformable_node* parent
):  transformable_node(parent), radius(0.0f), order(order),
    resolution(resolution)
{
}

texture sh_grid::create_target_texture(
    device_mask dev,
    int samples_per_probe
){
    int samples_per_invocation = 1;
    get_target_sampling_info(
        dev, samples_per_probe, samples_per_invocation
    );
    samples_per_probe /= samples_per_invocation;
    return texture(
        dev,
        uvec3(
            resolution.x,
            resolution.y * get_coef_count(),
            resolution.z * samples_per_probe
        ),
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage,
        vk::ImageLayout::eGeneral
    );
}

void sh_grid::get_target_sampling_info(
    device_mask dev,
    int& samples_per_probe,
    int& samples_per_invocation
){
    uint32_t z = resolution.z * samples_per_probe;
    uint32_t max_dim = UINT32_MAX;
    for(device& d: dev)
        max_dim = min(max_dim, d.props.limits.maxImageDimension3D);
    samples_per_invocation = (z + max_dim - 1) / max_dim;
    samples_per_probe = (samples_per_probe / samples_per_invocation)
        * samples_per_invocation;
}

texture sh_grid::create_texture(device_mask dev)
{
    return texture(
        dev,
        uvec3(
            resolution.x,
            resolution.y * get_coef_count(),
            resolution.z
        ),
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|
        vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eGeneral
    );
}

size_t sh_grid::get_required_bytes() const
{
    return resolution.x * resolution.y * get_coef_count() * resolution.z *
        4 * sizeof(uint16_t);
}

void sh_grid::set_resolution(uvec3 res)
{
    this->resolution = res;
}

uvec3 sh_grid::get_resolution() const
{
    return resolution;
}

void sh_grid::set_radius(float radius)
{
    this->radius = radius;
}

float sh_grid::get_radius() const
{
    return radius;
}

void sh_grid::set_order(int order)
{
    this->order = order;
}

int sh_grid::get_order() const
{
    return order;
}

int sh_grid::get_coef_count() const
{
    return get_coef_count(order);
}

int sh_grid::get_coef_count(int order)
{
    int coef = 0;
    for(int i = 0; i <= order; ++i)
        coef += i*2+1;
    return coef;
}

float sh_grid::point_distance(vec3 p) const
{
    vec3 local_p = transpose(get_global_inverse_transpose_transform()) * vec4(p, 1);

    if(all(lessThanEqual(abs(local_p), vec3(1.0f))))
        return 0.0f;

    if(all(lessThanEqual(abs(local_p), vec3(1.0f+radius))))
    {
        // Within guard distance; calculate.
        // https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
        // Using the cube volume as a distance field.
        vec3 q = abs(local_p) - 1.0f;
        return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
    }

    return -1.0f;
}

float sh_grid::calc_density() const
{
    return (resolution.x * resolution.y * resolution.z) / calc_volume();
}

float sh_grid::calc_volume() const
{
    vec3 size = get_global_scaling() * 2.0f;
    return size.x * size.y * size.z;
}

}
