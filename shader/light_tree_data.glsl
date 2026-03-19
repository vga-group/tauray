#ifndef LIGHT_TREE_DATA_GLSL
#define LIGHT_TREE_DATA_GLSL

#include "math.glsl"

const uint NULL_LIGHT_KIND = 0;
const uint TRI_LIGHT_KIND = 1;
const uint POINT_LIGHT_KIND = 2;
const uint DIRECTIONAL_LIGHT_KIND = 3;
const uint ENVMAP_KIND = 4;

struct light_link
{
    uint kind;
    uint primitive_id;
};

// Yes, I did try compacting this. It was more complicated, less precise and
// most importantly, slower! The tree traversal is compute-bound, not
// bandwidth-bound, so keeping compute at minimum is key.
struct light_tree_node
{
    // 1D radius is precalculated from size, so it's optional.
    vec4 center_radius;
    vec4 radius_brightness;
};

uint light_tree_layer_size(uint layer, uint tree_width, uint leaf_count)
{
    uint width = ipow(tree_width, layer+1);
    return (leaf_count + width - 1) / width * tree_width;
}

#endif
