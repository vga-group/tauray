#ifndef TAURAY_SHADOW_MAP_RENDERER_HH
#define TAURAY_SHADOW_MAP_RENDERER_HH
#include "context.hh"
#include "renderer.hh"
#include "atlas.hh"
#include "light.hh"
#include "camera.hh"
#include "shadow_map_stage.hh"

namespace tr
{

// This renderer is a bit odd in that it doesn't actually draw anything to the
// context; it only draws shadow maps into an internal atlas. As such, this
// renderer is not useful on its own and must be used as a part of a more
// comprehensive renderer (= raster_renderer).
class shadow_map_renderer
{
public:
    shadow_map_renderer(context& ctx, scene_stage& ss);
    ~shadow_map_renderer();

    dependencies render(dependencies deps);
    const atlas* get_shadow_map_atlas() const;

    struct shadow_map
    {
        unsigned atlas_index;
        unsigned map_index;
        uvec2 face_size;
        float min_bias;
        float max_bias;
        vec2 radius;
        std::vector<camera> faces;
        struct cascade
        {
            unsigned atlas_index;
            vec2 offset;
            float scale;
            float bias_scale;
            camera cam;
        };
        std::vector<cascade> cascades;
    };

    int get_shadow_map_index(const light* l) const;
    void update_shadow_map_params();
    const std::vector<shadow_map>& get_shadow_map_info() const;
    size_t get_total_shadow_map_count() const;
    size_t get_total_cascade_count() const;

private:
    void init_resources();
    void init_scene_resources();
    void add_stage(
        unsigned atlas_index,
        unsigned face_index,
        unsigned face_count
    );

    context* ctx;
    scene_stage* ss = nullptr;
    uint32_t scene_state_counter = 0;

    size_t total_shadow_map_count;
    size_t total_cascade_count;
    std::vector<shadow_map> shadow_maps;
    std::unordered_map<const light*, size_t /*index*/> shadow_map_indices;

    std::unique_ptr<atlas> shadow_atlas;
    std::vector<std::unique_ptr<shadow_map_stage>> smp;
};

}

#endif

