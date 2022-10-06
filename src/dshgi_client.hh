#ifndef TAURAY_DSHGI_CLIENT_HH
#define TAURAY_DSHGI_CLIENT_HH
#include "context.hh"
#include "stage.hh"
#include "texture.hh"
#include "scene_update_stage.hh"
#include "compute_pipeline.hh"
#include "sh_grid.hh"
#include <condition_variable>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace tr
{

class scene;
class dshgi_client_stage;
class dshgi_client
{
friend class dshgi_client_stage;
public:
    struct options
    {
        std::string server_address;
    };

    dshgi_client(context& ctx, const options& opt);
    dshgi_client(const dshgi_client& other) = delete;
    dshgi_client(dshgi_client&& other) = delete;
    ~dshgi_client();

    void set_scene(scene* s);
    // If this returns true, you will need to rebuild the scene buffers.
    bool refresh();
    dependencies render(dependencies deps);

private:
    static void receiver_worker(dshgi_client* s);

    context* ctx;
    options opt;
    scene* cur_scene;

    std::mutex remote_grids_mutex;
    struct sh_grid_data
    {
        bool topo_changed = true;
        bool data_updated = true;
        sh_grid grid;
        std::vector<uint8_t> data;
    };
    std::vector<sh_grid_data> remote_grids;

    // Updated only when refresh() is called, so it's safe from the main thread.
    std::vector<sh_grid_data> local_grids;
    // The latest states of the grid data should be uploaded here.
    std::unordered_map<sh_grid*, texture> sh_grid_upload_textures;
    std::unordered_map<sh_grid*, texture> sh_grid_tmp_textures;
    std::unordered_map<sh_grid*, texture> sh_grid_blended_textures;

    time_ticks remote_timestamp;
    bool new_remote_timestamp;

    bool exit_receiver;
    std::thread receiver_thread;
    std::unique_ptr<dshgi_client_stage> sh_refresher;
};

}

#endif

