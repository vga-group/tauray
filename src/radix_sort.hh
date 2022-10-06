#ifndef TAURAY_RADIX_SORT_HH
#define TAURAY_RADIX_SORT_HH
#include "vkm.hh"
#include "compute_pipeline.hh"

namespace tr
{

struct device_data;

class radix_sort
{
public:
    radix_sort(device_data& ctx);
    radix_sort(radix_sort&& other) = delete;
    radix_sort(const radix_sort& other) = delete;
    ~radix_sort();

    vkm<vk::Buffer> create_keyval_buffer(size_t max_items);

    void sort(
        vk::CommandBuffer cb,
        vk::Buffer input_items,
        vk::Buffer output_items,
        vk::Buffer item_keyvals, // Must be allocated via create_keyval_buffer
        size_t item_size,
        size_t item_count,
        size_t key_bits = 32
    );

private:
    device_data* dev;
    void* rs_instance;
    compute_pipeline reorder;
};

}

#endif

