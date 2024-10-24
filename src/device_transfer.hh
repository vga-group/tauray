#ifndef TAURAY_DEVICE_TRANSFER_HH
#define TAURAY_DEVICE_TRANSFER_HH
#include "dependency.hh"
#include <memory>

namespace tr
{

// Facilitates device-to-device memory transfers in multi-GPU contexts.
class device_transfer_interface
{
public:
    virtual ~device_transfer_interface() = default;

    virtual vk::ImageUsageFlagBits required_src_img_flags() = 0;
    virtual vk::ImageUsageFlagBits required_dst_img_flags() = 0;
    virtual vk::BufferUsageFlagBits required_src_buffer_flags() = 0;
    virtual vk::BufferUsageFlagBits required_dst_buffer_flags() = 0;

    struct image_transfer
    {
        vk::Image src;
        vk::Image dst;
        size_t bytes_per_pixel;
        vk::ImageCopy info;
        vk::ImageLayout src_layout = vk::ImageLayout::eTransferSrcOptimal;
        vk::ImageLayout dst_layout = vk::ImageLayout::eGeneral;
    };

    struct buffer_transfer
    {
        vk::Buffer src;
        vk::Buffer dst;
        vk::BufferCopy info;
    };

    virtual void reserve(
        const std::vector<image_transfer>& images,
        const std::vector<buffer_transfer>& buffers
    ) = 0;

    // You can re-record commands for the transfer method whenever needed.
    virtual void build(
        const std::vector<image_transfer>& images,
        const std::vector<buffer_transfer>& buffers
    ) = 0;

    // Run can only be called after build(). 'deps' can only be dependencies
    // for the 'src' buffers and their device. Returned dependency is for the
    // 'dst' buffers and their device.
    virtual dependency run(const dependencies& deps, uint32_t frame_index) = 0;
};

enum device_transfer_strategy
{
    DTI_AUTO = 0,
    DTI_EXTERNAL_SEMAPHORE_HOST_BUFFER
    //DTI_WAIT_THREAD_HOST_BUFFER
    //DTI_CUDA_INTEROP
    //DTI_RDMA_PEER_TO_PEER
};

std::unique_ptr<device_transfer_interface> create_device_transfer_interface(
    device& from,
    device& to,
    device_transfer_strategy strat = DTI_AUTO
);

}

#endif
