#include "server_context.hh"
#include <csignal>
#include <cstdlib>
#include <cstdio>

namespace
{
bool should_exit = false;

void exit_handler(int)
{
    should_exit = true;
}

};

namespace tr
{

server_context::server_context(const options& opt)
: context(opt)
{
    init_vulkan(vkGetInstanceProcAddr);
    image_array_layers = 0;
    init_devices();
    init_resources();
    signal(SIGINT, exit_handler);
}

server_context::~server_context()
{
    deinit_resources();
    deinit_devices();
    deinit_vulkan();
}

bool server_context::init_frame()
{
    return should_exit;
}

uint32_t server_context::prepare_next_image(uint32_t)
{
    return 0;
}

void server_context::finish_image(
    uint32_t, uint32_t, bool
){
}

bool server_context::queue_can_present(
    const vk::PhysicalDevice&, uint32_t,
    const vk::QueueFamilyProperties&
){
    return false;
}

}
