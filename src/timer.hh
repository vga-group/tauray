#ifndef TAURAY_TIMER_HH
#define TAURAY_TIMER_HH
#include "context.hh"

namespace tr
{

// You have to hold the timer instance as long as the command buffers using it
// exist. Otherwise, timing info will be broken! Typically, you would have this
// as a class member.
class timer
{
public:
    timer();
    timer(device_data& dev, const std::string& name);
    timer(timer&& other);
    ~timer();

    timer& operator=(timer&& other);

    void begin(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        vk::PipelineStageFlagBits stage = vk::PipelineStageFlagBits::eTopOfPipe
    );
    void end(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        vk::PipelineStageFlagBits stage = vk::PipelineStageFlagBits::eBottomOfPipe
    );

private:
    device_data* dev;
    int timer_id;
};

}

#endif
