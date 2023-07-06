#ifndef TAURAY_DEPENDENCY_HH
#define TAURAY_DEPENDENCY_HH

#include "device.hh"

namespace tr
{

// This class is similar to events in OpenCL, but I didn't name it as 'event' so
// that it wouldn't get mixed up with Vulkan events which are a different thing.
// You can insert dependencies between rendering steps by passing these around.
struct dependency
{
    device_id id;
    vk::Semaphore timeline_semaphore;
    uint64_t wait_value;
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTopOfPipe;
};

// Bundles of dependencies are stored in this structure instead of a vector,
// because this makes it easier to pass to Vulkan mostly. You can include
// semaphores for various different devices as well, and request ones for only
// specific devices.
class dependencies
{
public:
    template<typename... Args>
    dependencies(Args... deps) { (add(deps), ...); }

    void add(dependency dep);
    void concat(dependencies deps);
    void clear();
    void clear(device_id id);
    size_t size(device_id id) const;
    uint64_t value(device_id id, size_t index) const;

    void wait(device& dev);

    vk::TimelineSemaphoreSubmitInfo get_timeline_info(device_id id) const;
    vk::SubmitInfo get_submit_info(device_id id, vk::TimelineSemaphoreSubmitInfo& s) const;

private:
    void get_range(device_id id, size_t& begin, size_t& end) const;

    std::vector<device_id> ids;
    std::vector<vk::Semaphore> semaphores;
    std::vector<uint64_t> values;
    std::vector<vk::PipelineStageFlags> wait_stages;
};

}

#endif

