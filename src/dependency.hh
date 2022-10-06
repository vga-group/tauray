#ifndef TAURAY_DEPENDENCY_HH
#define TAURAY_DEPENDENCY_HH

#include "vkm.hh"

namespace tr
{

// This class is similar to events in OpenCL, but I didn't name it as 'event' so
// that it wouldn't get mixed up with Vulkan events which are a different thing.
// You can insert dependencies between rendering steps by passing these around.
struct dependency
{
    vk::Semaphore timeline_semaphore;
    uint64_t wait_value;
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTopOfPipe;
};

struct device_data;

// Bundles of dependencies are stored in this structure instead of a vector,
// because this makes it easier to pass to Vulkan mostly.
class dependencies
{
public:
    template<typename... Args>
    dependencies(Args... deps)
    : semaphores({deps.timeline_semaphore...}),
      values({deps.wait_value...}),
      wait_stages({deps.wait_stage...})
    {}

    void add(dependency dep);
    void concat(dependencies deps);
    void pop();
    void clear();
    size_t size() const;

    void wait(device_data& dev);
    uint64_t value(size_t index) const;

    vk::TimelineSemaphoreSubmitInfo get_timeline_info() const;
    vk::SubmitInfo get_submit_info(vk::TimelineSemaphoreSubmitInfo& s) const;

private:
    std::vector<vk::Semaphore> semaphores;
    std::vector<uint64_t> values;
    std::vector<vk::PipelineStageFlags> wait_stages;
};

}

#endif

