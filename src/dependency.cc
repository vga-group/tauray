#include "dependency.hh"
#include "context.hh"

namespace tr
{

void dependencies::add(dependency dep)
{
    semaphores.push_back(dep.timeline_semaphore);
    values.push_back(dep.wait_value);
    wait_stages.push_back(dep.wait_stage);
}

void dependencies::concat(dependencies deps)
{
    semaphores.insert(semaphores.end(), deps.semaphores.begin(), deps.semaphores.end());
    values.insert(values.end(), deps.values.begin(), deps.values.end());
    wait_stages.insert(wait_stages.end(), deps.wait_stages.begin(), deps.wait_stages.end());
}

void dependencies::pop()
{
    semaphores.pop_back();
    values.pop_back();
    wait_stages.pop_back();
}

void dependencies::clear()
{
    semaphores.clear();
    values.clear();
    wait_stages.clear();
}

size_t dependencies::size() const
{
    return values.size();
}

void dependencies::wait(device& dev)
{
    (void)dev.dev.waitSemaphores({{}, semaphores, values}, UINT64_MAX);
}

uint64_t dependencies::value(size_t index) const
{
    return values[index];
}

vk::TimelineSemaphoreSubmitInfo dependencies::get_timeline_info() const
{
    return vk::TimelineSemaphoreSubmitInfo(
        semaphores.size(), values.data(), 0, nullptr
    );
}

vk::SubmitInfo dependencies::get_submit_info(vk::TimelineSemaphoreSubmitInfo& s) const
{
    vk::SubmitInfo submit_info(
        semaphores.size(), semaphores.data(), wait_stages.data(),
        0, nullptr,
        0, nullptr
    );
    submit_info.pNext = (void*)&s;
    return submit_info;
}

}
