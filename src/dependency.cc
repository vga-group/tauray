#include "dependency.hh"
#include "context.hh"

namespace tr
{

void dependencies::add(dependency dep)
{
    auto it = std::lower_bound(ids.begin(), ids.end(), dep.id);
    size_t i = it - ids.begin();

    ids.insert(ids.begin() + i, dep.id);
    semaphores.insert(semaphores.begin() + i, dep.timeline_semaphore);
    values.insert(values.begin() + i, dep.wait_value);
    wait_stages.insert(wait_stages.begin() + i, dep.wait_stage);
}

void dependencies::concat(dependencies deps)
{
    for(size_t i = 0; i < deps.ids.size(); ++i)
        add(dependency{deps.ids[i], deps.semaphores[i], deps.values[i], deps.wait_stages[i]});
}

void dependencies::concat(dependencies deps, device_id only_id)
{
    size_t begin, end;
    get_range(only_id, begin, end);
    for(size_t i = begin; i < end; ++i)
        add(dependency{deps.ids[i], deps.semaphores[i], deps.values[i], deps.wait_stages[i]});
}

void dependencies::clear()
{
    ids.clear();
    semaphores.clear();
    values.clear();
    wait_stages.clear();
}

void dependencies::clear(device_id id)
{
    for(size_t i = 0; i < ids.size();)
    {
        if(ids[i] == id)
        {
            ids.erase(ids.begin() + i);
            semaphores.erase(semaphores.begin() + i);
            values.erase(values.begin() + i);
            wait_stages.erase(wait_stages.begin() + i);
        }
        else ++i;
    }
}

size_t dependencies::size(device_id id) const
{
    size_t begin, end;
    get_range(id, begin, end);
    return end-begin;
}

size_t dependencies::total_size() const
{
    return values.size();
}

size_t dependencies::count_unique_devices() const
{
    size_t count = 0;
    device_id prev = 0xFFFFFFFFu;
    for(device_id id: ids)
    {
        if(id != prev)
        {
            count++;
            prev = id;
        }
    }
    return count;
}

uint64_t dependencies::value(device_id id, size_t index) const
{
    size_t begin, end;
    get_range(id, begin, end);
    return *(values.begin() + begin + index);
}

void dependencies::wait(device& dev)
{
    size_t begin, end;
    get_range(dev.index, begin, end);
    (void)dev.dev.waitSemaphores({{}, uint32_t(end-begin), semaphores.data()+begin, values.data()+begin}, UINT64_MAX);
}

vk::TimelineSemaphoreSubmitInfo dependencies::get_timeline_info(device_id id) const
{
    size_t begin, end;
    get_range(id, begin, end);
    return vk::TimelineSemaphoreSubmitInfo(
        end-begin, values.data()+begin, 0, nullptr
    );
}

vk::SubmitInfo dependencies::get_submit_info(device_id id, vk::TimelineSemaphoreSubmitInfo& s) const
{
    size_t begin, end;
    get_range(id, begin, end);
    vk::SubmitInfo submit_info(
        end-begin, semaphores.data()+begin, wait_stages.data()+begin,
        0, nullptr,
        0, nullptr
    );
    submit_info.pNext = (void*)&s;
    return submit_info;
}

void dependencies::get_range(device_id id, size_t& begin, size_t& end) const
{
    bool running = false;
    begin = ids.size();
    for(size_t i = 0; i < ids.size(); ++i)
    {
        if(ids[i] == id && !running)
        {
            running = true;
            begin = i;
        }
        else if(ids[i] != id && running)
        {
            end = i;
            return;
        }
    }
    end = ids.size();
}

}
