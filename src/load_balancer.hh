#ifndef TAURAY_LOAD_BALANCER_HH
#define TAURAY_LOAD_BALANCER_HH
#include "context.hh"
#include "renderer.hh"

namespace tr
{

class load_balancer
{
public:
    load_balancer(context& ctx, const std::vector<double>& initial_weights = {});

    void update(renderer& ren);

private:
    void normalize_workloads();

    context* ctx;
    std::vector<double> workloads;
};

}

#endif
