#include "load_balancer.hh"

namespace tr
{

load_balancer::load_balancer(context& ctx, const std::vector<double>& initial_weights)
: ctx(&ctx), workloads(initial_weights)
{
    normalize_workloads();
}

void load_balancer::update(renderer& ren)
{
    double sum_speed = 0;
    for(size_t i = 0; i < workloads.size(); ++i)
    {
        double time = ctx->get_timing(i, "path tracing");
        double speed = max(workloads[i] / time, 0.0);
        sum_speed += speed;
    }

    if(sum_speed > 0 && std::isfinite(sum_speed))
    {
        for(size_t i = 0; i < workloads.size(); ++i)
        {
            double time = ctx->get_timing(i, "path tracing");
            double speed = workloads[i] / time;
            workloads[i] = mix(workloads[i], speed/sum_speed, 0.1);
        }
    }
    ren.set_device_workloads(workloads);
}

void load_balancer::normalize_workloads()
{
    workloads.resize(ctx->get_devices().size());
    double sum = 0;
    double add = 0;
    for(double w: workloads) sum += w;

    if(sum == 0)
    {
        add = 1.0f;
        sum = workloads.size();
    }

    for(double& w: workloads)
    {
        w = (max(w, 0.0)+add)/sum;
    }
}

}
