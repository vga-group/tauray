#ifndef TAURAY_MATH_TCC
#define TAURAY_MATH_TCC
#include "math.hh"

namespace tr
{

template<length_t L, typename T, glm::qualifier Q>
T vecmax(const vec<L, T, Q>& v)
{
    T m = v[0];
    for(length_t i = 1; i < L; ++i) m = max(m, v[i]);
    return m;
}

template<length_t L, typename T, glm::qualifier Q>
T vecmin(const vec<L, T, Q>& v)
{
    T m = v[0];
    for(length_t i = 1; i < L; ++i) m = min(m, v[i]);
    return m;
}

template<typename T>
T cubic_spline(T p1, T m1, T p2, T m2, float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float tmp = 2 * t3 - 3 * t2;
    return
        (tmp + 1) * p1 +
        (t3 - 2 * t2 + t) * m1 +
        (-tmp) * p2 +
        (t3 - t2) * m2;
}

}

#endif
