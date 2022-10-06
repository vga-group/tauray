#include "math.hh"
#include <algorithm>
#include <cstring>
#include <climits>
#include <limits>

namespace tr
{

void decompose_matrix(
    const glm::mat4& transform,
    glm::vec3& translation,
    glm::vec3& scaling,
    glm::quat& orientation
){
    translation = transform[3];
    scaling = glm::vec3(
        glm::length(transform[0]),
        glm::length(transform[1]),
        glm::length(transform[2])
    );
    orientation = glm::quat(glm::mat4(
        transform[0]/scaling.x,
        transform[1]/scaling.y,
        transform[2]/scaling.z,
        glm::vec4(0,0,0,1)
    ));
}

glm::vec3 get_matrix_translation(const glm::mat4& transform)
{
    return transform[3];
}

glm::vec3 get_matrix_scaling(const glm::mat4& transform)
{
    return glm::vec3(
        glm::length(transform[0]),
        glm::length(transform[1]),
        glm::length(transform[2])
    );
}

glm::quat get_matrix_orientation(const glm::mat4& transform)
{
    return glm::quat(glm::mat4(
        glm::normalize(transform[0]),
        glm::normalize(transform[1]),
        glm::normalize(transform[2]),
        glm::vec4(0,0,0,1)
    ));
}

glm::quat rotate_towards(
    glm::quat orig,
    glm::quat dest,
    float angle_limit
){
    angle_limit = glm::radians(angle_limit);

    float cos_theta = dot(orig, dest);
    if(cos_theta > 0.999999f)
    {
        return dest;
    }

    if(cos_theta < 0)
    {
        orig = orig * -1.0f;
        cos_theta *= -1.0f;
    }

    float theta = acos(cos_theta);
    if(theta < angle_limit) return dest;
    return glm::mix(orig, dest, angle_limit/theta);
}

glm::quat quat_lookat(
    glm::vec3 dir,
    glm::vec3 up,
    glm::vec3 forward
){
    dir = glm::normalize(dir);
    up = glm::normalize(up);
    forward = glm::normalize(forward);

    glm::quat towards = glm::rotation(
        forward,
        glm::vec3(0,0,-1)
    );
    return glm::quatLookAt(dir, up) * towards;
}

bool solve_quadratic(float a, float b, float c, float& x0, float& x1)
{
    float D = b * b - 4 * a * c;
    float sD = sqrt(D) * sign(a);
    float denom = -0.5f/a;
    x0 = (b + sD) * denom;
    x1 = (b - sD) * denom;
    return !std::isnan(sD);
}

void solve_cubic_roots(
    double a, double b, double c, double d,
    std::complex<double>& r1,
    std::complex<double>& r2,
    std::complex<double>& r3
){
    double d1 = 2*b*b*b - 9*a*b*c + 27*a*a*d;
    double d2 = b*b - 3*a*c;
    auto d3 = sqrt(std::complex<double>(d1*d1 - 4*d2*d2*d2));

    double k = 1/(3*a);

    auto p1 = std::pow(0.5*(d1+d3), 1/3.0f);
    auto p2 = std::pow(0.5*(d1-d3), 1/3.0f);

    std::complex<double> c1(0.5, 0.5*sqrt(3));
    std::complex<double> c2(0.5, -0.5*sqrt(3));

    r1 = k*(-b - p1 - p2).real();
    r2 = k*(-b + c1*p1 + c2*p2);
    r3 = k*(-b + c2*p1 + c1*p2);
}

double cubic_bezier(dvec2 p1, dvec2 p2, double t)
{
    // x = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    //   = (3*P1 - 3*P2 + 1)*t^3 + (-6*P1 + 3*P2)*t^2 + (3*P1)*t
    //   when P0=(0,0) and P3=(1,1)

    std::complex<double> r1;
    std::complex<double> r2;
    std::complex<double> r3;
    solve_cubic_roots(
        3.*p1.x-3.*p2.x+1., 3.*p2.x-6.*p1.x, 3.*p1.x, -t,
        r1, r2, r3
    );

    double xt = r1.real();
    double best = 0;
    if(r1.real() < 0) best = -r1.real();
    else if(r1.real() > 1) best = r1.real()-1;

    if(abs(r2.imag()) < 0.00001)
    {
        double cost = 0;
        if(r2.real() < 0) cost = -r2.real();
        else if(r2.real() > 1) cost = r2.real()-1;
        if(cost < best)
        {
            best = cost;
            xt = r2.real();
        }
    }

    if(abs(r3.imag()) < 0.00001)
    {
        double cost = 0;
        if(r3.real() < 0) cost = -r3.real();
        else if(r3.real() > 1) cost = r3.real()-1;
        if(cost < best)
        {
            best = cost;
            xt = r3.real();
        }
    }

    return (3.*p1.y-3.*p2.y+1.)*xt*xt*xt
        + (3.*p2.y-6.*p1.y)*xt*xt
        + (3.*p1.y)*xt;
}

unsigned calculate_mipmap_count(uvec2 size)
{
    return (unsigned)std::floor(std::log2(std::max(size.x, size.y)))+1u;
}

frustum operator*(const mat4& mat, const frustum& f)
{
    frustum res = f;
    mat4 m = glm::transpose(glm::affineInverse(mat));
    for(vec4& p: res.planes) p = m * p;
    return res;
}

bool obb_frustum_intersection(
    const aabb& box,
    const mat4& transform,
    const frustum& f
){
    frustum tf = f;
    mat4 m = glm::transpose(transform);
    for(vec4& p: tf.planes) p = m * p;
    return aabb_frustum_intersection(box, tf);
}

bool aabb_frustum_intersection(const aabb& box, const frustum& f)
{
    for(vec4 p: f.planes)
    {
        if(
            dot(p, vec4(box.min, 1.0f)) < 0 &&
            dot(p, vec4(box.min.x, box.min.y, box.max.z, 1.0f)) < 0 &&
            dot(p, vec4(box.min.x, box.max.y, box.min.z, 1.0f)) < 0 &&
            dot(p, vec4(box.min.x, box.max.y, box.max.z, 1.0f)) < 0 &&
            dot(p, vec4(box.max.x, box.min.y, box.min.z, 1.0f)) < 0 &&
            dot(p, vec4(box.max.x, box.min.y, box.max.z, 1.0f)) < 0 &&
            dot(p, vec4(box.max.x, box.max.y, box.min.z, 1.0f)) < 0 &&
            dot(p, vec4(box.max, 1.0f)) < 0
        ) return false;
    }

    return true;
}

unsigned ravel_tex_coord(uvec3 p, uvec3 size)
{
    return p.z * size.x * size.y + p.y * size.x + p.x;
}

ray operator*(const mat4& mat, const ray& r)
{
    ray res;
    res.o = mat * vec4(r.o, 1.0f);
    res.dir = inverseTranspose(mat) * vec4(r.dir, 0);
    return res;
}

bool flipped_winding_order(const mat3& transform)
{
    return determinant(transform) < 0;
}

template<typename T, typename U>
T bit_cast(U&& u) { T t; memcpy(&t, &u, sizeof(t)); return t; }

// https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
uint16_t float_to_half(float value)
{
    uint32_t bits = bit_cast<uint32_t>(value);
    uint32_t sign = bits & 0x80000000;
    bits ^= sign;
    bool is_nan = 0x7f800000 < bits;
    bool is_sub = bits < 0x38800000;
    float norm = bit_cast<float>(bits);
    float subn = norm;
    subn *= bit_cast<float>(0x01000000);
    subn *= bit_cast<float>(0x46000000);
    norm *= bit_cast<float>(0x07800000);
    bits = bit_cast<uint32_t>(norm);
    bits += ((bits >> 13) & 1) + 0x0fff;
    bits ^= -is_sub & (bit_cast<uint32_t>(subn) ^ bits);
    bits >>= 13;
    bits ^= -(0x7c00 < bits) & (0x7c00 ^ bits);
    bits ^= -is_nan & (0x7e00 ^ bits);
    bits |= sign >> 16;
    return bits;
}

uint32_t next_power_of_two(uint32_t n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

uint32_t align_up_to(uint32_t n, uint32_t align)
{
    if(align <= 1) return n;
    return (n+(align-1))/align*align;
}

uint32_t pcg(uint32_t seed)
{
    seed = seed * 747796405u + 2891336453u;
    seed = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
    return (seed >> 22) ^ seed;
}

float halton(int index, int base)
{
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0)
    {
        f = f / base;
        r = r + f * (index % base);
        index = (int)glm::floor(index / base);
    }

    return r;
}

std::vector<vec2> halton_2d_sequence(int sequence_length, int x_base, int y_base)
{
    std::vector<vec2> result(sequence_length);
    for(int i = 0; i < sequence_length; ++i)
    {
        result[i] = vec2(
            halton(i+1, x_base),
            halton(i+1, y_base)
        );
    }

    return result;
}

std::vector<vec2> get_camera_jitter_sequence(
    int sequence_length, uvec2 resolution
){
    std::vector<vec2> seq = halton_2d_sequence(sequence_length);
    for(vec2& v: seq)
        v = (v*2.0f-1.0f)/vec2(resolution);
    return seq;
}

}
