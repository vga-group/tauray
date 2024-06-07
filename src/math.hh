#ifndef TAURAY_MATH_HH
#define TAURAY_MATH_HH

#define GLM_ENABLE_EXPERIMENTAL
// Makes GLM angles predictable
#define GLM_FORCE_RADIANS
#define GLM_FORCE_INTRINSICS
#define GLM_FORCE_SSE2
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtx/string_cast.hpp>
#include <complex>
#include <vector>

// GLM has a lot of necessary functionality as 'experimental', which means that
// the API gets deprecated pretty fast. This check makes sure we don't get weird
// build problems if the version is something different.
//
// If you want to bump this version, not only make sure that the program builds,
// but also check that SSE works as expected.
#if GLM_VERSION != 998
#pragma warning "This program was written to use GLM 0.9.9.8. " \
    GLM_VERSION_MESSAGE
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace tr
{

// The normal *vec3 and *mat3 are aligned to 128 bits due to enabling SSE, so
// these are the packed types if they happen to be needed somewhere.
using pvec4 = glm::vec<4, float, glm::packed_highp>;
using pvec3 = glm::vec<3, float, glm::packed_highp>;
using pvec2 = glm::vec<2, float, glm::packed_highp>;
using pivec4 = glm::vec<4, int, glm::packed_highp>;
using pivec3 = glm::vec<3, int, glm::packed_highp>;
using pivec2 = glm::vec<2, int, glm::packed_highp>;
using puvec4 = glm::vec<4, unsigned, glm::packed_highp>;
using puvec3 = glm::vec<3, unsigned, glm::packed_highp>;
using puvec2 = glm::vec<2, unsigned, glm::packed_highp>;
using pmat4 = glm::mat<4, 4, float, glm::packed_highp>;
using pmat3 = glm::mat<3, 3, float, glm::packed_highp>;
using pmat2 = glm::mat<2, 2, float, glm::packed_highp>;

using namespace glm;

template<length_t L, typename T, glm::qualifier Q>
T vecmax(const vec<L, T, Q>& v);

template<length_t L, typename T, glm::qualifier Q>
T vecmin(const vec<L, T, Q>& v);

void decompose_matrix(
    const mat4& transform,
    vec3& translation,
    vec3& scaling,
    quat& orientation
);

vec3 get_matrix_translation(const mat4& transform);
vec3 get_matrix_scaling(const mat4& transform);
quat get_matrix_orientation(const mat4& transform);

quat rotate_towards(
    quat orig,
    quat dest,
    float angle_limit
);

quat quat_lookat(
    vec3 dir,
    vec3 up,
    vec3 forward = vec3(0,0,-1)
);

bool solve_quadratic(float a, float b, float c, float& x0, float& x1);
void solve_cubic_roots(
    double a, double b, double c, double d,
    std::complex<double>& r1,
    std::complex<double>& r2,
    std::complex<double>& r3
);
double cubic_bezier(dvec2 p1, dvec2 p2, double t);
template<typename T>
T cubic_spline(T p1, T m1, T p2, T m2, float t);

unsigned calculate_mipmap_count(uvec2 size);

// axis-aligned bounding box
struct aabb
{
    vec3 min;
    vec3 max;
};

struct frustum
{
    vec4 planes[6];
};

// Assumes affine transform!
frustum operator*(const mat4& mat, const frustum& f);

bool obb_frustum_intersection(
    const aabb& box,
    const mat4& transform,
    const frustum& f
);

bool aabb_frustum_intersection(const aabb& box, const frustum& f);

unsigned ravel_tex_coord(uvec3 p, uvec3 size);

struct ray
{
    vec3 o;
    vec3 dir;
};

ray operator*(const mat4& mat, const ray& r);

// This function simply checks if the given matrix causes the winding order
// of triangles in a model to flip.
bool flipped_winding_order(const mat3& transform);

uint16_t float_to_half(float f);

uint32_t next_power_of_two(uint32_t n);

uint32_t align_up_to(uint32_t n, uint32_t align);

uint32_t pcg(uint32_t seed);
float generate_uniform_random(uint32_t& seed);

float halton(int index, int base);

std::vector<vec2> halton_2d_sequence(
    int sequence_length,
    int x_base = 2, int y_base = 3
);

std::vector<vec2> get_camera_jitter_sequence(
    int sequence_length, uvec2 resolution
);

size_t hash_combine(size_t a, size_t b);


float r1_noise(float x);
vec2 r2_noise(vec2 x);
vec3 r3_noise(vec3 x);

}

#include "math.tcc"
#endif
