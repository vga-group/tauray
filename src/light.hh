#ifndef TAURAY_LIGHT_HH
#define TAURAY_LIGHT_HH
#include "transformable.hh"
#include "animation.hh"

namespace tr
{

struct ambient_light
{
    vec3 color;
};

class light
{
public:
    light(vec3 color = vec3(1.0));

    void set_color(vec3 color);
    vec3 get_color() const;

private:
    vec3 color;
};

class directional_light: public light
{
public:
    directional_light(
        vec3 color = vec3(1.0),
        float angle = 0.0f
    );

    float get_angle() const;
    void set_angle(float angle);

private:
    float angle;
};

class point_light: public light
{
public:
    point_light(
        vec3 color = vec3(1.0),
        float radius = 0.0f,
        float cutoff_brightness = 5.0f/256.0f
    );

    void set_radius(float radius);
    float get_radius() const;

    void set_cutoff_brightness(float cutoff_brightness = 5.0f/256.0f);
    float get_cutoff_brightness() const;
    void set_cutoff_radius(float cutoff_radius);
    float get_cutoff_radius() const;

private:
    float radius;
    float cutoff_brightness;
};

class spotlight: public point_light
{
public:
    spotlight(
        vec3 color = vec3(1.0),
        float cutoff_angle = 30,
        float falloff_exponent = 1,
        float radius = 0.02f
    );

    void set_cutoff_angle(float cutoff_angle);
    float get_cutoff_angle() const;

    void set_falloff_exponent(float falloff_exponent);
    float get_falloff_exponent() const;

    // Approximates falloff exponent from the inner angle representation.
    void set_inner_angle(float inner_angle, float ratio = 1.f/255.f);

private:
    float cutoff_angle;
    float falloff_exponent;
};

struct gpu_tri_light
{
    pvec3 pos[3];
    pvec3 emission_factor;

    pvec2 uv[3];
    int emission_tex_id;

    // TODO: Put pre-calculated data here, since we want to pad to a multiple of
    // 32 anyway. There's 5 ints left.
    float power_estimate; // Negative marks double-sided triangles.
    //int padding[5];
};

}

#endif
