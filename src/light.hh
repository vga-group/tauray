#ifndef TAURAY_LIGHT_HH
#define TAURAY_LIGHT_HH
#include "transformable.hh"
#include "animation.hh"

namespace tr
{
class camera_node;

class light: public animated_node
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
        vec3 direction = vec3(0,-1,0),
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

}

#endif
