#ifndef SPHERICAL_HARMONICS_GLSL
#define SPHERICAL_HARMONICS_GLSL
#extension GL_EXT_control_flow_attributes : enable
#include "math.glsl"

// Using SH formulation from
// https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf

#ifndef SH_ORDER
#define SH_ORDER 2
#define SH_COEF_COUNT 9
#endif

#if SH_ORDER > 4 || SH_ORDER < 0
#error "SH_ORDER > 4 not currently supported!"
#endif

struct sh_probe
{
    // RGB-D data.
    vec4 coef[SH_COEF_COUNT];
};

struct sh_lobe
{
    float coef[SH_COEF_COUNT];
};

const float sh_layer_height = 1.0f / float(SH_COEF_COUNT);

sh_lobe sh_basis(vec3 dir)
{
    vec3 d2 = dir * dir;
    return sh_lobe(float[](
        0.2820947917738781f
#if SH_ORDER >= 1
        , 0.4886025119029199f * dir.y,
        0.4886025119029199f * dir.z,
        0.4886025119029199f * dir.x
#endif
#if SH_ORDER >= 2
        , 1.0925484305920792f * dir.x * dir.y,
        1.0925484305920792f * dir.y * dir.z,
        0.3153915652525201f * (3 * d2.z - 1),
        1.0925484305920792f * dir.x * dir.z,
        0.5462742152960396f * (d2.x - d2.y)
#endif
#if SH_ORDER >= 3
        , 0.5900435899266435f * dir.y * (3 * d2.x - d2.y),
        2.8906114426405543f * dir.x * dir.y * dir.z,
        0.4570457994644658f * dir.y * (5 * d2.z - 1),
        0.3731763325901155f * dir.z * (5 * d2.z - 3),
        0.4570457994644658f * dir.x * (5 * d2.z - 1),
        1.4453057213202771f * dir.z * (d2.x - d2.y),
        0.5900435899266435f * dir.x * (d2.x - 3 * d2.y)
#endif
#if SH_ORDER >= 4
        , 2.503342941796705f * dir.x * dir.y * (d2.x - d2.y),
        1.770130769779931f * dir.y * dir.z * (3 * d2.x - d2.y),
        0.9461746957575602f * dir.x * dir.y * (7 * d2.z - 1),
        0.6690465435572893f * dir.y * dir.z * (7 * d2.z - 3),
        0.1057855469152043f * (35 * d2.z * d2.z - 30* d2.z + 3),
        0.6690465435572893f * dir.x * dir.z * (7 * d2.z - 3),
        0.4730873478787801f * (d2.x - d2.y) * (7 * d2.z - 1),
        1.770130769779931f * dir.x * dir.z * (d2.x - 3 * d2.y),
        0.6258357354491763f * (d2.x * d2.x - 6 * d2.x * d2.y + d2.y * d2.y)
#endif
    ));
}

sh_probe eval_sh_basis(vec3 dir, vec4 val)
{
    sh_probe res;
    sh_lobe l = sh_basis(dir);
    [[unroll]] for(int i = 0; i < SH_COEF_COUNT; ++i)
        res.coef[i] = val * l.coef[i];
    return res;
}

sh_lobe get_sh_cosine_lobe(vec3 dir)
{
    sh_lobe res = sh_basis(dir);
    res.coef[0] *= 1.0f;
#if SH_ORDER >= 1
    res.coef[1] *= 2.0f/3.0f;
    res.coef[2] *= 2.0f/3.0f;
    res.coef[3] *= 2.0f/3.0f;
#endif
#if SH_ORDER >= 2
    res.coef[4] *= 1/4.0f;
    res.coef[5] *= 1/4.0f;
    res.coef[6] *= 1/4.0f;
    res.coef[7] *= 1/4.0f;
    res.coef[8] *= 1/4.0f;
#endif
#if SH_ORDER >= 3
    res.coef[9]  = vec3(0);
    res.coef[10] = vec3(0);
    res.coef[11] = vec3(0);
    res.coef[12] = vec3(0);
    res.coef[13] = vec3(0);
    res.coef[14] = vec3(0);
    res.coef[15] = vec3(0);
#endif
#if SH_ORDER >= 4
    res.coef[16] *= -1/24.0f;
    res.coef[17] *= -1/24.0f;
    res.coef[18] *= -1/24.0f;
    res.coef[19] *= -1/24.0f;
    res.coef[20] *= -1/24.0f;
    res.coef[21] *= -1/24.0f;
    res.coef[22] *= -1/24.0f;
    res.coef[23] *= -1/24.0f;
    res.coef[24] *= -1/24.0f;
#endif

    return res;
}

sh_lobe get_ggx_specular_lobe(vec3 dir, float roughness)
{
    // Zonal harmonic coefficient curves were found using:
    // https://gist.github.com/juliusikkala/1031752338674b8afd8f1879e9105eca
    // then this curve was fit using python. The fits should be correct up to
    // the fourth decimal. Does not model anisotropy in any manner.
    vec4 zh = vec4(0.27793123f, 0.59372022f, 0.2400839f, 0.000250700498);

    zh += vec4(0.905501229f, 10.57518269f, 21.6480923f, 5.53340572f) * cos(
        fma(
            vec4(roughness),
            vec4(2.49220829f, 3.49132073f, 3.92510137f, 3.98902127f),
            vec4(2.88755638f, 0.56672964f, 0.50116945f, 0.705097221f)
        )
    );

    zh += vec4(1.98743320f, 9.52855312f, 19.90690569f, 3.23348085f) * cos(
        fma(
            vec4(roughness),
            vec4(1.79537159f, 3.58608449f, 4.01505002f, 4.63841986f),
            vec4(0.636261278f, 3.60689811f, 3.55551139f, 3.25144230f)
        )
    );

    zh += roughness * fma(
        inversesqrt(
            vec4(0.329615862f, 0.29109984f, 0.25094573f, 0.211655471f) +
            roughness*roughness
        ),
        vec4(1.54054310f, 4.35171889f, 7.58146856f, 9.84410536f),
        vec4(-4.73179141e-04f, -3.58678416f, -6.47567145f, -8.76804538f)
    );
    /* You can do the following too, but I think the precision is good enough.
    zh = clamp(zh, vec4(2.0f/3.0f, 1.0f/4.0f, 0, -1.0f/24.0f), vec4(1));
    */

    sh_lobe res = sh_basis(dir);
    //res.coef[0] *= 1;
#if SH_ORDER >= 1
    res.coef[1] *= zh[0];
    res.coef[2] *= zh[0];
    res.coef[3] *= zh[0];
#endif
#if SH_ORDER >= 2
    res.coef[4] *= zh[1];
    res.coef[5] *= zh[1];
    res.coef[6] *= zh[1];
    res.coef[7] *= zh[1];
    res.coef[8] *= zh[1];
#endif
#if SH_ORDER >= 3
    res.coef[9]  *= zh[2];
    res.coef[10] *= zh[2];
    res.coef[11] *= zh[2];
    res.coef[12] *= zh[2];
    res.coef[13] *= zh[2];
    res.coef[14] *= zh[2];
    res.coef[15] *= zh[2];
#endif
#if SH_ORDER >= 4
    res.coef[16] *= zh[3];
    res.coef[17] *= zh[3];
    res.coef[18] *= zh[3];
    res.coef[19] *= zh[3];
    res.coef[20] *= zh[3];
    res.coef[21] *= zh[3];
    res.coef[22] *= zh[3];
    res.coef[23] *= zh[3];
    res.coef[24] *= zh[3];
#endif

    return res;
}

vec4 calc_sh_value(in sh_probe sh, vec3 dir)
{
    sh_lobe bl = sh_basis(dir);
    vec4 res = vec4(0);
    [[unroll]] for(int i = 0; i < SH_COEF_COUNT; ++i)
        res += sh.coef[i] * bl.coef[i];
    return res;
}

vec3 calc_sh_irradiance(in sh_probe sh, vec3 normal)
{
    sh_lobe cl = get_sh_cosine_lobe(normal);
    vec3 irradiance = vec3(0);
    [[unroll]] for(int i = 0; i < SH_COEF_COUNT; ++i)
        irradiance += sh.coef[i].rgb * cl.coef[i];
    return max(irradiance, vec3(0));
}

vec3 calc_sh_ggx_specular(in sh_probe sh, vec3 dir, float roughness)
{
    sh_lobe cl = get_ggx_specular_lobe(dir, roughness);
    vec3 radiance = vec3(0);
    [[unroll]] for(int i = 0; i < SH_COEF_COUNT; ++i)
        radiance += sh.coef[i].rgb * cl.coef[i];
    return max(radiance, vec3(0));
}

vec3 calc_sh_convolution(in sh_probe sh, in sh_lobe cl)
{
    vec3 sum = vec3(0);
    [[unroll]] for(int i = 0; i < SH_COEF_COUNT; ++i)
        sum += sh.coef[i].rgb * cl.coef[i];
    return max(sum, vec3(0));
}

sh_probe sample_sh_grid(
    in sampler3D grid,
    vec3 grid_clamp,
    vec3 grid_pos,
    vec3 grid_normal
){
    sh_probe res;
    grid_pos = grid_pos*0.5f+0.5f;

    sh_probe pc;
#ifdef SH_INTERPOLATION_TRILINEAR
    grid_pos = clamp(grid_pos, grid_clamp.xyz, 1.0f - grid_clamp.xyz);
    grid_pos.y *= sh_layer_height;
    [[unroll]] for(int l = 0; l < SH_COEF_COUNT; ++l)
        pc.coef[l] = texture(grid, grid_pos + vec3(0, l*sh_layer_height, 0));
#else
    for(int l = 0; l < SH_COEF_COUNT; ++l)
        pc.coef[l] = vec4(0);

    const ivec3 grid_size = textureSize(grid, 0).xyz / ivec3(1,SH_COEF_COUNT,1);
    const vec3 inv_grid_size = 1.0f/vec3(grid_size);
    vec3 voxel_pos = grid_pos * vec3(grid_size) - vec3(0.5f);
    vec3 interp_pos = clamp(voxel_pos, vec3(0.0f), vec3(grid_size)-1.0f);
    ivec3 floor_pos = ivec3(interp_pos);
    ivec3 lo = clamp(floor_pos, ivec3(0), grid_size-1);
    ivec3 hi = clamp(floor_pos+1, ivec3(0), grid_size-1);

    float sum_weight = 0.0f;
    vec3 p_off = interp_pos-vec3(lo);
    vec3 m_off = 1.0f - p_off;

    const ivec3 sample_pos[8] = ivec3[8](
        ivec3(lo.x, lo.y, lo.z),
        ivec3(hi.x, lo.y, lo.z),
        ivec3(lo.x, hi.y, lo.z),
        ivec3(hi.x, hi.y, lo.z),
        ivec3(lo.x, lo.y, hi.z),
        ivec3(hi.x, lo.y, hi.z),
        ivec3(lo.x, hi.y, hi.z),
        ivec3(hi.x, hi.y, hi.z)
    );

    const float weight[8] = float[8](
        m_off.x*m_off.y*m_off.z,
        p_off.x*m_off.y*m_off.z,
        m_off.x*p_off.y*m_off.z,
        p_off.x*p_off.y*m_off.z,
        m_off.x*m_off.y*p_off.z,
        p_off.x*m_off.y*p_off.z,
        m_off.x*p_off.y*p_off.z,
        p_off.x*p_off.y*p_off.z
    );

    for(int k = 0; k < 8; ++k)
    {
        sh_probe lc;
        for(int l = 0; l < SH_COEF_COUNT; ++l)
        {
            ivec3 off = ivec3(0, l * grid_size.y, 0);
            lc.coef[l] = texelFetch(grid, sample_pos[k]+off, 0);
        }

        vec3 sample_dir = vec3(sample_pos[k])-interp_pos;
        float sample_dist = length(sample_dir);
        sample_dir /= sample_dist;
        float normal_factor = clamp((dot(grid_normal, sample_dir)+1.0f)*0.5f, 0.0f, 1.0f);
        float visibility = calc_sh_value(lc, -normalize(sample_dir*inv_grid_size)).w;
        float visibility_factor = clamp((visibility - sample_dist + 0.4f)*1.0f, 0.0f, 1.0f);
        float w = weight[k] * visibility_factor * normal_factor;

        for(int l = 0; l < SH_COEF_COUNT; ++l)
        {
            pc.coef[l] += lc.coef[l] * w;
        }
        sum_weight += w;
    }

    float inv_weight = sum_weight > 0 ? 1.0f / sum_weight : 0;
    for(int l = 0; l < SH_COEF_COUNT; ++l)
        pc.coef[l] *= inv_weight;
#endif

    return pc;
}

// Avoids temporary results and thus skips lots of registers (although, a wise
// optimizer should be able to skip those too.)
void sample_and_filter_sh_grid(
    in sampler3D grid,
    vec3 grid_clamp,
    vec3 grid_pos,
    vec3 sh_normal,
    vec3 sh_ref,
    float roughness,
    out vec3 diffuse,
    out vec3 reflection
){
    grid_pos = grid_pos*0.5f+0.5f;

    diffuse = vec3(0);
    reflection = vec3(0);

    sh_probe pc;
    grid_pos = clamp(grid_pos, grid_clamp.xyz, 1.0f - grid_clamp.xyz);
    grid_pos.y *= sh_layer_height;

    sh_lobe dl = get_sh_cosine_lobe(sh_normal);
    sh_lobe rl = get_ggx_specular_lobe(sh_ref, roughness);

    [[unroll]] for(int l = 0; l < SH_COEF_COUNT; ++l)
    {
        vec4 coef = texture(grid, grid_pos + vec3(0, l*sh_layer_height, 0));
        diffuse += coef.rgb * dl.coef[l];
        reflection += coef.rgb * rl.coef[l];
    }

    diffuse = max(diffuse, vec3(0));
    reflection = max(reflection, vec3(0));
}

#endif
