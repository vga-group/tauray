#version 450

layout(binding = 0, r32ui) uniform writeonly uimage3D occupancy;

layout(location = 0) flat in int orientation;

void main()
{
    ivec3 occupancy_size = imageSize(occupancy);

    float depth_range = abs(dFdy(gl_FragCoord.z)) + abs(dFdx(gl_FragCoord.z));

    ivec2 p = ivec2(gl_FragCoord.xy);

    p.y = occupancy_size.y-1-p.y;

    int min_z = int((gl_FragCoord.z - depth_range * 0.5) * occupancy_size.z);
    int mid_z = int(gl_FragCoord.z * occupancy_size.z);
    int max_z = int((gl_FragCoord.z + depth_range * 0.5) * occupancy_size.z);

    min_z = max(min_z, mid_z-1);
    max_z = min(max_z, mid_z+1);

    for (int i = min_z; i <= max_z; ++i)
    {
        ivec3 q = ivec3(p, i);

        if (orientation == 1) q = q.xzy;
        if (orientation == 2) q = q.zxy;

        imageStore(occupancy, q, ivec4(1));
    }
}
