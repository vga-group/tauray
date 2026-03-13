#version 450

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

layout(location = 0) out int orientation;
layout(location = 1) out vec2 z_range;

void main()
{
    vec3 a = gl_in[0].gl_Position.xyz;
    vec3 b = gl_in[1].gl_Position.xyz;
    vec3 c = gl_in[2].gl_Position.xyz;

    vec3 n = abs(cross(a-b, a-c));

    if (n.z > n.y && n.z > n.x)
    {
        orientation = 0;
    }
    else if (n.y > n.x)
    {
        a = a.xzy;
        b = b.xzy;
        c = c.xzy;
        orientation = 1;
    }
    else
    {
        a = a.yzx;
        b = b.yzx;
        c = c.yzx;
        orientation = 2;
    }

    a.z = a.z * 0.5 + 0.5;
    b.z = b.z * 0.5 + 0.5;
    c.z = c.z * 0.5 + 0.5;

    z_range = vec2(min(min(a.z,b.z),c.z), max(max(a.z, b.z), c.z));

    gl_Position = vec4(a, 1.0);
    EmitVertex();

    gl_Position = vec4(b, 1.0);
    EmitVertex();

    gl_Position = vec4(c, 1.0);
    EmitVertex();

    EndPrimitive();
}
