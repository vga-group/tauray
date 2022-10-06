#version 450

vec2 pos[6] = vec2[](
    vec2(1, -1), vec2(1, 1), vec2(-1, 1), vec2(-1, 1), vec2(-1, -1), vec2(1, -1)
);

void main()
{
    vec2 p = pos[gl_VertexIndex];
    gl_Position = vec4(p, 0.0f, 1.0f);
}
