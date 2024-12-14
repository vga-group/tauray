#ifndef BMFR_GLSL
#define BMFR_GLSL
#define BLOCK_SIZE 32
#define LOCAL_SIZE 16
#define BLOCK_EDGE_LENGTH 32
#define BLOCK_PIXELS (BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)
#define FEATURE_COUNT 10

#define BLOCK_OFFSETS_COUNT 16u
ivec2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = ivec2[](
   ivec2( -14, -14 ),
   ivec2(   4,  -6 ),
   ivec2(  -8,  14 ),
   ivec2(   8,   0 ),
   ivec2( -10,  -8 ),
   ivec2(   2,  12 ),
   ivec2(  12, -12 ),
   ivec2( -10,   0 ),
   ivec2(  12,  14 ),
   ivec2(  -8, -16 ),
   ivec2(   6,   6 ),
   ivec2(  -2,  -2 ),
   ivec2(   6, -14 ),
   ivec2( -16,  12 ),
   ivec2(  14,  -4 ),
   ivec2(  -6,   4 )
);

layout(push_constant) uniform push_constant_buffer
{
    ivec2 workset_size; // x and y block counts
    ivec2 size;
} control;

float scale(float value, float min, float max)
{
    if (abs(max - min) > 1.0)
    {
        return (value - min) / (max - min);
    }
    return value - min;
}

#endif
