#ifndef PATH_GUIDING_GLSL
#define PATH_GUIDING_GLSL

#define MAX_DIMENSION 8

struct path_guiding_state
{
};

// x: new primary sample (v)
// y: jacobian from u to new sample
vec2 path_guide(inout path_guiding_data s, float u, int dimension);
// Returns the PDF for the given sample.
float path_guide_pdf(inout path_guiding_data s, float v, int dimension);
// Updates all samples along path
void path_guide_report(inout path_guiding_data s, float value);

#endif
