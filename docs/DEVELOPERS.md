# Developing Tauray

## Adding a new rendering technique

Renderers are split into two parts in Tauray: the renderer class and the
pipeline class. The renderer class ties multiple pipelines together to form the 
final image. It is also responsible for load balancing.

You may not need to add a new renderer; for most ray-tracing-based techniques
(MLT, BDPT, etc.) you should be able to simply specialize `rt_renderer` for your
class. If you choose to do so, you get the benefit of all the post-processing
and distributed computing capabilities handled by `rt_renderer`. Other, more
novel kinds of pipelines are more likely to need a new renderer.

You will always need to create a new pipeline, though. When creating a pipeline,
you can choose from many different base classes. `gfx_pipeline` is the most
generic pipeline that should not pose any limitations to what you can do.
`rt_pipeline` is a more suitable base class for ray-tracing based methods, it
handles setting up all scene data. Further, if you want to make a camera-based
ray-tracing pipeline that renders to a rectangular image buffer, you should
derive from `rt_camera_pipeline` which also handles the camera buffer and output
image. See `feature_pipeline` for a simple example of the usage.

Whatever you do, please refrain from trying to tack on new methods as a flag to
existing pipelines. For example, do not implement bidirectional path tracing
as a flag to `path_tracer_pipeline`.

## Coding style

Tauray has a very specific coding style, please follow it as well as possible.
We would like to use clang-format, but there are some glaring omissions in it
which prevent its usage. Until they are implemented, it's up to you to write
the code in the proper style.

The following example should cover most of it. If there's some structure that is
missing, try to find an existing sample of it in the source code. Always indent
with 4 spaces (never tabs) and wrap at 80 columns. Almost all editors have an
option for showing a line at a specific column, so set one up for this.

Header:
```cpp
// Use //-comments instead of multiline if you can, but multiline is ok if it
// makes the code clearer.
// Namespace contents are not indented, because practically everything is in the
// same 'tr' namespace.
namespace tr
{

// Free functions are very normal.
unsigned discombobulate();

// Split lines at 80 columns, like this.
void discombobulate_cool_and_good_version(
    int cool,
    const texture& megatexture,
    asdf* good
);

// Short classes like this
class basic_thingy
{
// Access specifiers are typically in this order and not indented more than the
// class itself (similar to all other labels).
public:
    // Write constructors and destructors at the start.
    basic_thingy();

    // Please think about the copy and move constructors. Mark them as delete if
    // you're not sure. A lot of assets aren't deletable but are movable. It is
    // also good to use "= default" to explicitly use C++-generated default
    // versions of move and copy.
    basic_thingy(const basic_thingy& other) = delete;
    basic_thingy(basic_thingy&& other);

    // Write a virtual destructor if the class has virtual members. "= default"
    // is fine if you don't actually want to do anything in it.
    virtual ~basic_thingy() = default;

    // Write other functions here, ordered by subject. Obey const-correctness
    // (mark as const if the function doesn't modify this class).
    void do_thing(int arg) const;

    // Long member function declarations are split just like free functions.
    virtual void calculate_lots_of_super_cool_very_long_this_that_go_too_long(
        int arg1,
        int arg2,
        int arg3
    ) = 0;

protected:
    // Use protected things to mostly for providing handy class-specific helper
    // functions that subclasses may need.
    void perform_internal_mangling();

private:
    // Don't use m_ etc. prefixes in members. Just write them like a normal
    // human. Using default values instead of setting them in the constructor is
    // OK, especially with pointers (set them to nullptr just in case).
    int member = 0;
};

// Derive like this.
class thingy: public basic_thingy
{
public:
    // Mark overridden virtual functions as such.
    void calculate_lots_of_super_cool_very_long_this_that_go_too_long(
        int arg1,
        int arg2,
        int arg3
    ) override;
};

// Very rare but technically possible situation where we have to split up
// derives.
class very_complicated_super_important_cool_thingy_yes_totally_cool_name_yeah
:   public basic_thingy,
    public another_thingy
{
private:
    // Comment the key and item entries of maps as if they were arguments, this
    // makes it clearer for the reader. Doing the same with other containers is
    // also OK, but rarely necessary.
    std::map<std::string /* name */, int /* price */> products;

    // Like here, it doesn't make sense to comment what the strings are since
    // the name of the vector already documents that they are names.
    std::vector<std::string> all_names;
};

// Use a struct when the class version of it would just be a pile of get() and
// set() that would make hiding the entries meaningless. It's OK to have
// constructors, destructors and member functions in structs too (but not
// necessary)!
struct i_have_nothing_to_hide
{
    i_have_nothing_to_hide();

    int credit_card_number[16];
    int three_digits_on_the_back[3];
    std::time_t expiration_date;
};

}
```

Source:
```cpp
#include "my_header.hh"

namespace
{
// Write functions (or classes) that are only ever used inside this file in an
// unnamed namespace. This is an optimization thing but also a good habit to
// prevent "namespace pollution".

// Using a namespace is OK here, since it doesn't leak outside this namespace
// scope and it lets you forget about everything being in the tr-namespace.
using namespace tr;
}

// Instead of writing tr:: in front of every name, we just do the implementation
// in a namespace too. Again, no indent for this.
namespace tr
{

void discombobulate()
{
    // Infinite loops look like this.
    for(;;)
    {
    }

    // Use the "new" ranged-for syntax whenever possible. "auto" is OK here if
    // you're a bit lazy or the type is very difficult, prefer the actual name
    // if it's short or makes the code much clearer.
    for(auto& p: container)
    {
    }

    // Old-style for loops are OK too. (iterators are cool too but you don't
    // really have a need to use them often due to ranged-for syntax).
    for(int i = 0; i < 100; ++i)
    {
    }

    // Multiline for-loop when 80 columns aren't enough:
    for(
        int thing = 0;
        thing < 129849384;
        thing += rand()%2
    ){
    }

    // Use while-loops only when no for-loop makes sense.
    while(condition)
    {
    }

    // Use a switch when behavior depends on an enum value or a very limited set
    // of numbers.
    int a = rand();
    switch(a % 4)
    {
    // Just like other labels, cases aren't indented separately.
    case 0:
        do_a();
        break;
    case 1:
        do_b();
        break;
    default:
        break;
    }

    std::string id = get_id();

    // Usually, prefer using braces with if-conditions. It's OK to leave them
    // out if the 'if' doesn't have elses and can fit on a single line.
    if(id == "cool") do_a();

    if(id == "what")
    {
        do_a();
    }
    else if(id == "yes")
    {
        do_b();
    }
    else
    {
        do_a();
        do_b();
    }

    // Long if-conditions are split like this.
    if(
        (id == "no" || a == 3) &&
        can_i_get_permission_to_do_this_please(a) &&
        pretty_please()
    ){
    }

    // Split function calls to multiple lines like this.
    discombobulate_cool_and_good_version(
        ((a + 1) * 100) << 10,
        get_megatexture(),
        120314858
    );

    return 4;
}

// Line splitting function arguments is the same in definitions as in
// declarations.
void tr::thingy::calculate_lots_of_super_cool_very_long_this_that_go_too_long(
    int arg1,
    int arg2,
    int arg3
){
}

// The constructor initializer list is split like this.
tr::i_have_nothing_to_hide::i_have_nothing_to_hide()
:   credit_card_number({0}),
    three_digits_on_the_back({1,2,3})
    expiration_date(time(nullptr))
{
}

}
```

## Known artefacts

Below is a list of known rendering artefacts with explanations if known. This
list exists to save time in the future if someone decides to work on the
artefacts.

* Dark/too light band along concave edges
  * Caused by traceRayEXT's Tmin parameter which offsets the ray by the given
    amount to avoid self-intersection. Because of it, the rays may also step
    past the actual next intersection, which often happens with concave edges.
* Light leaking inside enclosed volumes
  * Caused by the concave edge problem, shadow rays in the corners of the
    volume step outside of the enclosed volume. This leaked light from the
    corners then propagates in path tracing.
* Fireflies
  * Low probability events with high contribution cause fireflies. In
    other words, the importance sampling grossly mismatched the actual lighting
    response.
  * Another possible cause is the russian roulette, if enabled.
* NaN values
  * Most likely a division by zero somewhere. Fix it immediately!

## FAQ

1. What's `vkm`?
  - It is a container for automatically managed Vulkan resources. Once they go
    out of scope, the resource is not immediately destroyed. Instead, it is
    queued to be destroyed once the current frame is no longer in-flight and the
    GPU cannot use that resource anymore. You should use it with all Vulkan
    resources that need deallocation (vkDestroy*, vkFree*)
