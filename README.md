Tauray
=======

A rendering framework, with a focus on distributed computing, scalability and
low latency (real-time rendering). Uses C++17 and Vulkan. The framework is
primarily relying on the `VK_KHR_ray_tracing` extension, but there is an ugly fallback
rasterization mode that can be used on devices that do not have that extension.

Shaders are built during runtime because this lets Tauray specialize the source.
This is generally not a good practice in Vulkan, but here it can be used to
implement several features, such as user-defined distance field objects and
dynamically selecting the number of generated output buffers.

Tauray is licensed under LGPL version 2.1. You can find the license text in
[COPYING.LESSER](COPYING.LESSER). External dependencies in the `external`
folder have their own licenses specified either at the start of each file or as
separate license text files.


## Building

Clone the repository recursively with
`git clone --recursive https://github.com/vga-group/tauray/`.

Tauray has been tested on the Ubuntu 22.04 operating system. Building on Ubuntu
22.04 can be done as follows:

1. Install dependencies: `sudo apt install build-essential cmake libsdl2-dev libglm-dev libczmq-dev libnng-dev libcbor-dev vulkan-tools libvulkan-dev vulkan-validationlayers libxcb-glx0-dev glslang-tools`
2. `cmake -S . -B build`
3. `cmake --build build`
4. `build/tauray my_scene.glb`

Building with Windows is also possible but not recommended. You can use the
CMakeLists.txt with Visual Studio. A vcpkg.json is provided in this repository
to handle dependencies with Windows builds. Note that multi-GPU rendering is not
supported on Windows.

## Usage

To launch a simple interactive path tracing session with the included test model:

```bash
build/tauray test/test.glb
```

[See the user manual for more detailed usage documentation.](docs/MANUAL.md)

## Benchmarking with Tauray

To measure representative benchmarks with Tauray, please build a release build
and **disable validation**:

1. Delete your existing build directory (if applicable)
2. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
3. `cmake --build build`
4. `build/tauray --validation=off -t my_scene.glb`

You may also want to set `--force-double-sided` for better performance if your
scene does not require single-sided surfaces. Also, remember to set
`--max-ray-depth` appropriately for the type of benchmark, the default is quite
high.

By default, Tauray will use all GPUs with support for the required extensions.
If you wish to use a specific GPU on a multi-GPU system, use `--devices=0` (or
any other index; the first thing Tauray prints is the devices it picked.)

In the output with the `-t` flag, lines starting with `HOST: ` are total
frametimes, which you most likely want to measure. The first few frametimes of
a run can be weird as in-flight frames haven't been queued properly and some
initialization still runs, so please exclude those if possible.
`--warmup-frames` can also be used for this purpose.

[See the user manual for more detailed info on configuring Tauray for you benchmark setup.](docs/MANUAL.md)
