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
