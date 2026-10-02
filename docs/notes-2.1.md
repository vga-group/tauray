Tauray v2.1 release
---------------------

## New features

- Radiance cascade path guiding (RCPG) (`--renderer=rc`, `--renderer=rc-restir`).
- Light tree for sampling point, spot and emissive triangle lights (`--enable-light-tree`).
    - This is a very rudimentary and simplistic light BVH.
- Multiview (camera grid) rendering composited into a single window (`--camera-grid`).
- Headless MSE comparison mode against reference images (`--filetype=mse`, `--reference=<path>`).
- Automatic SPP calibration to a target frame time (`--auto-spp=<ms>`).
- Scenes can be specified in config and preset files (`scene=<path>`).
- The path tracer is now a compute pipeline using ray queries instead of a ray tracing pipeline.
    - Experimentally, this appeared to improve performance on Nvidia GPUs (counterintuitively).
- glTF loader adds a white 1x1 placeholder texture for missing textures.
    - This allows unconditional texture loads in shaders, improving performance. 

## Changed defaults and behavior

- `--devices` now defaults to a single device (`-1`, the first compatible GPU) instead of using every ray tracing-capable GPU. Multi-GPU rendering must be requested explicitly.
    - This change was made due to modern integrated GPUs supporting ray tracing, which caused inadvertent multi-GPU configurations with the old default.
- `--shadow-terminator-fix` now defaults to `off` (was `on`).
- Materials with non-zero transmittance are no longer classified as potentially transparent (only alpha in albedo factor/texture counts).
    - Such materials do not need this flag because they do not use any-hit intersections.

## Removed

- Debian packaging (`debian/`) was removed as untested and unused.
- CMake no longer runs `git submodule update --init --recursive` automatically.
    - Submodules must be initialized manually (`git submodule update --init --recursive`).
- Removed `--spatial-reprojection` \& `--temporal-reprojection` as unmaintained and broken.

## Build and portability

- Migrated from SDL2 to SDL3.
- Updated dependencies: Vulkan-Headers, SPIRV-Headers, glslang, OpenXR-SDK, VMA, and GLM.
- Build now targets Ubuntu 26.04; updated `apt` package dependencies in README and manual.
- Third-party sources are warning-suppressed so that GCC 15 builds cleanly.
- Added Wayland include path for the OpenXR loader.

## Bug fixes

- Fixed `--accumulation` having no effect in replay/headless rendering, where accumulation was reset before every frame.
- Blender add-on modernized for Blender 5.2.
- `TR_data` material data (transmission, IOR, emission) is no longer used in favor of standard glTF extensions.
    - The extension still exists but is only used to add radius to point lights, solid angle to directional lights, and probe grids for DDISH-GI.
- Fixed environment map alias table generation reading past-the-end of pixel array.
- Fixed binary semaphores being signaled and waited with the wrong index.
- Added missing barriers after acceleration structure builds, instance uploads and previous point light buffer copies.
- Fixed the raster subpass dependency to include early fragment tests and depth/stencil writes.
- Fixed broken GPU timings caused by the frame delay stage.
- Fixed storage image format validation errors.
- Fixed errors during headless image saving freezing the program with no message.
- Fixed a crash when closing the Looking Glass context.
- Fixed camera switching with PageUp/PageDown indexing the camera list before the index was wrapped.
- Fixed headless output directory creation when the output path has no directory component.
- Fixed a missing space in the glTF scene loading log message.
- Fixed spurious timestamp errors.
- Fixed running out of timestamp slots in some multi-view setups.

## Documentation

- User manual updated to features introduced in Tauray 2.1.
- README notes that the project is currently mothballed.

---

# Acknowledgements

This work was supported by the Research Council of Finland under Grant 336357 (PROFI 6 - TAU Imaging Research Platform) and Grant 351623.
