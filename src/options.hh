#ifndef TAURAY_OPTIONS_HH
#define TAURAY_OPTIONS_HH

//==============================================================================
// START OF OPTIONS
//==============================================================================
// If you want to add new command-line options, add them here. One entry in this
// list adds the parsing, options struct entry and help string all at once.
#define TR_OPTIONS \
    TR_INT_SOPT(width, 'w', "Set viewport width.", 1280, 0, INT_MAX) \
    TR_INT_SOPT(height, 'h', "Set viewport height.", 720, 0, INT_MAX) \
    TR_BOOL_SOPT(fullscreen, 'f', "Enable fullscreen mode.") \
    TR_BOOL_SOPT(vsync, 's', "Enable vertical synchronization.") \
    TR_BOOL_SOPT( \
        progress, 'p', \
        "Add a progress bar, useful for long offline renders." \
    ) \
    TR_BOOL_OPT(hdr, "Try to find an HDR swap chain.", false) \
    TR_BOOL_SOPT(timing, 't', "Print frame times.") \
    TR_SETINT_OPT(devices, \
        "Specify used device indices, -1 uses the first compatible device.") \
    TR_STRING_OPT(headless, \
        "Run the program without a window, capturing frames using the first "\
        "camera in the scene. The captured frames will be saved as " \
        "${headless}<index>.exr.", "") \
    TR_BOOL_OPT(headful, \
        "Headless-but-not mode that works around some GPU drivers that do " \
        "not expose multiple devices in non-headless Vulkan instances.", \
        false) \
    TR_ENUM_OPT(compression, headless::compression_type, \
        "Compression algorithm for use with captured frames. Not all EXR " \
        "viewers support all algorithms, and some algorithms can cause " \
        "massive delays in saving. Uncompressed images have very large " \
        "on-disk footprint. All available algorithms are lossless. " \
        "This option is respected only when using the EXR filetype.", \
        headless::PIZ, \
        {"zip", headless::ZIP}, \
        {"zips", headless::ZIPS}, \
        {"rle", headless::RLE}, \
        {"piz", headless::PIZ}, \
        {"none", headless::NONE} \
    )\
    TR_ENUM_OPT(distribution_strategy, tr::distribution_strategy, \
        "Set the the rendering distribution strategy", \
        tr::distribution_strategy::DISTRIBUTION_SHUFFLED_STRIPS, \
        {"duplicate", tr::distribution_strategy::DISTRIBUTION_DUPLICATE}, \
        {"scanline", tr::distribution_strategy::DISTRIBUTION_SCANLINE}, \
        {"shuffled-strips", tr::distribution_strategy::DISTRIBUTION_SHUFFLED_STRIPS} \
    )\
    TR_VECFLOAT_OPT(workload, \
        "Specify initial workload ratios per device, default is even workload.") \
    TR_ENUM_OPT(format, headless::pixel_format, \
        "Data format for the pixels in captured frames. " \
        "This option is respected only when using the EXR filetype. " \
        "PNG uses 8-bit rgba, BMP uses 8-bit rgb, and HDR uses 8 bits per " \
        "color and a shared 8-bit exponent, 32 bits per pixel in total.", \
        headless::RGB16, \
        {"rgb16", headless::RGB16}, \
        {"rgb32", headless::RGB32}, \
        {"rgba16", headless::RGBA16}, \
        {"rgba32", headless::RGBA32} \
    )\
    TR_ENUM_OPT(filetype, headless::image_file_type, \
        "Image format for the output image. EXR files are the default, but " \
        "if you just want to look at pretty pictures, go for png. The " \
        "special 'none' type can be used to omit output. Note that the dynamic " \
        "range of the HDR filetype is not utilized by default. The (default) " \
        "filmic tonemapper clamps the output to [0, 1]. E.g. the linear " \
        "tonemapper allows larger values.", \
        headless::EXR, \
        {"exr", headless::EXR}, \
        {"png", headless::PNG}, \
        {"bmp", headless::BMP}, \
        {"hdr", headless::HDR}, \
        {"raw", headless::RAW}, \
        {"none", headless::EMPTY} \
    )\
    TR_BOOL_OPT(skip_render, \
        "Very rarely useful option that disables rendering and frame output " \
        "when headless.", false) \
    TR_STRING_OPT(camera_log, \
        "Writes the camera parameter log (projection matrix + per-frame view " \
        "matrices in JSON)", "" \
    )\
    TR_STRUCT_OPT(camera_grid, \
        "Replaces the camera with a grid of cameras, W for horizontal size " \
        "of the grid and H for the vertical size. X and Y specify the " \
        "distance between grid cells.", \
        TR_STRUCT_OPT_INT(w, 1, 1, INT_MAX) \
        TR_STRUCT_OPT_INT(h, 1, 1, INT_MAX) \
        TR_STRUCT_OPT_FLOAT(x, 0.02f, 0.0f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(y, 0.02f, 0.0f, FLT_MAX) \
    ) \
    TR_STRUCT_OPT(camera_clip_range, \
        "Overrides camera clip range. If set to negative, will not override.", \
        TR_STRUCT_OPT_FLOAT(near, -1, 0.0f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(far, -1, 0.0f, FLT_MAX) \
    ) \
    TR_FLOAT_OPT(camera_grid_roll, \
        "Rolls the camera grid along the z axis by the given angle.", \
        0.0f, -360.0f, 360.0f) \
    TR_VEC3_OPT(camera_offset,\
        "Offsets the camera/camera grid from its native position. This is in " \
        "camera-local coordinates.", \
        vec3(0.0f), vec3(-FLT_MAX), vec3(FLT_MAX)) \
    TR_FLOAT_OPT(camera_recentering_distance,\
        "Camera recentering distance for multi-view setups. Also known as " \
        "distance to the zero disparity plane.", \
        INFINITY, 1e-6f, INFINITY) \
    TR_BOOL_SOPT(replay, 'r', "Enable replay mode.") \
    TR_FLOAT_OPT(framerate, \
        "Set framerate for the replay.", 60.0f, 0.0f, FLT_MAX) \
    TR_INT_OPT(frames, \
        "Forces the number of frames rendered in replay mode", \
        0, 0, INT_MAX) \
    TR_INT_OPT(skip_frames, \
        "Skips rendering on the given number of frames. Useful when " \
        "continuing an animation render that was interrupted earlier.", \
        0, 0, INT_MAX) \
    TR_INT_OPT(warmup_frames, \
        "Sets the number of frames rendered before the first recorded frame. " \
        "This exists to initialize temporal algorithms properly. Animations " \
        "are not playing during the warmup frames.", \
        0, 0, INT_MAX) \
    TR_STRING_OPT(envmap, "Path to a lat-long .hdr environment map.", "") \
    TR_FLAG_STRING_OPT(animation, \
        "Play the given animation for all objects in the scene, excluding " \
        "camera in interactive mode. If specified as a flag, the first found " \
        "animation is played for all objects in the scene.", \
        "") \
    TR_STRING_OPT(camera, \
        "Uses the named camera in the scene file instead of the first one.", \
        "") \
    TR_ENUM_OPT(tonemap, tonemap_stage::operator_type, \
        "Sets the tonemapping operator.", \
        tonemap_stage::FILMIC, \
        {"filmic", tonemap_stage::FILMIC}, \
        {"gamma-correction", tonemap_stage::GAMMA_CORRECTION}, \
        {"linear", tonemap_stage::LINEAR}, \
        {"reinhard", tonemap_stage::REINHARD}, \
        {"reinhard-luminance", tonemap_stage::REINHARD_LUMINANCE} \
    )\
    TR_FLOAT_OPT(exposure, \
        "Sets the exposure used in tonemapping.", 1.0f, 0.0f, FLT_MAX) \
    TR_FLOAT_OPT(gamma, \
        "Sets the gamma used in gamma-correction.", 2.2f, 0.0f, FLT_MAX) \
    TR_ENUM_OPT(renderer, options::renderer_option_type, \
        "Selects the renderer to use. Some options only work with certain " \
        "renderers.", \
        options::PATH_TRACER, \
        {"path-tracer", options::PATH_TRACER}, \
        {"direct", options::DIRECT}, \
        {"raster", options::RASTER}, \
        {"dshgi", options::DSHGI}, \
        {"dshgi-server", options::DSHGI_SERVER}, \
        {"dshgi-client", options::DSHGI_CLIENT}, \
        {"restir-di", options::RESTIR_DI}, \
        {"restir", options::RESTIR}, \
        {"albedo", feature_stage::ALBEDO}, \
        {"world-normal", feature_stage::WORLD_NORMAL}, \
        {"view-normal", feature_stage::VIEW_NORMAL}, \
        {"world-pos", feature_stage::WORLD_POS}, \
        {"view-pos", feature_stage::VIEW_POS}, \
        {"distance", feature_stage::DISTANCE}, \
        {"world-motion", feature_stage::WORLD_MOTION}, \
        {"view-motion", feature_stage::VIEW_MOTION}, \
        {"screen-motion", feature_stage::SCREEN_MOTION}, \
        {"instance-id", feature_stage::INSTANCE_ID} \
    )\
    TR_FLOAT_OPT(min_ray_dist, \
        "Sets the minimum distance a ray must travel. 0 can cause " \
        "self-intersection issues, so this should be more than that.", \
        0.0001f, 0.0f, FLT_MAX) \
    TR_INT_OPT(max_ray_depth, \
        "Sets the maximum number of times a ray can bounce or refract in its " \
        "path.", \
        8, 0, INT_MAX) \
    TR_INT_OPT(samples_per_pixel, \
        "Sets the number of samples per pixel for path tracing, or MSAA " \
        "samples for rasterization.", \
        1, 1, INT_MAX) \
    TR_INT_OPT(samples_per_pass, \
        "Sets the number of samples per pass for path tracing. This is " \
        "useful when command buffers would otherwise get bloated with " \
        "extremely high SPP counts. Too high values can cause driver " \
        "timeouts. ", \
        1, 1, 128) \
    TR_BOOL_OPT(shadow_terminator_fix, \
        "Enables support for a workaround for the shadow terminator issue, " \
        "compatible with the method used in Blender 2.90. This does not " \
        "conserve energy, but unless it's manually specified for a model in " \
        "the input scene, it has no effect.", \
        true) \
    TR_ENUM_OPT(film, film_filter, \
        "Chooses the film type for path tracing. Point sampling can enable " \
        "some optimizations in > 1spp situations, and may be required for " \
        "certain post-processing effects. The other methods implement " \
        "antialiasing.", \
        film_filter::POINT, \
        {"point", film_filter::POINT}, \
        {"box", film_filter::BOX}, \
        {"blackman-harris", film_filter::BLACKMAN_HARRIS} \
    )\
    TR_FLOAT_OPT(film_radius, \
        "Sets the sampling radius for the film sampling. This is in pixels " \
        "for most rendering methods.", \
        0.5f, 0.0f, FLT_MAX) \
    TR_FLOAT_OPT(russian_roulette, \
        "Enables russian roulette sampling with the given delta.", \
        0.0f, 1.000001f, FLT_MAX) \
    TR_FLOAT_OPT(indirect_clamping, \
        "Limits indirect light sample brightness, causing energy loss in " \
        "unlikely rays but reducing fireflies.", \
        0.0f, 0.0f, FLT_MAX) \
    TR_FLOAT_OPT(default_value, \
        "Sets the default value to be used in a feature buffer output when " \
        "the ray misses all geometry. INF and NAN are allowed!", \
        NAN, NAN, NAN) \
    TR_INT_OPT(pcf, \
        "Sets the number of PCF samples used for shadow filtering in the "\
        "raster renderer. 0 disables PCF filtering.", \
        64, 0, 64) \
    TR_INT_OPT(pcss, \
        "Sets number of samples used for blocker search in soft shadow " \
        "filtering in the raster renderer. 0 disables soft shadows. ", \
        32, 0, 64) \
    TR_FLOAT_OPT(pcss_minimum_radius, \
        "Sets the minimum radius used for soft shadows in the raster " \
        "renderer.", \
        0.0f, 0.0f, FLT_MAX) \
    TR_INT_OPT(shadow_map_cascades, \
        "Sets number of shadow map cascades used in the raster renderer. " \
        "Larger values render shadows further from the camera.", \
        4, 1, INT_MAX) \
    TR_INT_OPT(shadow_map_resolution, \
        "Sets the resolution of every shadow map in the raster renderer.", \
        2048, 1, INT_MAX) \
    TR_FLOAT_OPT(shadow_map_bias, \
        "Sets the bias term of every shadow map in the raster renderer.", \
        0.05f, 0.0f, FLT_MAX) \
    TR_FLOAT_OPT(shadow_map_depth, \
        "Sets the depth range of directional shadow maps in the raster " \
        "renderer", \
        100.0f, 0.0f, FLT_MAX) \
    TR_FLOAT_OPT(shadow_map_radius, \
        "Sets the X and Y ranges of directional shadow maps in the raster " \
        "renderer.", \
        10.0f, 0.0f, FLT_MAX) \
    TR_BOOL_OPT(sample_shading, \
        "Enables sample shading for rasterization, which is similar to " \
        "supersampling. The performance hit is very high, but sharp edges " \
        "from shading are eliminated.", \
        false) \
    TR_INT_OPT(samples_per_probe, \
        "Sets the number of samples per probe for baking spherical harmonics " \
        "probes.", \
        512, 1, INT_MAX) \
    TR_FLOAT_OPT(dshgi_temporal_ratio, \
        "Sets the exponential blend factor for DDISH-GI.", \
        0.01f, 0.0f, 1.0f) \
    TR_BOOL_OPT(alpha_to_transmittance, \
        "Crudely translates albedo + alpha into transmittance for all " \
        "materials in the scene that have a constant alpha factor below 1.0. " \
        "Textures with an alpha channel are untouched if the constant factor " \
        "is still 1.0.", \
        false) \
    TR_FLOAT_OPT(transmittance_to_alpha, \
        "Crudely translates transmittance into alpha for all " \
        "materials in the scene. The alpha is derived from transmittance such "\
        "that it is between 1 and the given number.", \
        -1.0f, 0.0f, 1.0f) \
    TR_BOOL_OPT(force_single_sided, \
        "Makes all materials single-sided, unless the have non-zero " \
        "transmittance (making those single-sided would break refraction.)", \
        false) \
    TR_BOOL_OPT(force_double_sided, \
        "Makes all materials double-sided.", \
        false) \
    TR_VEC3_OPT(ambient,\
        "Ambient lighting used in raster renderers.", \
        vec3(0.1f), vec3(0), vec3(FLT_MAX)) \
    TR_INT_OPT(sh_order,\
        "Spherical harmonics order used for light probe-based renderers.", \
        2, 0, 4) \
    TR_FLOAT_OPT(aspect_ratio, \
        "Forces a specific aspect ratio for the cameras.", \
        0.0f, 0.0f, FLT_MAX) \
    TR_FLOAT_OPT(fov, \
        "Overrides the original field of view for the camera(s). Specified " \
        "as vertical field of view in degrees.", \
        0.0f, 0.0f, FLT_MAX)\
    TR_INT_OPT(rng_seed, \
        "Sets the RNG seed instead of using zero.", \
        0, INT_MIN, INT_MAX) \
    TR_BOOL_OPT(tonemap_post_resolve, \
        "Apply tonemapping only after resolve. This only affects multisampled "\
        "rasterization", \
        false) \
    TR_BOOL_OPT(use_white_albedo_on_first_bounce, \
        "Force white albedo on the first bounce. This is handy for debugging " \
        "and needed by some denoising algorithms.", \
        false) \
    TR_BOOL_OPT(hide_lights, \
        "Hide area lights from view rays.", \
        false) \
    TR_BOOL_OPT(use_probe_visibility, \
        "Use a visibility term in SH probes for smarter interpolation. " \
        "This should fix lots of light leaking issues, but comes at a high " \
        "bandwidth cost.", \
        false) \
    TR_BOOL_OPT(use_z_pre_pass, \
        "Use a Z pre pass in rasterization. This can speed up rendering when " \
        "overdraw is a significant concern. There should be no visual " \
        "difference.", \
        true) \
    TR_ENUM_OPT(force_projection, options::projection_option_type, \
        "Forces a specific projection type on the primary camera.", \
        std::optional<tr::camera::projection_type>(), \
        {"off", std::optional<tr::camera::projection_type>()}, \
        {"perspective", tr::camera::PERSPECTIVE}, \
        {"orthographic", tr::camera::ORTHOGRAPHIC}, \
        {"equirectangular", tr::camera::EQUIRECTANGULAR} \
    ) \
    TR_BOOL_OPT(ply_streaming, \
        "Stream .ply model continuously. Assumes that new ply model data is " \
        "appended to the given file while this program runs.", \
        false) \
    TR_ENUM_OPT(up_axis, int, \
        "Rotates the given axis as the up axis in the scene.", \
        1, \
        {"x", 0}, \
        {"y", 1}, \
        {"z", 2} \
    )\
    TR_ENUM_OPT(display, options::display_type, \
        "Sets the display type. This is overridden by some options, such as "\
        "--headless.", \
        options::display_type::WINDOW, \
        {"headless", options::display_type::HEADLESS}, \
        {"window", options::display_type::WINDOW}, \
        {"openxr", options::display_type::OPENXR}, \
        {"looking-glass", options::display_type::LOOKING_GLASS}, \
        {"frame-server", options::display_type::FRAME_SERVER}, \
        {"frame-client", options::display_type::FRAME_CLIENT} \
    )\
    TR_INT_OPT(port, \
        "Sets the initial port number used for server modes. Further ports " \
        "are reserved from successive numbers if needed.", \
        3333, 0, 65535) \
    TR_STRING_OPT(connect, \
        "Sets the server address for client modes.", \
        "localhost:3333") \
    TR_FLOAT_OPT(throttle, \
        "Set framerate throttle. Does not affect frametime in replay mode.", \
        0.0f, 0.0f, FLT_MAX) \
    TR_BOOL_OPT(validation, \
        "Enable Vulkan validation layers.", \
        VULKAN_VALIDATION_ENABLED_BY_DEFAULT) \
    TR_INT_OPT(fake_devices, \
        "Multiply the number of devices for debugging multi-GPU rendering.", \
        0, 0, 16) \
    TR_ENUM_OPT(sampler, rt_stage::sampler_type, \
        "Sets the sampling method used in path tracing. Defaults to uniform " \
        "random.", \
        rt_stage::sampler_type::UNIFORM_RANDOM, \
        {"uniform-random", rt_stage::sampler_type::UNIFORM_RANDOM}, \
        {"sobol-z2", rt_stage::sampler_type::SOBOL_Z_ORDER_2D}, \
        {"sobol-z3", rt_stage::sampler_type::SOBOL_Z_ORDER_3D}, \
        {"sobol-owen", rt_stage::sampler_type::SOBOL_OWEN} \
    )\
    TR_SETINT_OPT(spatial_reprojection, \
        "Specify active viewport indices for lightfield rendering. Others " \
        "are inactivated when this flag is used. Inactive viewports aren't " \
        "rendered, but are being reprojected to.") \
    TR_FLOAT_OPT(temporal_reprojection, \
        "Ratio of temporal reuse for temporal reprojection. 0 disables " \
        "temporal reprojection.", \
        0, 0, 0.9999f) \
    TR_STRUCT_OPT(lkg_params, \
        "Sets parameters for rendering to a Looking Glass display. " \
        "v is the number of viewports, m is the distance of the plane of " \
        "convergence from the camera, d is the \"depthiness\", and " \
        "r is the view distance (relative to display size) used for "\
        "calculating the vertical FOV.", \
        TR_STRUCT_OPT_INT(viewports, 48, 1, INT_MAX) \
        TR_STRUCT_OPT_FLOAT(midplane, 2.0f, 0.001f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(depth, 2.0f, 0.001f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(relative_dist, 2.0f, 0.001f, FLT_MAX) \
    )\
    TR_STRUCT_OPT(lkg_calibration, \
        "Overrides calibration parameters for a Looking Glass display. " \
        "Can be used to run one such display without the USB connection. " \
        "These values can be found from the LKG_calibration folder if you " \
        "mount the display USB as a drive.", \
        TR_STRUCT_OPT_INT(display_index, -1, 0, INT_MAX) \
        TR_STRUCT_OPT_FLOAT(pitch, 0, -FLT_MAX, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(slope, 0, -FLT_MAX, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(center, 0, -FLT_MAX, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(fringe, 0, -FLT_MAX, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(viewCone, 0, 0.0f, FLT_MAX) \
        TR_STRUCT_OPT_INT(invView, 0, 0, 1) \
        TR_STRUCT_OPT_FLOAT(verticalAngle, 0, -FLT_MAX, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(DPI, 0, 0, FLT_MAX) \
        TR_STRUCT_OPT_INT(screenW, 0, 1, INT_MAX) \
        TR_STRUCT_OPT_INT(screenH, 0, 1, INT_MAX) \
        TR_STRUCT_OPT_INT(flipImageX, 0, 0, 1) \
        TR_STRUCT_OPT_INT(flipImageY, 0, 0, 1) \
        TR_STRUCT_OPT_INT(flipSubp, 0, 0, 1) \
    )\
    TR_STRUCT_OPT(taa, \
        "Sets parameters for temporal antialiasing.", \
        TR_STRUCT_OPT_INT(sequence_length, 0, 1, INT_MAX) \
    )\
    TR_ENUM_OPT(denoiser, options::denoiser_type, \
        "Selects the denoiser to use.", \
        options::denoiser_type::NONE, \
        {"none", options::denoiser_type::NONE}, \
        {"svgf", options::denoiser_type::SVGF}, \
        {"bmfr", options::denoiser_type::BMFR} \
    ) \
    TR_STRUCT_OPT(svgf_params, \
        "Parameters for the SVGF denoiser.\n" \
        "atrous-diffuse-iter: number of iterations of the atrous filter for the diffuse channel\n"\
        "atrous-spec-iter: number of iterations of the atrous filter for the specular channel\n"\
        "atrous-kernel-radius: atrous filter radius\n"\
        "sigma-l: luminance weight for atrous filter\n"\
        "sigma-z: depth weight for atrous filter\n"\
        "sigma-n: normal weight for atrous filter\n"\
        "min-alpha-color: controls temporal accumulation speed for diffuse and specular color\n" \
        "min-alpha-moments: controls temporal accumulation speed for moments used to drive the variance guidance\n", \
        TR_STRUCT_OPT_INT(atrous_diffuse_iter, 5, 1, 16) \
        TR_STRUCT_OPT_INT(atrous_spec_iter, 5, 0, 16) \
        TR_STRUCT_OPT_INT(atrous_kernel_radius, 2, 1, 16) \
        TR_STRUCT_OPT_FLOAT(sigma_l, 10.0f, 0.001f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(sigma_z, 1.0f, 0.001f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(sigma_n, 128.0f, 0.0f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(min_alpha_color, 0.02f, 0.001f, 1.0f) \
        TR_STRUCT_OPT_FLOAT(min_alpha_moments, 0.2f, 0.001f, 1.0f) \
    )\
    TR_BOOL_OPT(svgf_color_contains_direct_light, \
        "If set to true, SVGF output will be added to the contents of the color buffer instead of overwriting the color buffer.", \
        false \
    ) \
    TR_BOOL_OPT(accumulation, \
        "Whether to accumulate samples from multiple frames or not. " \
        "For interactive mode, samples are accumulated when the camera is " \
        "still. For offline rendering, the specified number samples is " \
        "reached by accumulating the same frame.", \
        false \
    ) \
    TR_ENUM_OPT(tri_light_mode, tri_light_sampling_mode, \
        "Sets the sampling method used for triangle area lights.", \
        tri_light_sampling_mode::SOLID_ANGLE, \
        {"area", tri_light_sampling_mode::AREA}, \
        {"solid-angle", tri_light_sampling_mode::SOLID_ANGLE}, \
        {"hybrid", tri_light_sampling_mode::HYBRID} \
    )\
    TR_BOOL_OPT(transparent_background, \
        "Replaces background with alpha transparency, regardless of " \
        "environment map usage.", \
        false \
    ) \
    TR_FLOAT_OPT(sample_point_lights, \
        "NEE sampling weight for point lights. If zero, punctual point " \
        "lights will not appear at all.", \
        1.0f, 0.0f, FLT_MAX)\
    TR_FLOAT_OPT(sample_directional_lights, \
        "NEE sampling weight for directional lights. If zero, punctual " \
        "directional lights will not appear at all.", \
        1.0f, 0.0f, FLT_MAX)\
    TR_FLOAT_OPT(sample_envmap, \
        "NEE sampling weight for the environment map, if present. Non-zero " \
        "values have a minor performance hit, and can make some rare scenes " \
        "noisier, but generally reduces noise significantly.", \
        1.0f, 0.0f, FLT_MAX)\
    TR_FLOAT_OPT(sample_emissive_triangles, \
        "NEE sampling weight for triangle lights in next event estimation. " \
        "All emissive triangles take part in this. Can result in less noise, " \
        "but has a slight performance hit.", \
        1.0f, 0.0f, FLT_MAX)\
    TR_ENUM_OPT(bounce_mode, bounce_sampling_mode, \
        "Sets the method used to pick bounce directions in path tracing.", \
        bounce_sampling_mode::MATERIAL, \
        {"hemisphere", bounce_sampling_mode::HEMISPHERE}, \
        {"cosine", bounce_sampling_mode::COSINE_HEMISPHERE}, \
        {"material", bounce_sampling_mode::MATERIAL} \
    )\
    TR_ENUM_OPT(multiple_importance_sampling, multiple_importance_sampling_mode, \
        "Sets the multiple importance sampling heuristic used in path tracing. ", \
        multiple_importance_sampling_mode::MIS_POWER_HEURISTIC, \
        {"off", multiple_importance_sampling_mode::MIS_DISABLED}, \
        {"balance", multiple_importance_sampling_mode::MIS_BALANCE_HEURISTIC}, \
        {"power", multiple_importance_sampling_mode::MIS_POWER_HEURISTIC} \
    )\
    TR_FLOAT_OPT(regularization, \
        "Sets the path space regularization gamma. Path regularization " \
        "reduces noise without clamping brightness. It still causes some " \
        "bias, but is a much less noticeable method.", \
        0.0f, 0.0f, 10.0f \
    ) \
    TR_STRUCT_OPT(depth_of_field, \
        "Sets depth of field parameters.", \
        TR_STRUCT_OPT_FLOAT(f_stop, 0, 0.001f, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(distance, 1, 0, FLT_MAX) \
        TR_STRUCT_OPT_FLOAT(sensor_size, 0.036f, 0.0f, FLT_MAX) \
        TR_STRUCT_OPT_INT(sides, 0, 3, INT_MAX) \
        TR_STRUCT_OPT_FLOAT(angle, 0, 0, 360) \
    ) \
    TR_ENUM_OPT(trace, tracing_record::trace_format, \
        "Sets the performance trace output format.", \
        tracing_record::SIMPLE, \
        {"simple", tracing_record::SIMPLE}, \
        {"trace-event-format", tracing_record::TRACE_EVENT_FORMAT} \
    )\
    TR_BOOL_OPT(scene_stats, \
        "Shows the scene stats including triangles count, static and dynamic objects count, texture count, and the number of light sources.", \
        false \
    ) \
    TR_BOOL_OPT(pre_transform_vertices, \
        "Pre-calculate transformed vertices into a separate buffer." \
        "Increases memory usage, but speeds up multi-bounce path tracing " \
        "performance.", \
        false \
    )\
    TR_ENUM_OPT(as_strategy, blas_strategy, \
        "Acceleration structure strategy; i.e. how geometries are assigned " \
        "into BLASes. per-material assigns each material of each model a " \
        "different BLAS. per-model assigns each model a BLAS. " \
        "static-merged-dynamic-per-model merges all static geometries into " \
        "one BLAS, while dynamic geometries are given per-model BLASes. " \
        "all-merged puts everything in one. Each approach has different " \
        "performance and memory tradeoffs.", \
        blas_strategy::STATIC_MERGED_DYNAMIC_PER_MODEL, \
        {"per-material", blas_strategy::PER_MATERIAL}, \
        {"per-model", blas_strategy::PER_MODEL}, \
        {"static-merged-dynamic-per-model", blas_strategy::STATIC_MERGED_DYNAMIC_PER_MODEL}, \
        {"all-merged", blas_strategy::ALL_MERGED_STATIC} \
    ) \
    TR_BOOL_OPT(silent, \
        "Disables general prints. Errors and timing data is still shown.", \
        false \
    ) \
    TR_STRING_OPT(timing_output, \
        "Sets the timing data output file. Default is stdout.", \
        "" \
    ) \
    TR_STRUCT_OPT(restir_di, \
        "The implementation is biased if sample_visibility = true and " \
        "shared_visibility = true. sample_visibility only has an effect when " \
        "shared_visibility = true.\n", \
        TR_STRUCT_OPT_INT(spatial_samples, 4, 0, 5000) \
        TR_STRUCT_OPT_FLOAT(max_confidence, 64, 0, 10000) \
        TR_STRUCT_OPT_INT(ris_samples, 8, 1, 5000) \
        TR_STRUCT_OPT_FLOAT(search_radius, 32, 0, 500) \
        TR_STRUCT_OPT_BOOL(shared_visibility, false) \
        TR_STRUCT_OPT_BOOL(sample_visibility, false) \
    ) \
    TR_ENUM_OPT(demo, int, \
        "Selects the demo type.", \
        0, \
        {"sun", 0}, \
        {"flashlight", 1} \
    ) \
    TR_BOOL_OPT(show_dude, \
        "Show 3d scanned dude", \
        true \
    )
//==============================================================================
// END OF OPTIONS
//==============================================================================

#include "math.hh"
#include "headless.hh"
#include "tonemap_stage.hh"
#include "path_tracer_stage.hh"
#include "rt_renderer.hh"
#include "rt_common.hh"
#include "feature_stage.hh"
#include "raster_stage.hh"
#include "camera.hh"
#include "scene.hh"
#include <string>
#include <variant>
#include <stdexcept>
#include <optional>
#include <climits>
#include <cfloat>
#include <filesystem>

namespace fs = std::filesystem;

#ifdef ENABLE_VULKAN_VALIDATION
static constexpr bool VULKAN_VALIDATION_ENABLED_BY_DEFAULT = true;
#else
static constexpr bool VULKAN_VALIDATION_ENABLED_BY_DEFAULT = false;
#endif

// Let's fix the standard library real quick
template<typename T, class... Types>
inline bool operator==(const T& t, const std::variant<Types...>& v)
{
    if(auto c = std::get_if<T>(&v))
        return *c == t;
    else return false;
}

template<typename T, class... Types>
inline bool operator==(const std::variant<Types...>& v, const T& t)
{ return operator==(t, v); }

namespace tr
{

class option_parse_error: public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct options
{
    enum class display_type
    {
        HEADLESS = 0,
        WINDOW,
        OPENXR,
        LOOKING_GLASS,
        FRAME_SERVER,
        FRAME_CLIENT
    };

    enum class denoiser_type
    {
        NONE = 0,
        SVGF,
        BMFR
    };

    enum basic_pipeline_type
    {
        PATH_TRACER = 0,
        DIRECT,
        RASTER,
        DSHGI,
        DSHGI_SERVER,
        DSHGI_CLIENT,
        RESTIR_DI,
        RESTIR
    };
    using renderer_option_type = std::variant<
        tr::options::basic_pipeline_type, feature_stage::feature>;

    using projection_option_type = std::optional<tr::camera::projection_type>;

    bool running = true;
    std::vector<std::string> scene_paths;

#define TR_BOOL_OPT(name, description, default) bool name = default;
#define TR_BOOL_SOPT(name, shorthand, description) bool name = false;
#define TR_INT_OPT(name, description, default, min, max) int name = default;
#define TR_INT_SOPT(name, shorthand, description, default, min, max) \
    int name = default;
#define TR_FLOAT_OPT(name, description, default, min, max) float name = default;
#define TR_STRING_OPT(name, description, default) std::string name = default;
#define TR_FLAG_STRING_OPT(name, description, default) \
    std::string name = default; \
    bool name##_flag = false;
#define TR_VEC3_OPT(name, description, default, min, max) vec3 name = default;
#define TR_ENUM_OPT(name, type, description, default, ...) type name = default;
#define TR_SETINT_OPT(name, description) std::set<int> name;
#define TR_VECFLOAT_OPT(name, description) std::vector<double> name;
#define TR_STRUCT_OPT(name, description, ...) struct { __VA_ARGS__ } name;
#define TR_STRUCT_OPT_INT(name, default, min, max) int name = default;
#define TR_STRUCT_OPT_FLOAT(name, default, min, max) float name = default;
#define TR_STRUCT_OPT_BOOL(name, default) bool name = default;
    TR_OPTIONS
#undef TR_BOOL_OPT
#undef TR_BOOL_SOPT
#undef TR_INT_OPT
#undef TR_INT_SOPT
#undef TR_FLOAT_OPT
#undef TR_STRING_OPT
#undef TR_FLAG_STRING_OPT
#undef TR_VEC3_OPT
#undef TR_ENUM_OPT
#undef TR_SETINT_OPT
#undef TR_VECFLOAT_OPT
#undef TR_STRUCT_OPT
#undef TR_STRUCT_OPT_INT
#undef TR_STRUCT_OPT_FLOAT
#undef TR_STRUCT_OPT_BOOL
};

void parse_command_line_options(char** argv, options& opt);
bool parse_config_options(const char* config_str, fs::path relative_path, options& opt);
bool parse_command(const char* config_str, options& opt);
void print_command_help(const std::string& command);
void print_help(const char* program_name);
void print_options(options& opt, bool full);

}

#endif

