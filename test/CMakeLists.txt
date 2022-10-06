find_program(PYTHON_PROG python3 OPTIONAL)
find_program(COMPARE_PROG compare OPTIONAL)

#message(SEND_ERROR ${PYTHON_PROG})
function(renderer_test renderer stage tolerance)
    add_executable("${renderer}_crash"
        crash_test.cc
    )
    target_link_libraries("${renderer}_crash" PUBLIC tauray-core SDL2::SDL2main)
    target_compile_definitions("${renderer}_crash" PUBLIC "TEST_RENDERER=tr::${stage}")
    target_include_directories("${renderer}_crash" PUBLIC "${CMAKE_SOURCE_DIR}/src")
    add_test(NAME "${renderer}_crash_test"
        COMMAND "${renderer}_crash"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(NAME "validate_${renderer}_test"
        COMMAND ${PYTHON_PROG}
            "${CMAKE_CURRENT_SOURCE_DIR}/validate_render.py"
            "--executable=${CMAKE_BINARY_DIR}/tauray"
            "--scene=${CMAKE_CURRENT_SOURCE_DIR}/test.glb"
            "--renderer=${renderer}"
            "--reference=${CMAKE_CURRENT_SOURCE_DIR}/references/validate_${renderer}.exr"
            "--metric=mse"
            "--tolerance=${tolerance}"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endfunction()

renderer_test("raster" "options::RASTER" 1)
renderer_test("path-tracer" "options::PATH_TRACER" 10000)
renderer_test("whitted" "options::WHITTED" 1)
renderer_test("dshgi" "options::DSHGI" 10)
renderer_test("albedo" "feature_stage::ALBEDO" 1)
renderer_test("world-normal" "feature_stage::WORLD_NORMAL" 1)
renderer_test("view-normal" "feature_stage::VIEW_NORMAL" 1)
renderer_test("world-pos" "feature_stage::WORLD_POS" 1)
renderer_test("view-pos" "feature_stage::VIEW_POS" 1)
renderer_test("distance" "feature_stage::DISTANCE" 1)
