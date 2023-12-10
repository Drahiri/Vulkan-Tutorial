set(VERT_SHADER_FILES
    shader.vert
)

set(FRAG_SHADER_FILES
    shader.frag
)

set(SHADER_FILES ${VERT_SHADER_FILES} ${FRAG_SHADER_FILES})

function(compile_shader INPUT_SHADER OUTPUT_SPIRV)
    add_custom_command(
        OUTPUT ${OUTPUT_SPIRV}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_SHADER} -o ${OUTPUT_SPIRV}
        DEPENDS ${INPUT_SHADER}
        COMMENT "Compiling ${INPUT_SHADER} to SPIR-V"
    )
endfunction()

foreach(SHADER_FILE ${SHADER_FILES})
    set(SPIRV_FILE ${CMAKE_BINARY_DIR}/shaders/${SHADER_FILE}.spv)
    compile_shader(${SHADER_FILE} ${SPIRV_FILE})
    list(APPEND SPIRV_FILES ${SPIRV_FILE})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})