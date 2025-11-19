cmake_minimum_required(VERSION 3.10)
project(basednn C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# Backend options
option(ENABLE_WEBGPU "Enable WebGPU acceleration backend" ON)

# Include directories
include_directories(core/include)

# Source files
set(SOURCES
    core/src/tensor.c
    core/src/ops.c
    core/src/layer.c
    core/src/network.c
    core/src/optimizer.c
    core/src/registry.c
)

# Create library
add_library(basednn ${SOURCES})
target_link_libraries(basednn m)

# WebGPU backend (cross-platform)
if(ENABLE_WEBGPU)
    set(WEBGPU_SOURCES
        backend/webgpu/webgpu_backend.c
        backend/webgpu/webgpu_ops.c
    )
    
    target_sources(basednn PRIVATE ${WEBGPU_SOURCES})
    target_include_directories(basednn PRIVATE backend/webgpu)
    
    # Find WebGPU library (Dawn)
    # Users should set WEBGPU_DIR to the Dawn installation directory
    if(DEFINED ENV{WEBGPU_DIR})
        set(WEBGPU_DIR $ENV{WEBGPU_DIR})
    else()
        set(WEBGPU_DIR "$ENV{HOME}/.dawn/install")
    endif()
    
    if(EXISTS "${WEBGPU_DIR}/include/webgpu/webgpu.h")
        target_include_directories(basednn PRIVATE 
            ${WEBGPU_DIR}/include
            ${WEBGPU_DIR}/include/webgpu
        )
        
        # Link Dawn libraries
        target_link_libraries(basednn 
            ${WEBGPU_DIR}/lib/libwebgpu_dawn.a
            ${WEBGPU_DIR}/lib/libdawn_proc.a
        )
        
        # Dawn requires Metal framework and C++ runtime on macOS
        if(APPLE)
            target_link_libraries(basednn 
                "-framework Metal" 
                "-framework CoreGraphics" 
                "-framework Foundation" 
                "-framework QuartzCore" 
                "-framework IOKit" 
                "-framework IOSurface"
                "c++"
            )
        endif()
        
        target_compile_definitions(basednn PRIVATE HAS_WEBGPU)
        message(STATUS "WebGPU acceleration enabled using Dawn (${WEBGPU_DIR})")
    else()
        message(WARNING "WebGPU enabled but Dawn not found at ${WEBGPU_DIR}")
        message(WARNING "Build Dawn from: https://dawn.googlesource.com/dawn")
    endif()
endif()

# Enable testing
enable_testing()

# Unit tests
set(TEST_SOURCES
    core/tests/unit/test_tensor.c
    core/tests/unit/test_ops.c
    core/tests/unit/test_registry.c
    core/tests/unit/test_layer.c
    core/tests/unit/test_network.c
    core/tests/unit/test_optimizer.c
)

# Create individual test executables
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} basednn m)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Full tests
add_executable(xor core/tests/full/xor.c)
target_link_libraries(xor basednn m)
add_executable(mnist core/tests/full/mnist.c)
target_link_libraries(mnist basednn m)

# Add WebGPU flags to test executables if enabled
if(ENABLE_WEBGPU AND WEBGPU_DIR)
    target_compile_definitions(mnist PRIVATE HAS_WEBGPU)
    target_compile_definitions(xor PRIVATE HAS_WEBGPU)
endif()

# Examples (if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/custom_tensor_op.c")
    add_executable(custom_tensor_op examples/custom_tensor_op.c)
    target_link_libraries(custom_tensor_op basednn m)
endif()