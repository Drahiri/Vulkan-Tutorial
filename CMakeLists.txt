cmake_minimum_required(VERSION 3.25)
project(VulkanTutorial VERSION 1.0.0)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_subdirectory(shaders)

find_package(Vulkan REQUIRED)

set(INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/extern/include/" "${Vulkan_INCLUDE_DIRS}")
set(LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/extern/lib/" "${Vulkan_LIBRARIES}")
set(LIBRARIES glfw3 vulkan dl pthread X11 Xxf86vm Xrandr Xi)

add_executable(${PROJECT_NAME} src/main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets SYMBOLIC)

add_dependencies(${PROJECT_NAME} Shaders)