# Mininimum required CMake version
cmake_minimum_required(VERSION 2.8)

# Project name
project(edge_boxes)

# Find the OpenCV installation
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Other directories where header files for linked libraries can be found
include_directories(${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include")

# executable produced as a result of compilation
add_executable(edge_boxes src/edge_boxes.cpp)
# libraries to be linked with this executable - OpenCV
target_link_libraries(edge_boxes ${OpenCV_LIBS})
