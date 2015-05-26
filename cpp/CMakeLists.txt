cmake_minimum_required(VERSION 2.8)
IF(DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
    build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
    Release RelWithDebInfo MinSizeRel.")
ELSE()
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
    options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
    RelWithDebInfo MinSizeRel.")
ENDIF()

project(edge_boxes)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include" "/home/samarth/Documents/MATLAB/cvmatio/include")

add_executable(edge_boxes src/edge_boxes.cpp)
target_link_libraries(edge_boxes ${OpenCV_LIBS})

add_executable(edge_boxes_mat src/edge_boxes_mat.cpp)
target_link_libraries(edge_boxes_mat ${OpenCV_LIBS} /home/samarth/Documents/MATLAB/cvmatio/lib/libcvmatio.so)
