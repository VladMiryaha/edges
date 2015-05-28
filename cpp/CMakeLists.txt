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
find_package(Eigen3 3.2.0 REQUIRED)

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wno-sign-compare -pedantic") 
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DEIGEN_DONT_PARALLELIZE")

include_directories(${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include")

#add_executable(edge_boxes src/edge_boxes.cpp)
#target_link_libraries(edge_boxes ${OpenCV_LIBS})

add_executable(edge_boxes_mat src/edge_boxes_mat.cpp src/edge_nms.cpp src/util.cpp)
target_link_libraries(edge_boxes_mat ${OpenCV_LIBS} /home/samarth/Documents/MATLAB/cvmatio/lib/libcvmatio.so)
target_include_directories(edge_boxes_mat PRIVATE "/home/samarth/Documents/MATLAB/cvmatio/include")

add_executable(edge_boxes src/edge_boxes.cpp src/edge_detect.cpp src/edge_nms.cpp src/util.cpp)
target_link_libraries(edge_boxes ${OpenCV_LIBS} /home/samarth/research/gop_1.3/build/lib/contour/libcontour.a /home/samarth/research/gop_1.3/build/lib/imgproc/libimgproc.a /home/samarth/research/gop_1.3/build/lib/learning/liblearning.a /home/samarth/research/gop_1.3/build/lib/util/libutil.a)
target_include_directories(edge_boxes PRIVATE "/home/samarth/research/gop_1.3/lib/" ${EIGEN3_INCLUDE_DIR})
