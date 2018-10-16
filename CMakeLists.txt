cmake_minimum_required (VERSION 3.5)

set (CMAKE_CXX_STANDARD 11)
set("OpenCV_DIR" "/home/wojtek/opencv-3.4.1/build")
find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 REQUIRED)

add_definitions ("-Wall")
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_executable (sfm sfm.cpp)

target_link_libraries(sfm ${OpenCV_LIBS})
