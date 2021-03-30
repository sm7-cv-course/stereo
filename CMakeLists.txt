cmake_minimum_required(VERSION 3.5)

project(stereo-cpp-naive LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
    #SET(OpenCV_DIR /home/evgeny/soft/opencv/opencv/release)
    LINK_DIRECTORIES(/usr/local/lib/)
endif (UNIX)

find_package(PkgConfig)
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV not found")
endif()

set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_calib3d opencv_features2d opencv_xfeatures2d opencv_flann opencv_photo opencv_aruco) #opencv_contrib

add_executable(stereo-cpp-naive imagepairmatcher.cpp imageshiftestimator.cpp main.cpp)

if (UNIX)
    target_link_libraries(stereo-cpp-naive ${OpenCV_LIBS} stdc++ m)
endif()
