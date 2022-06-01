
cmake_minimum_required(VERSION 3.17)

project(CS205Project)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "D:\\opencv\\opencv\\mingw-build\\install")

#find_library(OpenCV)

find_package(OpenCV)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD)

add_executable(CS205Project main.cpp Matrix.h)

target_link_libraries(CS205Project ${OpenCV_LIBS})