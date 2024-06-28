cmake_minimum_required(VERSION 3.10)
project(U-Net)

find_package(OpenCV REQUIRED)

add_executable(U-Net main.cpp)
target_link_libraries(U-Net ${OpenCV_LIBS})
