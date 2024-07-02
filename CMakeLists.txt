cmake_minimum_required(VERSION 3.10)


project(unet)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Усполняемый файл
add_executable(unet main.cpp)
