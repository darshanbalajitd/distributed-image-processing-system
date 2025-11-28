cmake_minimum_required(VERSION 3.20)
project(DistributedImageProcessingSystem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_MASTER "Build master executable" ON)
option(BUILD_WORKER "Build worker executable" ON)

add_subdirectory(src)