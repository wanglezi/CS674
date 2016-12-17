cmake_minimum_required(VERSION 3.6)
project(sparse_opt_parallel_single_machine)
include_directories(./include /usr/local/include/Eigen)
set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp include/common.h src/common.cpp include/solve.h src/solve.cpp include/topK.h src/topK.cpp)

add_executable(sparse_opt_parallel_single_machine ${SOURCE_FILES} include/common.h src/common.cpp include/solve.h include/topK.h src/topK.cpp)