cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(reduction LANGUAGES CXX CUDA)
add_executable(reduction main.cu)
set_property(TARGET reduction PROPERTY CUDA_STANDARD 11)
