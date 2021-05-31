#==========================================================================
# Levenberg-Marquardt Minimizer
#==========================================================================
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(LevenbergMarquardtMinimizer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")


set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS graph log log_setup program_options system date_time filesystem thread)

find_package(OpenMP)

if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(CUDA 8 EXACT REQUIRED)

include_directories(
   ${Boost_INCLUDE_DIRS} 
   /home/zzhan896/VTE/IO)

source_group("CUDA Source Files" REGULAR_EXPRESSION "cu$")
source_group("CUDA Header Files" REGULAR_EXPRESSION "cuh$")

cuda_add_executable(
  LevenbergMarquardtMinimizer
  main.cpp
  CostFunction.hpp
  CurvatureCostFunction.cuh
  CurvatureCostFunction.hpp
  CurvatureCostFunction.inl
  DistanceCostFunction.cuh
  DistanceCostFunction.hpp
  DistanceCostFunction.inl
  DoLevenbergMarquardtMinimizer.hpp
  DoLevenbergMarquardtMinimizer.cu
  DoProjectionOntoLine.hpp
  DoProjectionOntoLine.cu
  DotProduct.cuh
  DualNumber.cuh
  LevenbergMarquardtMinimizer.hpp
  LevenbergMarquardtMinimizer.inl
  LinearCombination.hpp
  LinearCombination.inl
  LSQR.hpp
  ProjectionOntoLine.cuh
  ProjectionOntoLine.hpp
  ProjectionOntoLine.inl)

target_link_libraries(
  LevenbergMarquardtMinimizer
  IO
  ${Boost_LIBRARIES})

install(
  TARGETS LevenbergMarquardtMinimizer
  RUNTIME DESTINATION Bin)
