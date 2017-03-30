#==========================================================================
# Levenberg-Marquardt Minimizer
#==========================================================================
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(LevenbergMarquardtMinimizer)

set(IO_INCLUDE_DIR "" CACHE PATH "Include directory for IO.")
set(IO_LIB "" CACHE FILEPATH "Library required to link IO.")

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS log program_options)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${IO_INCLUDE_DIR})
                        
find_package(CUDA REQUIRED)

# Find optional OpenMP.
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

source_group("CUDA Source Files" REGULAR_EXPRESSION "cu$")
source_group("CUDA Header Files" REGULAR_EXPRESSION "cuh$")

cuda_add_executable(
  ${PROJECT_NAME}
  main.cpp
  CostFunction.hpp
  CurvatureCostFunction.cuh
  CurvatureCostFunction.hpp
  CurvatureCostFunction.inl
  DistanceCostFunction.cuh
  DistanceCostFunction.hpp
  DistanceCostFunction.inl
  DotProduct.cuh
  DualNumber.cuh
  LevenbergMarquardtMinimizer.hpp
  LevenbergMarquardtMinimizer.inl
  LinearCombination.hpp
  LinearCombination.inl
  LSQR.hpp
  Minimize.cu
  Minimize.hpp
  ProjectionOntoLine.cuh
  ProjectionOntoLine.hpp
  ProjectionOntoLine.inl)

target_link_libraries(
  LevenbergMarquardtMinimizer
  ${Boost_LIBRARIES}
  ${IO_LIB})
