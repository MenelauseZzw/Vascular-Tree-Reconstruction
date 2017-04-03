#==========================================================================
# Levenberg-Marquardt Minimizer
#==========================================================================
project(LevenbergMarquardtMinimizer)

include_directories(${Boost_INCLUDE_DIRS})

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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