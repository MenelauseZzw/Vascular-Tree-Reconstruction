cmake_minimum_required(VERSION 2.8.11)
project(LevenbergMarquardtMinimizer)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Find Google gflags library
set(GFLAGS_NAMESPACE gflags)
find_package(gflags REQUIRED)
                                       
find_package(CUDA REQUIRED)

set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS CXX)

include_directories(
  ${GFLAGS_INCLUDE_DIRS} 
  ${HDF5_INCLUDE_DIR} 
  "$ENV{NVCUDASAMPLES_ROOT}/common/inc"
  )

cuda_add_executable(
  LevenbergMarquardtMinimizer
  main.cpp
  DoubleMultiVariableFunction.cpp
  DoubleMultiVariableFunction.h
  DoubleParameterizedFunctional.cpp
  DoubleParameterizedFunctional.h
  LevenbergMarquardtMinimizer.cpp
  LevenbergMarquardtMinimizer.h
  PairwiseCostFunction.h
  PairwiseCostFunctionAndItsGradientWithRespectToParams.cu
  PairwiseCostFunctionAndItsGradientWithRespectToParams.h
  ProjectionOntoLineAndItsJacobian.cu
  ProjectionOntoLineAndItsJacobian.cuh
  ProjectionOntoLineAndItsJacobian.h
  TestLevenbergMarquardtMinimizer.cu
  TestLevenbergMarquardtMinimizer.h
  UnaryCostFunction.h
  UnaryCostFunctionAndItsGradientWithRespectToParams.cu
  UnaryCostFunctionAndItsGradientWithRespectToParams.h
  )

target_link_libraries(
  LevenbergMarquardtMinimizer
  ${GFLAGS_LIBRARIES}
  ${HDF5_LIBRARIES}
  )