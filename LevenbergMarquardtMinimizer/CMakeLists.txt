cmake_minimum_required(VERSION 2.8.11)
project(LevenbergMarquardtMinimizer)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Find Google gflags library.
set(GFLAGS_NAMESPACE gflags)
find_package(gflags REQUIRED)
              
# Find Google glog library.
find_package(glog REQUIRED)

find_package(flann REQUIRED)
                         
find_package(CUDA REQUIRED)

set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS CXX)

include_directories(
  ${GFLAGS_INCLUDE_DIRS} 
  ${GLOG_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIR} 
  ${FLANN_INCLUDE_DIRS}
  "$ENV{NVCUDASAMPLES_ROOT}/common/inc"
  )

cuda_add_executable(
  LevenbergMarquardtMinimizer
  main.cpp
  AdjustLineEndpoints.cu
  AdjustLineEndpoints.h
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
  TestKnnSearch.cpp
  TestKnnSearch.h
  TestLevenbergMarquardtMinimizer1.cu
  TestLevenbergMarquardtMinimizer1.h
  TestLsqr.cu
  TestLsqr.h
  UnaryCostFunction.h
  UnaryCostFunctionAndItsGradientWithRespectToParams.cu
  UnaryCostFunctionAndItsGradientWithRespectToParams.h
  )

target_link_libraries(
  LevenbergMarquardtMinimizer
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${FLANN_LIBRARIES}
  )