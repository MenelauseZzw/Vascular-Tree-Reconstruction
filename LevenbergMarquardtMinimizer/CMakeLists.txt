cmake_minimum_required(VERSION 2.8.11)
project(LevenbergMarquardtMinimizer)

set(HDF5_USE_STATIC_LIBRARIES ON)
                                        
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS CXX)

include_directories(
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
  ProjectionGradient.cu
  UnaryCostFunction.h
  )

target_link_libraries(
  LevenbergMarquardtMinimizer
  ${HDF5_LIBRARIES}
  )