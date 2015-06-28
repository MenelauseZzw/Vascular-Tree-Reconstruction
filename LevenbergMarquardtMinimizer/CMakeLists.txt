#==========================================================================
#
#
#==========================================================================
cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)


project(LevenbergMarquardtMinimizer)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")

# Find Google gflags library.
set(GFLAGS_NAMESPACE gflags)
find_package(Gflags REQUIRED)
              
# Find Google glog library.
find_package(Glog REQUIRED)

# Find FLANN library.
find_package(Flann REQUIRED)
                         
find_package(CUDA REQUIRED)

set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS CXX)

include_directories(
  ${GFLAGS_INCLUDE_DIRS} 
  ${GLOG_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIR} 
  ${FLANN_INCLUDE_DIRS}
  )

source_group("CUDA Source Files" REGULAR_EXPRESSION "cu$")
source_group("CUDA Header Files" REGULAR_EXPRESSION "cuh$")

#------------------------------------------------------------------------------

cuda_add_executable(
  LevenbergMarquardtMinimizer
  main.cpp
  CommandLineArgs.cpp
  CommandLineArgs.h
  CostFunction.hpp
  CurvatureCostFunction.cuh
  CurvatureCostFunction.hpp
  CurvatureCostFunction.inl
  DistanceCostFunction.cuh
  DistanceCostFunction.hpp
  DistanceCostFunction.inl
  DotProduct.cuh
  DualNumber.cuh
  GraphData.hpp
  GraphDataReader.cpp
  GraphDataReader.hpp
  GraphDataWriter.cpp
  GraphDataWriter.hpp
  LevenbergMarquardtMinimizer.hpp
  LevenbergMarquardtMinimizer.inl
  LinearCombination.hpp
  LinearCombination.inl
  LSQR.hpp
  Minimize.cu
  Minimize.h
  ProjectionOntoLine.cuh
  ProjectionOntoLine.hpp
  ProjectionOntoLine.inl
  TestKnnSearch1.cpp
  TestKnnSearch1.h
  )

target_link_libraries(
  LevenbergMarquardtMinimizer
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${FLANN_LIBRARIES}
  )
