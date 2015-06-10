cmake_minimum_required(VERSION 2.8.11)
project(LevenbergMarquardtMinimizer)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

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

cuda_add_executable(
  LevenbergMarquardtMinimizer
  main.cpp
  CommandLineArgs.cpp
  CommandLineArgs.h
  DeviceFunctions.hpp
  DualNumber.hpp
  GraphData.hpp
  GraphDataReader.cpp
  GraphDataReader.hpp
  GraphDataWriter.cpp
  GraphDataWriter.hpp
  LevenbergMarquardtMinimizer.cpp
  LevenbergMarquardtMinimizer.h
  PairwiseCostFunction.hpp
  PairwiseCostGradientWithRespectToParams.hpp
  ProjectionOntoLine.hpp
  SparseLeastSquares.h
  TestKnnSearch1.cpp
  TestKnnSearch1.h
  TestLevenbergMarquardtMinimizer1.cu
  TestLevenbergMarquardtMinimizer1.h
  UnaryCostFunction.hpp
  UnaryCostGradientWithRespectToParams.hpp
  )

target_link_libraries(
  LevenbergMarquardtMinimizer
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${FLANN_LIBRARIES}
  )
