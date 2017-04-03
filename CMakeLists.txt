#==========================================================================
#                         Vascular Tree Estimation                        #
#==========================================================================
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(VascularTreeEstimation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
 
find_package(Boost REQUIRED COMPONENTS log program_options)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Flann REQUIRED)
find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS CXX static)
find_package(ITK REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(CropImageFilter)
add_subdirectory(FrechetDistance)
add_subdirectory(GenerateNeighborhoodGraph)
add_subdirectory(IO)
add_subdirectory(LevenbergMarquardtMinimizer)
add_subdirectory(NonMaximumSuppressionFilter)
add_subdirectory(ObjectnessMeasureImageFilter)
