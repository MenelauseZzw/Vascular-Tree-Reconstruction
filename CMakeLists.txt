#==========================================================================
#                         Vascular Tree Estimation                        #
#==========================================================================
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(VascularTreeEstimation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/Lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS graph log log_setup program_options)

find_package(OpenMP)

if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(AdditiveGaussianNoiseImageFilter)
add_subdirectory(BinaryThresholdImageFilter)
add_subdirectory(ConvertBinaryMaskVolumeToH5File)
add_subdirectory(ConvertTubeTKFileToH5File)
add_subdirectory(CropImageFilter)
add_subdirectory(DistanceMapImageFilter)
add_subdirectory(FrechetDistance)
add_subdirectory(GenerateMinimumSpanningTree)
add_subdirectory(GenerateNeighborhoodGraph)
add_subdirectory(GenerateSeedListFile)
add_subdirectory(IO)
add_subdirectory(LevenbergMarquardtMinimizer)
add_subdirectory(NonMaximumSuppressionFilter)
add_subdirectory(ObjectnessMeasureImageFilter)
add_subdirectory(ThresholdObjectnessMeasureImageFilter)
