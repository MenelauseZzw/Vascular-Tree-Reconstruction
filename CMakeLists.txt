#==========================================================================
#                         Vascular Tree Estimation                        #
#==========================================================================
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(VascularTreeEstimation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")

# set path to boost and HDF5 directory
set(HDF5_DIR "/home/z889zhan/VascularTreeEstimation/VTE-dependencies/CMake-hdf5-1.10.1/HDF5-1.10.1-Linux/HDF_Group/HDF5/1.10.1/share/cmake")
set(BOOST_ROOT "/home/z889zhan/VascularTreeEstimation/VTE-dependencies/boost_1_65_0/install")

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/Lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS graph log log_setup program_options system date_time filesystem thread)

find_package(OpenMP)

if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(GenerateTreeTopology)
add_subdirectory(GenerateNeighborhoodGraph)
add_subdirectory(IO)
add_subdirectory(NonMaximumSuppressionFilter)
add_subdirectory(ObjectnessMeasureImageFilter)
add_subdirectory(LevenbergMarquardtMinimizer)



