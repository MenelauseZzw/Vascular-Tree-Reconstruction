#==========================================================================
# Frechet Distance
#==========================================================================
cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)

project(FrechetDistance)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(IO_INCLUDE_DIR "" CACHE PATH "Include directory for IO.")
set(IO_LIB "" CACHE FILEPATH "Library required to link IO.")

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

# Find Boost library.
find_package(Boost REQUIRED COMPONENTS graph)

# Find Google gflags library.
find_package(Gflags REQUIRED)

# Find Google glog library.
find_package(Glog REQUIRED)

# Find Eigen library.
find_package(Eigen3 REQUIRED)

# Find Flann library.
find_package(Flann REQUIRED)

# Find OpenMP (optional).
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${IO_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${FLANN_INCLUDE_DIRS})

add_executable(
  ${PROJECT_NAME}
  main.cpp)

target_link_libraries(
  ${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${IO_LIB}
  ${FLANN_LIBRARIES})
