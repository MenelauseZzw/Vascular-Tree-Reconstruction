cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(GenerateNeighborhoodGraph)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS graph log log_setup program_options system date_time filesystem thread)

set(ITK_DIR "/home/zzhan896/InsightToolkit-4.13.0/build")
find_package(ITK REQUIRED)
find_package(Eigen REQUIRED)
find_package(Flann REQUIRED)

include(${ITK_USE_FILE})
include_directories(
${Boost_INCLUDE_DIRS}
${EIGEN_INCLUDE_DIRS}
${FLANN_INCLUDE_DIRS}
/home/zzhan896/VTE/IO)

add_executable(
  GenerateNeighborhoodGraph
  main.cpp)

target_link_libraries(
  GenerateNeighborhoodGraph 
  IO 
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES}
  ${FLANN_LIBRARIES}) 

install(
  TARGETS GenerateNeighborhoodGraph
  RUNTIME DESTINATION Bin)
