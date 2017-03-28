cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(GenerateNeighborhoodGraph)

set(IO_INCLUDE_DIR "" CACHE PATH "Include directory for IO.")
set(IO_LIB "" CACHE FILEPATH "Library required to link IO.")

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS log program_options)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${IO_INCLUDE_DIR})

add_executable(
  GenerateNeighborhoodGraph
  main.cpp)

target_link_libraries(
  GenerateNeighborhoodGraph
  ${Boost_LIBRARIES}
  ${IO_LIB}
  ${ITK_LIBRARIES}) 
