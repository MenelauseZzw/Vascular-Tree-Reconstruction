cmake_minimum_required(VERSION 2.8)

project(ObjectnessMeasureImageFilter)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
 
find_package(Boost REQUIRED COMPONENTS log program_options)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(ObjectnessMeasureImageFilter main.cpp)
 
target_link_libraries(ObjectnessMeasureImageFilter ${ITK_LIBRARIES} ${Boost_LIBRARIES})