cmake_minimum_required(VERSION 2.8)

project(NonMaximumSuppressionFilter)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set(Boost_USE_STATIC_LIBS    ON) # only find static libs
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
 
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(NonMaximumSuppressionFilter main.cpp)
 
target_link_libraries(NonMaximumSuppressionFilter ${ITK_LIBRARIES} ${Boost_LIBRARIES})