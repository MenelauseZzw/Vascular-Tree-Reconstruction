cmake_minimum_required(VERSION 2.8)

project(CropImageFilter)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(CropImageFilter main.cpp)
 
target_link_libraries(CropImageFilter ${ITK_LIBRARIES} ${Boost_LIBRARIES})