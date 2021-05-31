project(CropImageFilter)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  CropImageFilter
  main.cpp)

target_link_libraries(
  CropImageFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS CropImageFilter 
  RUNTIME DESTINATION Bin)
