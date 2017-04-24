project(CropImageFilter)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  AdditiveGaussianNoiseImageFilter
  main.cpp)

target_link_libraries(
  AdditiveGaussianNoiseImageFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS AdditiveGaussianNoiseImageFilter 
  RUNTIME DESTINATION Bin)
