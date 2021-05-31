project(DistanceMapImageFilter)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  DistanceMapImageFilter
  main.cpp)

target_link_libraries(
  DistanceMapImageFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS DistanceMapImageFilter
  RUNTIME DESTINATION Bin)
