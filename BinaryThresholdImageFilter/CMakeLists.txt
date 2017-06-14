project(BinaryThresholdImageFilter)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  BinaryThresholdImageFilter
  main.cpp)

target_link_libraries(
  BinaryThresholdImageFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS BinaryThresholdImageFilter
  RUNTIME DESTINATION Bin)
