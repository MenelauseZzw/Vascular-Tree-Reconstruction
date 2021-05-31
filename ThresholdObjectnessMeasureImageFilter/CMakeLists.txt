project(ThresholdObjectnessMeasureImageFilter)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  ThresholdObjectnessMeasureImageFilter
  main.cpp)

target_link_libraries(
  ThresholdObjectnessMeasureImageFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS ThresholdObjectnessMeasureImageFilter
  RUNTIME DESTINATION Bin)
