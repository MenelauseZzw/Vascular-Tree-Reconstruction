project(NonMaximumSuppressionFilter)
 
include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  NonMaximumSuppressionFilter
  main.cpp)
 
target_link_libraries(
  NonMaximumSuppressionFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS NonMaximumSuppressionFilter
  RUNTIME DESTINATION Bin)