project(ObjectnessMeasureImageFilter)
 
include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  ObjectnessMeasureImageFilter
  main.cpp)
 
target_link_libraries(
  ObjectnessMeasureImageFilter
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS ObjectnessMeasureImageFilter
  RUNTIME DESTINATION Bin)