#==========================================================================
# Frechet Distance
#==========================================================================
project(FrechetDistance)

include_directories(
  ${Boost_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS} 
  ${FLANN_INCLUDE_DIRS})

add_executable(
  FrechetDistance
  main.cpp)

target_link_libraries(
  FrechetDistance
  IO
  ${Boost_LIBRARIES}
  ${FLANN_LIBRARIES})

install(
  TARGETS FrechetDistance
  RUNTIME DESTINATION Bin)