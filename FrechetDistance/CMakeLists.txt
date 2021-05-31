#==========================================================================
# Frechet Distance
#==========================================================================
project(FrechetDistance)

find_package(Eigen REQUIRED)
find_package(Flann REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIRS} 
  ${EIGEN_INCLUDE_DIRS} 
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
