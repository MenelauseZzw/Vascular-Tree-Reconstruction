project(GenerateTreeTopology)

find_package(Eigen REQUIRED)
find_package(Flann REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${FLANN_INCLUDE_DIRS})

add_executable(
  GenerateTreeTopology
  main.cpp)

target_link_libraries(
  GenerateTreeTopology
  IO
  ${Boost_LIBRARIES}
  ${FLANN_LIBRARIES})

install(
  TARGETS GenerateTreeTopology
  RUNTIME DESTINATION Bin)
