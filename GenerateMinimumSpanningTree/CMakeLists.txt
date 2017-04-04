project(GenerateMinimumSpanningTree)

find_package(Eigen REQUIRED)
find_package(Flann REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${FLANN_INCLUDE_DIRS})

add_executable(
  GenerateMinimumSpanningTree
  main.cpp)

target_link_libraries(
  GenerateMinimumSpanningTree
  IO
  ${Boost_LIBRARIES}
  ${FLANN_LIBRARIES})

install(
  TARGETS GenerateMinimumSpanningTree
  RUNTIME DESTINATION Bin)
