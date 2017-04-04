project(GenerateMinimumSpanningTree)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
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