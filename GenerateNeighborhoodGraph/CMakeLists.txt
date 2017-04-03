project(GenerateNeighborhoodGraph)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  GenerateNeighborhoodGraph
  main.cpp)

target_link_libraries(
  GenerateNeighborhoodGraph 
  IO 
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES}) 

install(
  TARGETS GenerateNeighborhoodGraph
  RUNTIME DESTINATION Bin)
