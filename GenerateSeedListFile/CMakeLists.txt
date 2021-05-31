project(GenerateSeedListFile)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  GenerateSeedListFile
  main.cpp)

target_link_libraries(
  GenerateSeedListFile
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS GenerateSeedListFile
  RUNTIME DESTINATION Bin)
