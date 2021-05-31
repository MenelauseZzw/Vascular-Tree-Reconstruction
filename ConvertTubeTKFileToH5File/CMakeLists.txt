project(ConvertTubeTKFileToH5File)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  ConvertTubeTKFileToH5File
  main.cpp)

target_link_libraries(
  ConvertTubeTKFileToH5File
  IO
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS ConvertTubeTKFileToH5File
  RUNTIME DESTINATION Bin)
