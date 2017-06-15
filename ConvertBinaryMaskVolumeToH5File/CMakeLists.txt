project(ConvertBinaryMaskVolumeToH5File)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(
  ConvertBinaryMaskVolumeToH5File
  main.cpp)

target_link_libraries(
  ConvertBinaryMaskVolumeToH5File
  IO
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES})

install(
  TARGETS ConvertBinaryMaskVolumeToH5File
  RUNTIME DESTINATION Bin)
