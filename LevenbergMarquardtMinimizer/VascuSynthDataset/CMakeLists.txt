cmake_minimum_required(VERSION 2.8.11)
project(VascuSynthDataset)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Find Google gflags library.
set(GFLAGS_NAMESPACE gflags)
find_package(Gflags REQUIRED)
              
# Find Google glog library.
find_package(Glog REQUIRED)

# Find Itk.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Find Vtk.
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Find Hdf5.
# find_package(HDF5 REQUIRED NAMES hdf5 COMPONENTS CXX)

include_directories(
  ${GFLAGS_INCLUDE_DIRS} 
  ${GLOG_INCLUDE_DIRS}
#  ${HDF5_INCLUDE_DIR}
)

add_executable(
  VascuSynthDataset
  main.cpp
  pugiconfig.hpp
  pugixml.cpp
  pugixml.hpp
)

target_link_libraries(
  VascuSynthDataset
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
#  ${HDF5_LIBRARIES}
  ${ITK_LIBRARIES}
  ${VTK_LIBRARIES}
)
