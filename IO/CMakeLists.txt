cmake_minimum_required(VERSION 2.8.11)
project(IO)

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

find_package(HDF5 NAMES hdf5 COMPONENTS CXX static)
include_directories(${HDF5_INCLUDE_DIR})

if (WIN32)
  add_definitions("-DFileReaderClassDeclSpec=__declspec(dllexport)")
  add_definitions("-DFileWriterClassDeclSpec=__declspec(dllexport)")
endif()

add_library(
  ${PROJECT_NAME} SHARED
  FileReader.cpp
  FileReader.hpp
  FileReaderImpl.hpp
  FileWriter.cpp
  FileWriter.hpp
  FileWriterImpl.hpp
  H5Helper.hpp)

target_link_libraries(
  ${PROJECT_NAME}
  ${HDF5_CXX_STATIC_LIBRARY})

install(
  TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION Bin
  ARCHIVE DESTINATION Lib
  LIBRARY DESTINATION Lib)

install(
  FILES FileReader.hpp FileWriter.hpp DESTINATION Include)
