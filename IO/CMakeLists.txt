project(IO)

include_directories(${HDF5_INCLUDE_DIR})

if (WIN32)
  add_definitions("-DFileReaderClassDeclSpec=__declspec(dllexport)")
  add_definitions("-DFileWriterClassDeclSpec=__declspec(dllexport)")
endif()

add_library(
  IO SHARED
  FileReader.cpp
  FileReader.hpp
  FileReaderImpl.hpp
  FileWriter.cpp
  FileWriter.hpp
  FileWriterImpl.hpp
  H5Helper.hpp)

target_include_directories(
  IO PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
  $<INSTALL_INTERFACE:Include>)

target_link_libraries(
  IO PRIVATE 
  ${HDF5_CXX_STATIC_LIBRARY})

install(
  TARGETS IO
  RUNTIME DESTINATION Bin
  ARCHIVE DESTINATION Lib
  LIBRARY DESTINATION Lib)

install(
  FILES FileReader.hpp FileWriter.hpp DESTINATION Include)