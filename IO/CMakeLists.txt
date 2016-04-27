cmake_minimum_required(VERSION 2.8.11)
project(IO)

find_package(HDF5 NAMES hdf5 COMPONENTS CXX static)
include_directories(${HDF5_INCLUDE_DIR})

add_definitions("-DFileReaderClassDeclSpec=__declspec(dllexport)")
add_definitions("-DFileWriterClassDeclSpec=__declspec(dllexport)")

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
