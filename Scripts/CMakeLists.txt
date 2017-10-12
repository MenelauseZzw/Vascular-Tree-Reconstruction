file(GLOB InFiles "*.cmd.in")

find_package(PythonInterp 2.7)

if (NOT PythonInterp_FOUND)
  set(PYTHON_EXECUTABLE "C:/Python27python.exe/")
endif(NOT PythonInterp_FOUND)

foreach(InFile ${InFiles})
  get_filename_component(FileNameWE "${InFile}" NAME_WE)
  set(OutFile 
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${FileNameWE}.cmd")

  configure_file("${InFile}" "${OutFile}" @ONLY)
  install(FILES "${OutFile}" DESTINATION Scripts)
endforeach(InFile)
