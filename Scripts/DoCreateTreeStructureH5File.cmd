@echo off

set voxelPhysicalSize=0.046

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateTreeStructureH5File" ^
--inputFileName ".\image001\tree_structure.xml" ^
--outputFileName ".\image001\tree_structure.h5" ^
--voxelPhysicalSize %voxelPhysicalSize%

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\tree_structure.h5" ^
--outputFileName ".\image001\tree_structure.vtp" ^
--pointsArrName "positions"
