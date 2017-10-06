@echo off

"C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe" "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\tree_structure.h5" ^
--outputFileName ".\image001\tree_structure.vtp" ^
--pointsArrName "positions"

"C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe" "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\ObjectnessMeasureVolume.h5" ^
--outputFileName ".\image001\ObjectnessMeasureVolume.vtp" ^
--pointsArrName "measurements"

"C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe" "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\NonMaximumSuppressionVolume.vtp" ^
--pointsArrName "measurements"

"C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe" "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\1.95\NonMaximumSuppressionCurvVolume.h5" ^
--outputFileName ".\image001\1.95\NonMaximumSuppressionCurvVolume.vtp" ^
--pointsArrName "positions"
