@echo off

"C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe" "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateRadiusesPolyDataFile" ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\NonMaximumSuppressionVolumeRadiuses.vtp" ^
--pointsArrName "measurements"

"C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe" "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateRadiusesPolyDataFile" ^
--inputFileName ".\image001\1.95\NonMaximumSuppressionCurvVolume.h5" ^
--outputFileName ".\image001\1.95\NonMaximumSuppressionCurvVolumeRadiuses.vtp" ^
--pointsArrName "positions"
