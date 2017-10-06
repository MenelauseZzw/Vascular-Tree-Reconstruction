@echo off

set lambdaValue=1.95
set maxDistance=0.092

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoGenerateEuclideanMinimumSpanningTree" ^
--inputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolume.h5" ^
--outputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolumeEMST.h5" ^
--pointsArrName "positions" ^
--maxDistance %maxDistance%

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--pointsArrName "positions" ^
--inputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolumeEMST.h5" ^
--outputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolumeEMST.vtp"
