# Where is everything?
 * here
 * copy is in tars:c:\echesakov\source\VascularTree...

    binary is in tars:c:\echesakov\Build

    installed in tars:c:\echesakov\VascularTreeEstimationRelease

# How to run

 1. Do ObjectenessMasureImageFilter.cmd
 
    → ObjectnessMeasureVolume --- data
 
 2. DoNonmaximumSuppressionVolume.cmd
 
    → NonMaximumSuppressionVolume.h5 --- graph
 
    Important `--thresholdValue` --- this is threshold for plots
 
 3. DoLevenbergMarquardtMinimizer.cmd
    → NonMaximumSuppressionCurvVolume.h5
 
    Important `--lambda` --- curvature coefficient
    
    `--voxelPixelSize` --- defines allowed error in ||l_p - p||_+
 
    DoLevenbergMarquardtMinimizerP.cmd --- same but on TensorFlow
 
 4. DoGenerateMinimumSpanningTree.cmd
 
    → NonMaximumSuppressionCurvVolumeEMST.h5 --- tree
 
    Note: to use different graph set `--knn X` where
    * `X=-1` uses complete graph
    * `X>0` uses KNN graph with `X` neighbours
 
    `--optionNum Y`: 
    * `Y=1` for Eucledian weights,
    * `Y=2` for sum of arcs' lengths,
    * `Y=3` for min of arcs' lengths.
 
 5. DoComputeOverlapMeasure.cmd
 
    → _to console_
    
    computes different metrics
    
6. DoAnalyzeOurOverlapMeasure

    averages over volumes
 
