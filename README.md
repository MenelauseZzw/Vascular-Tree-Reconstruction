
# Vascular-Tree-Reconstruction (continue to be updated)

## How to use 

 1. Do ObjectenessMasureImageFilter
 
    → ObjectnessMeasureVolume --- data
 
 2. DoNonmaximumSuppressionVolume
 
    → NonMaximumSuppressionVolume.h5 --- graph
 
    Important `--thresholdValue` --- this is threshold for plots
 
 3. DoLevenbergMarquardtMinimizer
    → NonMaximumSuppressionCurvVolume.h5
 
    Important `--lambda` --- curvature coefficient
	
	`--beta` --- divergence coefficient
    
    `--voxelPhysicalSize` --- defines allowed error in ||l_p - p||_+
 
 4. DoGenerateMinimumSpanningTree
 
    → NonMaximumSuppressionCurvVolumeEMST.h5 --- tree
 
    Note: to use different graph set `--knn X` where
    * `X=-1` uses complete graph
    * `X>0` uses KNN graph with `X` neighbours
 
    `--optionNum Y`: 
    * `Y=1` for Eucledian weights,
    * `Y=2` for sum of arcs' lengths,
    * `Y=3` for min of arcs' lengths.