
# Vascular-Tree-Reconstruction

## How to install
The following installation is tested using cmake-3.10.2 under Ubuntu 18.04.3 LTS.
1. c/c++ compiler: gcc-4.8/g++-4.8
2. CUDA: 8.0
3. Dependencies: boost-1.65.0, hdf5-1.10.1, eigen-3.3.4, ITK-4.13.3, flann-1.8.4, cusp-0.5.1
4. If these dependencies are installed in a custom folder, you may need to help cmake find them by explicitly setting some variables in [CMakeLists.txt](CMakeLists.txt), [CMake/FindEigen.cmake](CMake/FindEigen.cmake), [CMake/FindFlann.cmake](CMake/FindFlann.cmake) and [CMake/FindITK.cmake](CMake/FindITK.cmake).
5. Create a new folder *Build* inside the folder where this repository is cloned.
```
mkdir Build
cd Build
```
6. To build the project
```
cmake -DCMAKE_INSTALL_PREFIX=[path to install the project] ..
make -j 8
make install
```

## How to use 
The example script is contained in the [Experiments](Experiments). To obtain the synthetic vessel volume data, use [fetchData.sh](Experiments/fetchData.sh). After downloading the data, simply run [ExprOutter.sh](Experiments/ExprOutter.sh).

 1. Do ObjectenessMasureImageFilter (Frangi filtering)
 
 2. Do NonmaximumSuppressionVolume
  
    Important 
    ```
    --thresholdValue --- this is threshold for plots (ROC, angle error)
    ```
	
 3. Do GenerateNeighborhoodGraph
  
 4. Do LevenbergMarquardtMinimizer (use GPU)
  
    Important 
    ```
    --lambda  --- curvature coefficient
    --beta  --- divergence coefficient
    --tau  --- the hyperparameter in the oriented curvature
    --voxelPhysicalSize  --- defines allowed error in ||l_p - p||_+
    ```
 
 5. Do GenerateTreeTopology
  
    Important 
    ```
    --directedLabel   --- a flag used to decide whether to use minimum arborescence or MST
    --root   --- root index needs to be specified if using the minimum arborescence
    ```
	
	Code for the minimum arborescence: https://github.com/atofigh/edmonds-alg.git
