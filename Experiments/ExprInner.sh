#!/bin/bash

num=$1
thresholdBelow=$2
knn=4
dirname="./image$num"

newdir="./image$num/$thresholdBelow"
[ ! -d $newdir ] && mkdir -p $newdir

/home/z889zhan/VascularTreeEstimation/VTE-release/Bin/NonMaximumSuppressionFilter \
--inputFileName "$dirname/ObjectnessMeasureVolume.mhd" \
--outputFileName "$newdir/NonMaximumSuppressionVolume.mhd" \
--thresholdBelow $thresholdBelow

/home/z889zhan/VascularTreeEstimation/VTE-release/Bin/GenerateNeighborhoodGraph \
--thresholdBelow $thresholdBelow \
--aKnnGraph true \
--knnValue $knn \
--aspectRatio 10 \
--mutualLink false \
--distConstraint 2 \
--voxelSize 0.046 \
--inputFileName "$newdir/NonMaximumSuppressionVolume.mhd" \
--outputFileName "$newdir/NonMaximumSuppressionVolumeMalK4.h5"

gpuDevice=1
lambda=0.65
voxelPhysicalSize=0.00001
subdir="./image$num/$thresholdBelow/$lambda"
beta=4.25
tau=0.42
if [ ! -d $subdir ]; then
mkdir $subdir
fi
/home/z889zhan/VascularTreeEstimation/VTE-release/Bin/LevenbergMarquardtMinimizer \
--gpuDevice $gpuDevice \
--lambda $lambda \
--voxelPhysicalSize $voxelPhysicalSize \
--beta $beta \
--tau $tau \
--inputFileName "$newdir/NonMaximumSuppressionVolumeMalK4.h5" \
--outputFileName "$subdir/NonMaximumSuppressionVolumeMalK4OriAbsCurvDivVolume.h5"

# If using the minimum arborescence, root (index) has to be specified
# otherwise, no need for root
/home/z889zhan/VascularTreeEstimation/VTE-release/Bin/GenerateTreeTopology \
--directedLabel true \
--root 0 \
--knn 100 \
--optionNum 2 \
--inputFileName "$subdir/NonMaximumSuppressionVolumeMalK4OriAbsCurvDivVolumeWithRoot.h5" \
--outputFileName "$subdir/NonMaximumSuppressionVolumeMalK4OriAbsCurvDivVolumeWithRootMArb.h5"
