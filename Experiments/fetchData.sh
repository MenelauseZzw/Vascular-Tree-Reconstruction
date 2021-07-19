#!/bin/sh

# download synthetic vessel volume data including both std10 and std15
if [ ! -f "synthetic_data.zip" ];
then
	wget https://cs.uwaterloo.ca/~z889zhan/synthetic_data.zip
	unzip synthetic_data.zip
        rm synthetic_data.zip
fi

# std15 volumes
for num in {001..015} ; do
        ln -s ./synthetic_data/std15/image$num image$num
done
