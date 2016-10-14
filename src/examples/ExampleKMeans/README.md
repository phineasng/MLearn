# KMeans Clustering example

This is just an example which can be used to test the kmeans clustering
implementation.
In the main.cpp file:
We generate a chosen number of points (first input to the program) in a circle using
a uniform distribution for the angle and the center. This means there is
a higher point density towards the center of the circle.
The second input is the desired number of clusters K.

By default, the kmeans clustering algorithm is run 10 times with a kmeans ++
initialization, but the initialization can be changed to random or to
user-defined cluster centers.
The default distance function is the squared euclidian distance, but another
distance function can be implemented in MLearn/Clustering/DistanceFunction.h.

The python file ShowClusters.py uses the output saved in a .csv file by the cpp
executable for visualization.
The following image shows 1 million points clustered into 14
clusters, with a kmeans++ initialization.


![](https://github.com/phineasng/MLearn/blob/master/src/examples/ExampleKMeans/img/14.png)

