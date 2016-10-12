# Display resulting clusters saved in csv file

import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D

# read out data and store in 2 numpy arrays
def readData(filename):
	info = []
	data = []
        data.append([])
        data.append([])
        label = []
        centroids = []
        centroids.append([])
        centroids.append([])
	file_to_read = open(filename, "r")
	lines = file_to_read.readlines()
        for i in lines[0].strip().split(","):
            info.append(int(i))
        for i in lines[1].strip().split(","):
            data[0].append(float(i))
        for i in lines[2].strip().split(","):
            data[1].append(float(i))
        for i in lines[3].strip().split(","):
            label.append(int(i))
        for i in lines[4].strip().split(","):
            centroids[0].append(float(i))
        for i in lines[5].strip().split(","):
            centroids[1].append(float(i))
	file_to_read.close()
	X = np.array(data)
        l = np.array(label)
        c = np.array(centroids)
	return X, l, c	


def run(filename):
    X, l, c = readData(filename)
    fig = plt.figure(1)
    plt.grid(True)
    plt.scatter(X[0,:],X[1,:],c=l.astype(np.float),s = 5,lw = 0)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    axes = plt.gca()
    axes.set_xlim([0,100])
    axes.set_ylim([0,100])
    fig.savefig("KMeans example.png", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
	run("clustering.csv")

