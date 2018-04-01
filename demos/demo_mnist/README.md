# Gaussian Processes demo

This demo will plot a gaussian process with a user-specified kernel.

## Setup 

Be sure to run the [setup](https://github.com/phineasng/MLearn/tree/master/demos) steps.
Compile specifically this demo from the build directory with `make FCNetMNIST`

## How to run the demo

This demo can be run from the correct subfolder with the command
`./FCNetMNIST`. By using the `--help` option, it is possible to visualize a list of available commands:
```bash
./FCNetMNIST --help
This is a demo showing a simple FC neural net trained on MNIST.:
  --help                Show the help
  --data_folder arg     Folder where to find the uncompressed data.
  -v [ --visualize ]    Visualize test samples with classification.
```
Note that the `--data_folder` option is mandatory. This is used to indicate the folder where the 
MNIST dataset has been stored. You can obtain the dataset [here](http://yann.lecun.com/exdb/mnist/).
All 4 files should be downloaded and UNZIPPED in the same folder. 
![](https://github.com/phineasng/MLearn/blob/master/demos/demo_mnist/img/demo_mnist.png)
