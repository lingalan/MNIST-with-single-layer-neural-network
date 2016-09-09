# MNIST-with-single-layer-neural-network
Implements a single-layer neural-network on MNIST data

[MNIST](https://en.wikipedia.org/wiki/MNIST_database) database of handwritten digits can be downloaded from [this link](http://yann.lecun.com/exdb/mnist/). The training data consists of 60000 images of size 24x24 pixels with lables being the 10 digits. Goal is to predict the labels for the 10000 images of test data.

Here we implement a single layer neural network on this data. The theory and accuracy obtained is discussed in `MNISTslnnExplanation.pdf`

The code   `MNISTslnn.py`  should be executed after unzipping the data files obtained from above link.

Parameters that can be changed in the code are:  
`L` is the number of neurons in the hidden layer.  
`lrnRt` is the learning rate.  
`Epochs` is the number of epochs for which training should be performed.

