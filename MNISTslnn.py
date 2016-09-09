# GET THE DATA
import numpy as np
# mnist2numpy.py is an altered version of http://g.sweyla.com/blog/2012/mnist-numpy/ 
with open("mnist2numpy.py") as f:
    code = compile(f.read(), "mnist2numpy.py", 'exec')
    exec(code)
Xtrain, Ytrain = load_mnist(dataset="training", digits=np.arange(10), path=".")
Xtest, Ytest = load_mnist(dataset="testing", digits=np.arange(10), path=".")

# NORMALIZE THE DATA
Xtrain = Xtrain/255 # pixel values range between 0 and 255
Xtest = Xtest/255

# READ SOME PROPERTIES OF THE DATA
N = Xtrain.shape[0]
d = Xtrain.shape[1]
Ntest = Xtest.shape[0]

# SET THE PROPERTIES OF THE SINGLE-LAYER NETWORK
K = 10 # number of classes corresponding to 10 digits. Should not be changed.
L = 30 # number of hidden layers
Epochs = 10 # number of times to train
lrnRt = 0.04 # learning rate


# INITIALIZE THE WEIGHTS AND BIASES
W1 = (1/np.sqrt(d))*np.random.randn(L,d)
W2 = (1/np.sqrt(L))*np.random.randn(K,L)
b1 = np.zeros(shape = (L), dtype = float)
b2 = np.zeros(shape = (K), dtype = float)

# DEFINE THE HIDDEN AND OUTPUT UNITS
def softmax(v):
    tmp = np.exp(v-np.amax(v))
    return( tmp/np.sum(tmp) )
def sigmaPrimeReLU(c): # derivative of rectified linear unit
    return( (c > 0).astype(float) )
def ReLU(c): # Rectified linear unit
    return( np.maximum(c,0) )

# LEARN THE MODEL
AccAtEachEpoch = np.zeros(Epochs) # container for storing accuracies
for j in range(Epochs):
    reorder = np.random.permutation(N) # shuffle
    for i in range(N):
        p = reorder[i]
        # forward propagation
        Z1 = np.dot(W1,Xtrain[p,:]) + b1
        a2 = ReLU(Z1)
        Z2 = np.dot(W2,a2) + b2
        # backward propagation
        targetp = np.zeros((K))
        targetp[Ytrain[p]] += 1
        dLdb2 = targetp - softmax(Z2)
        dLdW2 = np.outer(dLdb2, a2)
        dLdb1 = np.multiply( sigmaPrimeReLU(Z1), np.dot(dLdb2, W2))
        dLdW1 = np.outer(dLdb1, Xtrain[p,:])
        # updating the weights
        W1 = W1 + lrnRt*dLdW1
        b1 = b1 + lrnRt*dLdb1
        W2 = W2 + lrnRt*dLdW2
        b2 = b2 + lrnRt*dLdb2
    # check predictions at the end of each epoch
    Ypredict = np.zeros((Ntest, 1), dtype=np.int8)
    for p in range(Ntest):
        Z1 = np.dot(W1,Xtest[p,:]) + b1
        a2 = ReLU(Z1)
        Z2 = np.dot(W2,a2) + b2
        Ypredict[p] = np.argmax( softmax(Z2) )
    AccAtEachEpoch[j] = np.mean(Ytest == Ypredict)
    print("Accuracy after epoch {} is: {}".format(j+1, AccAtEachEpoch[j]))                
