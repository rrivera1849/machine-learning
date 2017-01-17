# [CS-231n Neural Networks Part 2 Summary](http://cs231n.github.io/neural-networks-2/)

Topics Covered:
- Data Preprocessing
- Weight Initialization
- Loss Functions

## Data Preprocessing

### Mean Subtraction
Just subtract the mean accross every **individual feature** in the data. Just centers the 
data around the origin.

Numpy:
`X -= np.mean (X, axis=0)`

### Normalization
Just normalizes the dimensions of the data so that they are roughly the same scale.

Two common ways:
- Do mean subtraction then divide by standard deviation `X /= X.std (X, axis=0)`
- Normalize to interval [-1,1] Only apply this if you have a reason to believe that different input 
  features have different scales but should be of approximately equal importance for the learning algorithm

### PCA
The goal of PCA is to reduce the dimensionality of the data to include only those values which contain 
the highest variance.

To achieve this we first mean center the data and calculate the covariance matrix
`X -= np.mean (X, axis=0)`
`cov = np.dot (X.T, X) / X.shape[0]`

We then calculate the SVD factorization of the covariance matrix
`U,S,V = np.linalg.svd (cov)`
Here, U is are the eigenvectors sorted by the eigenvalues and S is a 1-D array 
of the singular values.

We can now reduce the dimensionality and de-correlate the data
`Xrot_reduced = np.dot (X, U[:,:100])`

We now have the 100 dimensions with the most variance, this can be viewed as those that contain 
the most information.

### Whitening
This operation can be applied right after PCA, it has the effect of normalizing the scale
of the PCA decorrelated dimensions.
`Xwhite = Xrot_reduced / np.sqrt (S[:100] + 1e-5)`
Here, the term 1e-5 is just noise.

### Very Important Note
This applies for any preprocessing statistic (e.g. the data mean).
The mean should only be calculated on the training data set and then subtracted off the
train/validation/test set.

## Weight Initialization

### Pitfall: All Zero Initialization
If the network computes all the outputs to be the same then during backpropagation it will 
compute the same gradient update for every parameter. This can be extended to "All Same Initialization".

### Small Random Numbers
Initialize the weights to small numbers from a Gaussian or Uniform distribution (doesn't make much difference).
`W = 0.01 * np.randn (D, H)`

Warning:
Be careful with small numbers since the gradient updates are proportional to the weights these control de 
gradient signal strength flowing backwards. This might be a concern for deeper networks.

### Calibrating Variances with 1 / SQRT (n)
A problem with the above is that the variance of the outputs grows with the number of the inputs.
We can normalize the variance of each neurons output to 1 by scaling its weight vector by the 
square root of fann-in (number of inputs).
`w = np.random.randn (n) / sqrt (n)`
Where n is the number of inputs, this means each output has approximately the same distribution 
which **improves the rate of convergence**.

### Spare Initialization
Initialize all the weights to zero but break the symmetry by allowing every neuron to be **randomly connected** 
with weights from a small gaussian to a fixed number of neurons below it. 

TODO: Dropout Layer?

### Biases
They can all be zero since symmetry breaking is done by weights.

### Batch Normalization
Discussed in BN paper in directory, it is like doing preprocessing at every layer but 
fully integratd into the network!
