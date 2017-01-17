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
