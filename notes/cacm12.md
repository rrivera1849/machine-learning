# A Few Useful Things to Know about Machine Learning

## Notes

### Learning = Representation + Evaluation + Optimization
Representation
- Every algorithm in capable of representing classifiers in different ways, this yields
  in different hypothesis spaces
Evaluation
- This is the objective or scoring function, what we use to make sure our algorithm is doing
  well
Optimization
- This is our strategy for searching among the classifiers for the highest scoring one

### Generalization is all that counts
Keep your test-set seperate and pristine. Do not touch it for anything else but evaluating the 
test error.

### Data Alone is not Enough
A learner can not possibly do better than random on all the examples it has not seen 
without making some assumptions. Learning is an inductive process, given some knowledge and 
assumptions, it creates classifiers that generalize well. Use algorithms whose assumptions fit 
the data that you have.

### Overfitting has many faces
Bias - How consistently it learns the wrong thing
Variance - Learns random things irrespective of the real signal
Strong false assumptions can be better than weak true assumptions in a learner.

### Intuition fails in high dimensions
As the numbers of features increase, the dimensionality of our input space increases as well. 
Simply, the more features ratio of training-data vs input-space decreases.

The higher the dimensionality, the more alike our examples look. If we have irrelevant features 
this could add to the noise of our model.

### Theoretical gurantees are not what they seem
A theoretical gurantee does not mean that our algorithm will work. Take all of these 
gurantees with a grain of salt.

### Feature engineering is the key
Which features you use is one of the most important parts of machine learning. Unfortunately, this is both 
mostly an art and domain specific.

### More data beats a cleverer algorithm
Many learners can come up with the same result. Try simpler algorithms before moving on to more 
complicated ones.

A learners representation may grow with the training data (Decision Trees) or it may be fixed (Linear).
Thus more data fed to a former will tend to do better in the end. 

### Learn many models, not just one
Consider using ensembles, they increase the hypothesis space.

### Representable does not imply learnable
Just because some model, say, Neural Networks can represent any non-linear function it does not mean 
that you will have the time to find the correct function thus you wont learn it.

