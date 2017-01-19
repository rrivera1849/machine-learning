# [CS-231n Neural Networks Part 3 Summary](http://cs231n.github.io/neural-networks-3/)

This entire part is focused on the dynamics of the neural network, or in other words,
the process of learning the parameters and finding good hyperparameters.

## Gradient Checks
Comparison of the analytical gradient to the numeric gradient.

### Use the centered derivative formula:

df(x)/dx = (f (x+h) - f(x-h)) / h

for very small h e.g. 1e-5

### Use the relative error when comparing them:

abs (analytical - numerical) / max (abs (analytical), abs (numerical))

If both are zero then gradient check passes, don't divide by zero!
Guidelines:
- relative error > 1e-2 usually means the gradient is probably wrong
- 1e-2 > relative error > 1e-4 should make you feel uncomfortable
- 1e-4 > relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high.
- 1e-7 and less you should be happy.

For deeper networks the error will be larger.

### Use Double Precision
Use double precision, floats can be misleading.

### Stick around active range of floating point
Be careful with how small your loss values are, if they get too small you might want to 
scale them up by a constant and then compare & evaluate.

### Kinks in the objective
If there are kinks in the objective, as is the case with non-differentiable functions such as ReLU and 
the SVM loss, then evaluating very small values might be tricky. Consider some value such as 1e-6, it could be 
that when evaluating the numerical gradient this crosses over the boundary of 0 in the ReLU function thus the numerical 
gradient will not be exactly zero. This is much more common than one thinks.

### Use only few datapoints
Don't gradcheck all the data points which means it is both more efficient and you're able to do it for 
every batch.

### Be careful with the step size h
Smaller might not be better, look at:
https://en.wikipedia.org/wiki/Numerical_differentiation

### Gradcheck during a “characteristic” mode of operation
Because gradchecks are performed at a particular and usually random points in the space of parameters, this 
does not mean that the gradient is implemented correctly. 
Having random gradchecks does not gurantee the correctness of your implementation, you should let your network 
burn-in a little bit and then do random grad-checks to confirm the accuracy of your implementation.

### Don’t let the regularization overwhelm the data
Test your code with loss only first, then add regularization. You should see the conribution of your reg term as well.

### Remember to turn off dropout/augmentations
Turn off all non-determenistic things such as dropout before proceeding.

### Check only few dimensions
Only  a few of the gradient dimensions should be checked for really large networks.

## Before Learning Sanity Checks / Tips & Tricks
- Look for the correct loss during the first iteration
- Increasing regularization strength should increase the loss
- Overfit on a very tiny portion of your data with regularization lambda = 0

## Babysitting the learning process
Plot in terms of EPOCHS. Epochs are the amount of times that your full dataset has been observed, as opposed to iterations
which are a function of your batch size.

### Loss vs Epoch Plot
First good plot is epoch vs loss. Loss should monotonically decrease if you use the entire dataset and no 
momentum or similar things are used in update.

### Train/Val Accuracy vs Epoch Plot
Plot training and validation accuracy, this should give you good insight on whether you're overfitting or not.

### Track the ratio of Weights vs Updates
```
Assume parameter vector W and its gradient vector dW
param_scale = np.linalg.norm(W.ravel())
update = -learning_rate*dW # simple SGD update
update_scale = np.linalg.norm(update.ravel())
W += update # the actual update
print update_scale / param_scale # want ~1e-3
```

### Activation / Gradient distributions per layer
Plot Activation / Gradient histograms for all layers of the networks, you should see the values 
distributed well accross the activation function. They should not be stuck in a few values e.g. -1 and 1.

## Parameter updates

### Vanilla Update
```
# Vanilla update
x += - learning_rate * dx
```

### Momentum Update
With the momentum update, the parameter vector will build up velocity in a direction that is consistent with the 
gradient. The hyperparameter passed is analogous to friction, thus the higher the slower the velocity increases.

### Nesterov Momentum
Slightly different version from regular momentum that enjoys better convergence.

### Annealing Learning Rate
- Step decay: Reduce the learning rate by some factor every few epochs. Typical values might be reducing the learning rate by a half every 5 epochs, or by 0.1 every 20 epochs. These numbers depend heavily on the type of problem and the model. One heuristic you may see in practice is to watch the validation error while training with a fixed learning rate, and reduce the learning rate by a constant (e.g. 0.5) whenever the validation error stops improving.
- Exponential decay. has the mathematical form α=α0e−ktα=α0e−kt, where α0,kα0,k are hyperparameters and tt is the iteration number (but you can also use units of epochs).
- 1/t decay has the mathematical form α=α0/(1+kt)α=α0/(1+kt) where a0,ka0,k are hyperparameters and tt is the iteration number.

### Second order methods
This is basically Newton's Method, problem is that inverting a hessian of 1million parameters is extremely difficult.
Acutally, it's infeaseable.
There are quasi-newton methods such as [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
The huge downside is that it needs to loop over the entire training data thus it can't operate in batches like G.D.

### Per-parameter adaptive learning rate methods
AdaGrad:
```
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Essentially the cache variable has the same size of the gradient and keeps all squared sum of the derivatives.
Epsilon is usually between 1e-4 to 1e-8. Weights with smaller updates have their learning rate increased, the converse 
is true.

RMSPROP:
```
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

A modification on AdaGrad that uses a moving average of the squared gradients and a decay rate.
Just like AdaGrad, updates are not monotonically decreasing.

Adam:
```
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

It is exactly like RMSPROP but we use the smoothed version of the gradient m. We also combine this 
idea with momentum. Recommended values are eps = 1e-8, beta1 = 0.9, beta2 = 0.999.

## Hyperparameter optimization
These are techniques on finding the best hyperparameters, this also includes how one should implement their 
system.

### Implementation
Two main components:
- Worker = continually samples random hyperparameters and performs optimization, checkpoints every time
- Master = launches and kills workers accross the computing workstation

### Prefer one validation fold to cross-validation
Self-explainatory

### Hyperparameter ranges
Search for hyperparameters in a log scale for example:

`learning_rate = 10 ** uniform (-6, 1)`

### Prefer random search to grid search
Self explainatory, paper in repo.

### Careful with best values on border
If you search for a parameter, such as learning rate along some interval as below.

`learning_rate = 10 ** uniform (-6, 1)`

Make sure that the best value is not at the end of one of the intervals, if it is then 
extend your interval along the appropiate direction.

### Stage your search from coarse to fine
Start with coarse ranges and low epochs e.g. 1 epoch. Then move on to finer-grained ranges 
and higher epochs e.g. 5,10,15.

### Bayesian Hyperparameter Optimization
Just some more methods for hyperparameter optimization, these methods aren't well understood yet. 
In practice random-search performs much better.

## Evaluation

### Model Ensembles
A combination of different NNet models, the prediction is averaged out. The performance typically monotonically 
increases.

- Same model, different initializations
- Top models discovered during cross-validation
- Different checkpoints of a single model
- Running average of parameters during training

For more information, look at the notes.
