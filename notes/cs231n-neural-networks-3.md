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
First good plot is epoch vs loss. Loss should monotonically decrease if you use the entire dataset.

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


