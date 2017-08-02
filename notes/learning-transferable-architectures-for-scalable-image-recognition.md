
# Learning Transferable Architectures for Scalable Image Recognition

The idea is to use the Neural Architecture Search (NAS) algorithm previously 
proposed to optimize architecture optimizations. In contrasts to previous efforts, 
the paper tries to optimize a small convolutional cell on the CIFAR-10 dataset making 
it much less expensive.

NAS Training Procedure: 
  A controller RNN samples child networks with different architectures
  The child networks are trained to convergence to obtain some accuracy on
    the held-out validation set
  The resulting accuracies are used to update the controller so that it will 
    generate better architectures over time. The weights are updated using the
    policy gradient method.

They attempt to build two convolutional cells which they term NormalCell
and ReductionCell. The number of filters is doubled whenever the spatial dimension
is halved.
The initial convolution and #motifs are free parameters.

Fun note: They use the word cell because they're similar to RNN's in that the structure
  of the cell is independent of the number of timesteps.

The set of hidden states includes real hidden states from RNN's and the input image.
In general the RNN predicts the following:
1. Select h from the set of hidden states created in previous blocks
2. Select another hidden state as in 1
3. Select operation to apply to h1
4. Select operation to apply to h2
5. Concatenate step 3 and 4 to a new h3

If B is the number of blocks in the convolutonal cell then we must perform the above
5 * B times. If we include the Reduction Cell then it is 2 * 5 * B. In their experiments B=5.
PPO (Proximal Policy Optimization) used with a global workqueue of 450 GPU's.

Things that Worked/N-Worked:
  BNorm and/or ReLU between depthwise and pointwise operations did NOT help.
  L1 Regularization did NOT work.
  Dropout did NOT work.
  ELU did about the SAME.
  Stochastically dropping paths did well.

Overall, the work is quite impressive. The key insight here is to create convolutional cells
instead of an entire network architecture. The architectures exceed state of the art in many cases
while using much less parameters. The most impressive part for me was this fact.
