# Distilling the Knowledge in a Neural Network

The goal of this paper is to "transfer" the knowledge of a bigger neural network
into a smaller "distilled" neural network for deployment. We want to transfer as
much knowledge about generalization as possible.

A key insight on getting networks to generalize is that the probabilities of incorrect
classifications tell us a lot about how a model tends to generalize. The probabilities
of incorrect classifications can vary by orders of magnitude even though the magnitude
is very small.

The authors introduce the "temperature" parameter on the softmax function. If we increase
this value then we get a softer probability distribution over the classes. We then train
the smaller network on these "softer distributions" produced by the bigger model.

There are two ways to transfer knowledge:
1. The simplest way is to keep an unlabeled "transfer" set that is used to train our smaller
   model after our cumbersome model has been trained. We set the TEMPERATURE of both models
   to the SAME HIGH value and use as the objective function cross entropy with the SOFT
   DISTRIBUTION of the cumbersome model.

   This has the advantage that none of the data needs to be labeled.

2. If we have the true labels available, then we can do a weighted average of cross entropy
   on the SOFT DISTRIBUTION and cross entropy with LOGITS. Where the weight of the SOFT DISTRIBUTION
   is larger. We also make sure to multiply by T^2 which ensures that the contribution from both the
   SOFT and HARD targets remain relatively the same even if we change the hyperparameter T. The 
   temperature of the cross entropy with LOGITS should be set to 1. 
