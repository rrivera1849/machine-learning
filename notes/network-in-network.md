# Network In Network

Convolution operations can be thought of as GLM's. That is, the convolution
filter only applies a linear operation. This has the disadvantage that all
the latent information pertaining to your task must be liearly separable.
The key point of this paper is to use a Multi-Layer-Perceptron as a filter,
this mitigates the assumption of a linearly separable field. Instead, it provides
a universal function approximator. We can interpret this as extracting higher
quality abstract feature maps to use during our classification.

The architecture also makes use of what they term "Global Average Pooling".
The procedure is as follows:

     pooled_vector = []
     for f in feature_maps:
       pooled_vector.append(average(f))

We have a total of NUM_CLASSES feature maps. This methodology forces each feature
map to offer information about each class. The last figure of the paper shows a nice
visualization of this.
