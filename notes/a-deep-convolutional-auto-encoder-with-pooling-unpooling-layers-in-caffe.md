
# A Deep Convolutional Auto-Encoder with Pooling Unpooling Layers in Caffe

Rules of thumb:
1. Model should be symmetric in terms of the total size of feature maps and number
   of neurons in hidden layers.
2. They use CrossEntropyLoss coupled with the MSELoss 
   What are the targets for CEL?
3. Visualize intermediate values along with the numerical representations of the
   filters to get a better understanding of what is going on.
4. Because the nature of convolutions is multiplication, then the values will just
   increase accross the model with hinders convergence. Use Tanh and Sigmoid to combat.
5. Same convergence results should be obtained on at least 3 runs, this they call "Stability".

Preprocessing was only Normalize between [0,1].
Five models were attempted with different number of convolutions, linear and pooling-unpooling 
methodologies. 

Just pinning the unpooled features to a pre-defined corner, i.e. centralined unpooling, encourages
the model to rely on the learned encoded representation instead of being overly sensitive to the
input.

Model 3 might've not performed well due to the simplicity of the MNIST dataset images.
