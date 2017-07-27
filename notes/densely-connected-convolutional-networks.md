
# Densely Connected Convolutional Networks

Paper award winner from CVPR 2017. Idea is to connect every layer l_ to every L_ - l_
layer ahead of it through the concatenation operation. Instead of a regular stacked CNN 
which can be considered to transform one state to another, densenet explicitly keeps 
all of the information and generates new feature maps as well. The number of feature maps
is much lower than regular networks, e.g. F=12 thus it has less number of parameters.

ResNet allows gradients to flow backwards but it does so through a summation, which may
impede the information flow in the network.

Every layer is connected to every subsequent layer. Consequently layer l_ receives 
the feature maps of all preceding layers x_0, ..., x_l-1. 

x_l = H_l([x_0, ..., x_l-1])
Where [] refers to concatenation of the feature maps.
H_l = BatchNorm -> ReLU -> Conv-3x3 (DenseNetA)

Pooling is an important part of CNNs but we can't concatenate feature maps accross pools.
The network is then divided into "DenseBlocks" that wherein the transition operation can be:
Convolution -> Pooling
BN -> 1x1 Conv -> 2x2 Average Pooling

H_l = BatchNorm -> Relu -> Conv-1x1 (4*k) -> BN -> ReLU -> Conv3x3 (DenseNetB)

To further reduce the number of parameters, they reduce the number of feature maps at
transition layers. If m_ is the number of feature maps then theta_ * m_ where theta_
is the compression factor and theta_ less_equal_ 1 

If less than 1 then it is DensnetC
If less than 1 and bottleneck then DenseNetBC

Overall, DenseNet-BC outperforms other networks in many image classification tasks with much
fewer parameters. It only needs about 1/3 parameters of previously state of the art methods.
Implicit Deep Supervision is caused by the connectivity, thus one classifier can supervise
all layers through at most 2 or 3 transition layers. Overall feature reuse is enhanced when
using the BC variant of the network. The heatmap in the next to last page shows that the first
dense block does indeed take advantage of previous layer feature maps but this does not hold true
accross transition layers. It seems that the reuse is lowered accross transition layers and that
the output of transition layers is very diluted or never used.
