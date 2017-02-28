# Deep Residual Learning for Image Recognition

This paper attempts to solve the degradation problem posed when training very
deep networks. We're talking about 30+ layers. The typical problem one runs
into is the "degradation" problem. During training, the error of a deeper network
tends to be higher than the error of a more shallow network. This is probably
due to the difficulty of converging on such deep architectures.

The idea is to approximate the "Residual" function instead. If we assume that
some combination of layers can asymptotically approximate complicated functions,
then it follows that they can also approximate residual functions such as
H(x) - x = F(x). This yields the following formulat F(x) + x = H(x). We can
leverage shortcut connections that perform element wise addition to achieve this
goal. 

The hypothesis is that F(x) + x is an easier function to converge on thus mitigating
the degradation problem. This formula assumes that F(x) and x are both the same dimension
but in practice we might have to adjust the dimensionality of x. Thus it becomes
y = F(x,Wi) + Ws x. Where Ws is some parameter that increases or decreases dimensionality.
In practice we either zero pad or use 1x1 convolutions before and after to control the
dimensionality of our input and output.

Further reading:
Network in Network
