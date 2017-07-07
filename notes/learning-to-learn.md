# Learning to Learn by Gradient Descent by Gradient Descent

CODE: https://github.com/deepmind/learning-to-learn

Instead of having a regular update rule that just multiplies a learning rate
by the gradients, the authors propose to use a Recurrent Neural Network that
automatically learns how to optimize for the current problem. 

The LSTM optimizer takes in as inputs the gradients for the current timestep as well
and outputs the next update. It does this in a coordinate-wise way such that we run
the LSTM once for each parameter in the network. This results in a very small network
that can be run very quickly. 
