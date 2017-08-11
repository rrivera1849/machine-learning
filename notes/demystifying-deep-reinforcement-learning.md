
# Demystifying Deep Reinforcement Learning
[Blog Post](https://www.intelnervana.com/demystifying-deep-reinforcement-learning/)

## Preliminaries

Starts off with the very famous atari paper from DeepMind and states the goal to
be "demystifying" these methods. 

RL is between unsupervised and supervised learning, wherein we have sparse and
time-delayed labels -- the rewards. Oftentime, getting reward does not directly 
correlate with the actions that you took right before the event, figuring out which 
actions correlate with the reward is called the *credit-assignment problem*.

If we look at the Q-Learning update equation:

Q[s,a] = Q[s,a] + alpha * (r + discount * argmax_a' Q[s',a'] - Q[s,a])

If alpha = 1 then:
Q[s,a] = r + discount * argmax_a' Q[s,a']

Which gets us back to the Bellman Equation.

## Deep Q-Network

The above is a table-based approach which lacks scalability and knowledge won't transfer 
to other games. In the game of breakout, a state can be represented by the position of 
the paddle, position and direction of the ball and the presence of each individual brick.
The problem with this representation is that it is game-specific just as the table based method 
above. Because of this, we use screen pixels, two frames can get us the direction as well.

A Neural Network is a natural representation to go from our 4-frames to Q-values. It could take
either the state or both the state and action as input. In the former case, a Q-value for each
action would be the output and in the latter only a single Q-value would be the output. The former
has the advantage of only requiring one forward pass to find the maximum value. 

All we need now is to minimize the squared error loss:
L = 1/2 * (r + max_a' Q(s',a') - Q(s,a))

The problem with these non-linear systems is that they are very unstable. Getting them to converge
requires a bag of tricks, one such trick is experience replay. Here, we build a dataset of (S,A,R,S')
tuples and simply sample them over and over in order to train. This makes every example look different
from the next or iid like and allows our network to converge. 


