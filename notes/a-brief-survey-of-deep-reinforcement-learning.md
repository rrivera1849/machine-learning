
# A Brief Survey of Deep Reinforcement Learning

Reinforcement Learning - a principled mathematical framework for experience-driven
  autonomous learning

Deep Learning provides RL with powerful *function approximation* and *representation 
learning* which in turn allows RL to overcome some of the memory and computational 
complexity issues.

## Reward-Driven Behaviour

The state is a *sufficient statistic*, that is, it contains all the information necessary
to take action u_t.

The goal of an RL agent is to learn the policy, that maximizes the expected return or 
cummulative and discounted rewards. 

The main tool are the mathematical formalisms for optimal control, thus RL attempts to 
solve these problems as a subset of optimal control.

RL can be defined as a Markov Decision Process (MDP), which consist of a couple of things:
1. A state space S, plus a distribution of starting states p(s_0)
2. A set of actions A
3. Transition dynamics T(s+1 | s_t, a_t)
4. Instantaneous reward R(s_t, a_t, s_t+1)
5. Some discount factor [0,1]

Given the above description, a policy is defined as:
  pi: S -> P(A = a | S)
In other words, given a state, we get a probability distribution over the actions.

The goal of RL is to find:
  pi^star = argmax_pi E[R|pi]

A disadvantage of the MDP is that it upholds the Markov Property, i.e., only the current 
state affects the next state. Thus the past does not matter.

We can get away from this Markov Property assumption py refraiming the problem as a POMDP 
(Partially Observable Markov Decision Process). In this system, the agent receives an observation 
o_t where the distribution is defined py p(o_t+1 | s_t, a_t). POMDP algorithms mantain a belief
of the current state given previous belief state, the action taken and the current state. 
Using RNN's, a type of dynamical systems we can have some sort of "internal memory".

Some challenges faced by RL:
- The optimal policy must be inferred by trial-and-error interaction thus we only receive the
  reward as our learning signal.
- Observations of the agent depend on its actions and can contain strong temporal correlations.
- Agents must deal with long range dependencies, the consequence of an action is usually not
  seen until the long-term. This is known as the credit-assignment problem.

## RL Algorithms

There are two main approaches, methods based on value functions and those based on policy search.
Actor critic is a hybrid between both of these methods.

A Value function represents the goodness of a given state S, i.e., it estimates the expected return
of starting in state s and following policy pi.
  V(s) = E[R|s,pi]

The optimal value function has an optimal value function and vice-versa.
  V^*(s) = max_pi V(s) ; for all s in S

Given the optimal value function, then the optimal policy can be retrieved
by choosing the action that maximizes:
  E_s_t+1 ~ T(s_t+1 | st, a) [V^*(s_t+1)]

The problem with the value function aboveis that it relies on the transition dynamics which are 
unavailable. Therefore, another function called the state-action value function Q(s,a) is used.
This is similar, except that the initial action is provided thus there is no need for the 
transition mechanics.
  Q(s,a) = E[R|s,a,pi]

### Dynamic Programming 

The Bellman Equation can be used to solve for Q:
  Q(s,a) = Es_t+1[r_t+1 + gamma * Q(s_t+1, policy(s_t+1))]

In short, this means that we can improve while using our current estimates.

This is the foundation of Q-Learning and SARSA as well as GPI where in two steps are applied:
1. Policy Evaluation - improve the estimate of our value function which can be done by minimizing
                       the TD error of some trajectory
2. Policy Improvement - Choose actions greedily based on the evaluated value function

### Monte Carlo

MC methods simply sample the episodic MDP and average the returns, this process is repeated 
many times to allow for convergence.

Another value-function based method is the Advantage function A(s,a) which learns relative
state-action pairs instead of the actual return. This is easier to learn relative to Q.
The relationship is defined as follows: A = V - Q.

### Policy Search Methods

Instead of looking for some value function, we instead optimized a parametrized policy that
could be represented by a neural network. There are two approaches, gradient-free and gradient-based 
methods but in practice gradient-based methods are more sample-efficient when the policies 
contain a large number of parameters. In other words, they can be optimized more efficiently 
with gradients if the parameter space is too high.

The biggest advantage of gradient-free methods is that they can also optimize non-differentiable 
policies. Most approaches involve a heuristic search accross a class of models. Some methods include:
evolution strategies and compressed network search.

In actor-critic methods the actor policy learns by taking feedback from the value function critic. 

### Planning & Learning

Planning - any method which utilises an environment model to produce or improve any policy

## Current Algorithms

### Value Functions 

The Atari 2600 DQN research demonstrates the strength of this approach. Given pixel inputs of size
210x160x3 and 18 joystick actions then the state space is 18 * 256^(210x160x3) which is too large
of a table to construct. A Neural Network allows us to compactly represent this entire feature space.

Experience replay stores (s_t, a_t, r_t+1, s_t+1) tuples in a buffer of size N. We can the sample 
minibatches from this buffer whcih reduces the variance of the learning updates. Furthermore, by 
sampling from the replay buffer the temporal relations are broken.
Prioritizing samples based on TD errors is more effective for learning.

Target networks freeze the weights of the neural network for M steps. This network is then used to 
calculate the TD errors. This is much better than using the rapidly fluctuating real targets.

Using a single esitmator in the Q-learning update rule overestimates the maximum return. It uses the
maximum action value as an approximation of the maximum expected value.
Dueling networks decompose the Q value funtion into other meaningful funtions, e.g., A and V.

### Policy Search 

Trust Region Policy Optimization (TRPO)
Gurantees monotonic improvement in the policy by preventing it from deviating too much from 
previous policies. It does this by using a trust region which restricts the optimisation steps 
to within a region where the approximation of the true cost function still holds. Essentially, 
TRPO constrains each policy to a fixed KL divergence from the current policy.

### Actor Critic 

Deterministic Policy Gradients extend from stochastic applications to deterministic policies.
DPGS only integrate over the state space as opposed to both state and action spaces. This 
requires less samples overall.

In the same work, other work developed a method for stochastic policies called SVG which 
gets rid of the high variance REINFORCE estimator.

Another vein exploits parallel computation in which asynchronous gradient updates are performed 
for both single machine and distributed machine systems.
Asynchronous advantage actor critic methods (A3C) are some of the most popular algorithms. A3C 
combines advantage advantage updates with the actor-critic formulation and relies on asynchronous updated 
policy and value function networks trained in parallel.
