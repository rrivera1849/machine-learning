
# Introduction to GANS
These notes are based on [Ian Goodfellow's Intro](https://www.youtube.com/watch?v=RvgYvHyT15E&list=PLJscN9YDD1buxCitmej1pjJkR5PMhenTF&index=1)

Two Neural Networks are the 'players'. One generates the worst possible input and the other 
attempts to classify correctly.

Find PDF or generate more samples from the training distribution.

Both of these networks compete against each other.

D -- Discriminator

G -- Generator

The Discriminator attempts to answer the question 'Did this example come from the
training distribution?'.
While the generator attempts to generate examples that are as close as possible to
the training distribution such that D(x) =~ 0.5. In other words it attempts to generate
examples that are so close to the training distribution that the discriminator can no
longer distinguish between what is real & what is fake.

Two cost functions were defined according to the Minimax Game formulation. Two loss
functions are defined for each network. The equilibrium of the game is the saddle point
of the discriminator loss.
Estimates the ratio between the distribution of the data and the distribution of the model.

Ian recommends that we use the Non-Saturating game. This is more of a practical tip & trick
to generate these. You can do math with the Z latent variables.

Problems with non-convergence. We want to do MinMax not MaxMin but gradient descent is very
symmetric which means that we don't know where we will end up.
