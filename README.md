#title nlpod
hi all, I'm Mehrdad. I will explain about a hybrid reduced order model as an
example of our work in cfdlab

#overview
Please raise your hand if you've ever heard about Convolutional Auto Encoder or CAE for short. please raise your hand
if you've ever heard about Proper Orthogonal Decomposition or POD for short.
  who know about LSTM or time series predictor in general.

#cae
lets say we have a collection of snapshots or images from a system. In such
situation we can use autoencoder network to compress data without chopping out part of it.
As you can see in this slide, there is two part to an autoencoder,
encoder and decoder. While training we use the snapshots as both input and
output to train an encoder which can compress to a small latent space in the
middle and a docker capable of decompressing the latent space.
convolutional is the type of filters that connect each consecutive layers.

#pod
Proper Orthogonal Decomposition is a numerical method that also enables
reduction in complexity of a system. By operating Singular Value Decomposition
on a set of snapshots we transform the system state ino orthogonal modes sorted
by their significance in the sytem or their energy. Since they are
sorted we can select a few number of modes with highest energy to achieve
a reduced order model.

#lstm
For the last two slides I covered a couple of methods to compress snapshots.
OK, one last step to model the dynamics of a system is to evolve the compressed
modes in time. There are many time series predictors in deep-learning area. One
of which is LSTM or Long-Short term memory. LSTM can recognize long and
complicated patterns and is easy to use with Tensorflow Framework.

#nlpod
OK, so Now we have all the building blocks for a NLPOD reduced hybrid model.
We want to compress the system as much as possible without loosing much
accuracty so let me explain to you the main flow of the information in this
network after it is trained.
Snapshots go through a POD and a generous number of low energy modes are
discarded. The remaining modes will be compressed with an encoder to
a latent-space with only a handfull of modes which go through a time-series
predictor on the latent space to predict the evolution of latent space in time.
The latent space then can be turned back to pod modes with pod reconstruction
and back to input snapshots space with the decoder.

#fom
simulation of Riemann problem governing by the Euler equation in 1x1 square
with initial value of 0.1, 0.2, 0.3, 0.4, 0.5 for top right quadrant and 1 for
the other three quadrant. As you can see the Pressure filed for one of the
cases in this slide. A shock will be generated and move toward the top right of
the square.
the case 0.1,0.2,0.3,and 0.5 are used for the training and 0.4 case for the
testing of the performance of the model.

#table

#thanks
