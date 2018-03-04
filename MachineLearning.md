# Machine Learning

- Classification : classify data. The Line separates the various data. The output is discrete/categorical
- Regression : predict. The Line fits the data. The output is real number/continuous

- Supervised Learning
  - Support Vector Machine (SVM)
  - Neural Networks
    - Convolutional Neural Networks (CNN)
    - Deep Neural Networks (DNN)
- Unsupervised Learning
  - Clustering
    - k-means
  - Dimensionality Reduction
    - t-SNE
  - Neural Networks
    - Generative Adversarial Networks (GAN)
- Reinforcement Learning
  - Hybrid of the two


## Functions

- Sum (Z = Weight * x + Bias)
  - z = wx + b

- Logistic (Sigmoid)
  - S-shaped range [0, 1]
  - Sigmoid = 1/(1 + exp(-x))

- Tanh (Rescaled Sigmoid)
  - S-shaped range [-1, 1]
  - Tanh = 2*Sigmoid - 1

- ReLU (Rectified Linear Unit)
  - ReLU = max(0, x)

- Cross-Entropy
  - CE = y*log(a(z)) + (1 - y) * log(1-a(z))

- Mean Squared Error (MSE)
  - MSE = (y - z(x))^2/2

- Softmax
  - range [0, 1], add up to 1
  - Softmax = exp(z(i)) / sum(z(i))


## Linear & Non-Linear

- Linear Classifiers (The simplest ml algorithm)
  - Separate a p-dimensional vector Data with a (p-1)-dimensional hyperplane.

- Cost Function : Cost, Loss, Penalty, Quadratic Cost, Squared-Error Cost function. (MSE)

- Linear Regression : Find a Hypothesis function that minimizes cost function.
  - Hypothesis Function : Ordinary Least Squares

- Gradiebt Descent : A gradient is a multidimensional generalization vector containing each of the partial derivatives of the cost function with respect to each variable.

- Non-Linear : Minimize cost function using Gradient Descent.

## Neural Networks

### Concepts

- Neurons
  - Activation function : take inputs and outputs hypothesis()
    - Sigmoid, Tanh, ReLU
    - Always output one value, no matter how many subsequent Neurons it sends it to.

- Bias Units
  - No inputs or connections going into them
  - Always output value +1

- Layers
  - A series of layers of Neurons with connections

- Receptive Field

- Forward Propagation
  - Input Layer > Hidden Layer > Output Layer

- Back Propagation
  - An efficient way to compute partial derivatives.
  - Effectively Solve linearly inseparable patterns of Exclusive-or (XOR) function in Multi-layer Networks.
  - Non-Linear classification. Classification and sub-classification.
  - An expression for the partial derivative (the cost function with respect to any weight or bias) which tells how quickly the cost changes when the weights and biases are changed.
  - Compute all the elements of the gradient in a single forward and backward pass through the network.
  - A practical application of Chain Rule.
  - Propagate back based on how much that weight contribute to the error.

### NN Prototype & Evolution

```
NN Loop
  |- DataSet
  |   |
  |  Input Layer
  |   |- Data
  |   |- Weights
  |   |- Biases
  |   |- Sum
  |   |- Activation Function
  |  Hidden Layer
  |   |- Deepth
  |  Output Layer
  |   |- Output Function
  |   |- Cost Function
  |   |- Back Propagation
  |
  |- Error Analysis
  |- Result Analysis
```

```
NN Loop - (Matrices & Vectors Acceleration)
  |- DataSet
  |   |- Data Normalization
  |   |- Training/Validation/Test
  |       |- Cross-fold validation
  |- Hpyer-Parameters Random Initialization
  |   |- Symmetry Breaking
  |- Mini-Batch
  |   |
  |  Input Layer
  |   |- Data
  |   |- Polynomial Features
  |       |- x_1^2, x_2^2, x_1*x_2, etc
  |   |- Weights
  |   |- Biases
  |   |- Sum
  |   |- Activation Function
  |       |- Sigmoid
  |       |- Tanh
  |       |- ReLU
  |  Hidden Layer
  |   |- Convolution
  |   |- Deepth
  |   |- Dropout
  |   |- Maxpooling
  |   |- Fully Connected
  |  Output Layer
  |   |- Output Function
  |   |- Cost Function
  |   |   |- Regularization Term
  |   |   |- Cross-Entropy Function
  |   |- Back Propagation
  |       |- Gradient Descent
  |           |- Stochastic Gradient Descent
  |           |- Momentum Gradient Descent
  |           |- Accelerated Gradient Descent
  |           |- Adaptive SubGradient (AdaGrad, AdaDelta, RMSProp, Adam)
  |
  |- Error Analysis
  |   |- Hyper-Parameters
  |- Result Analysis
  |   |- Under-Fitting (Bias)
  |   |- Over-Fitting (Variance)
  |- Visualization
```

### DataSet

#### Problems

- Normalize DataSet
- Utilize DataSet

The model is over-optimized to accurately predict the training set. The model greatly twists itself to perfectly conform the to training set, even capturing its underlying noise.

#### Improvements

- Scaling DataSet
  1. x_i = x_i - mean(X)
  2. Matrices and Vectors

- Training/Validation/Test DataSets
  1. Avoid Over-Fitting problem, ignore noise.
  2. Maximize the use of data.

- Cross-fold validation
  1. equally-sized partitions, and each partition takes turns being the validation set.



### Gradient Descent

Gradient Descent evaluates over all of the points in the dataset (Batch Gradient Descent).

It is a "convex" function which is good to find minimum.

Learning Rate (denoted as alpha). (low, too long time; high, overshoot the correct path)

#### Problems

- Local Minimum Problem
- Learning Rate is a Hyper-Parameter
- Saddle Points

#### Improvements

- Stochastic Gradient Descent (SGD)
  1. Shuffle the dataset.
  2. Go through each sample individually.
  3. Calculate the gradient with respect to that single point.
  4. Perform a weight update for each.

- Mini-batch Gradient Descent (MB-GD)
  1. Dataset is randomly subdivided into N equally-sized mini-batches of K samples each.
  2. K=1, it is SGD, K=dataset.size, it is BGD.

- Momentum Gradient Descent
  1. Gradient is gradually adjusted from the rate of the previous update.
  2. The higher parameter beta is set, the more momentum update is.
  3. beta=0, it is ordinary GD.

Momentum Gradient Descent helps to escape saddle points and local minima by rolling out from them via speed built up from previous updates.

- Accelerated Gradient Descent
  1. Rather than evaluating the gradient where it currently is, instead evaluate the gradient at approximately where it will be at the next time step.

- Adaptive SubGradient (AdaGrad, AdaDelta, RMSProp, Adam)
  1. Adapt the Learning Rate alpha to each parameter individually.
  2. RMSProp adds epsilon, Adam adds beta_1, beta_2

- Learning Rate Decay
  1. Adjust Learning Rate according to the loop epoches.


### Over-Fitting

#### Improvements

- Regularization Term
  - Impose constraints to prevent over-fitting or otherwise discourage undesirable properties.
  - Terms
    - L1-Regularization
    - L2-Regularization

Regularization term helps Gradient Descent find a parameterization which does not accumulate large weights.

- Dropout
  - Layer applied with Dropout randomly deactivate 20% - 50% Neurons along with their connections.

Dropout is to reduce the network's tendency to come to over-depend on some neurons which forces the network to learn a more balanced representation, and helps combat overfitting.



### Others

#### Hyper-Parameters

Random Initialize Hyper-Parameters.

#### Activation Function

Sigmoid will lead to Vanishing Gradient Problem in a neural networks with many layers. The rectified Linear Unit (ReLU) is used to solve this problem.

#### Learning Slowdown

If the output neurons are linear neurons then the quadratic cost will not give rise to any problems with a learning slowdown.

#### Debug

1. More training data
2. Smaller sets of data
3. Try getting additional features
4. Try adding polynomial features
5. Try decreasing lambda
6. Try increasing lambda

Bias (under-fitting) vs Variance (over-fitting)
Bias : J_train = J_validation are high, try large lambda.
Variance : J_train is low, J_validation is high, try small lambda.

Adjust lambda : 0, 0.01, 0.02, 0.04, ... 10 to find 'just right' Bias and Variance.

Precision vs Recall

Precision = TruePositives / (TruePositives + FalsePositives)
Recall = TruePositives / (TruePositives + FalseNegatives)


## Neural Networks Milestones

### CNN

```
1. Convolution : Extract Convolved features (activation maps)
2. ReLU : Rectified Linear Units Layer
2. Pooling (Max or Mean) : Aggregate statistics of features at various locations (down-sample the activation maps)
3. Subsampling
4. Back Propagation (up-sample)

>>> Convolutional layers
  filter size, stride, spatial arrangement, padding

>>> Pooling layers (Subsampling)
  a factor of 2 in both dimensions.
```

### DNN

```
Deep Learning can learn useful representations or features directly from images, text and sound.
```

### RNN

```
RNN
  |- Long Short Term(LSTM) : Cell state and Hidden state.
  |- Gated Recurrent Unit(GRU)
```

### GAN

```
Generator and Discriminator
```

```
LeNet - Yann LeCun (1998)
  Convolution > Subsampling > Convolution > Subsampling > Full Connection > Full Connection > Gaussian Connections

ImageNet - Fei-Fei Li
  14 million labeled images & ImageNet Challenge

AlexNet - Geoffrey Hinton, Ilya Sutskever, Alex Krizhevsky (2012)
  Convolution > Max pooling > Convolution > Max pooling > Convolution > Convolution > Convolution > Fully connected > Fully connected > Fully connected(Softmax)

VGGNet (2014)

GoogleLeNet (2014)
  Inception module

ResidualNet (2015)
```

```
AlexNet

- ReLu instead of Sigmoid : Sigmoid will cause Vanishing Gradient Problem due to the vanished updates to the weights in the saturating caused by small derivative.
- Dropout : reduce over-fitting

VGG16

- Multiple 3x3 kernel-sized filters instead of 11x11 and 5x5 : multiple Receptive Field allows to learn more complex features with less parameters.
- Blocks/Modules

GoogLeNet/Inception

- Refine Dense Connection Architecture by applying concept of Sparse Weight/Connection using Inception modeule (Convolute 1x1 to reduce sizes)
- Global Average Pooling instead of Fully-Connected Layers
- Large width and depth
- Improve the accuracy and save on computation

Residual Networks

- Residual module to learn the features on top of already available input which makes sure deeper networks actually work.
```

## Neural Networks Examples

### MNIST

### Artistic Style Transfer

Find an output image which minimizes a loss function that is the sum of content loss (the content dissimilarity between the content image and the output image)
and style loss (the style dissimilarity between the style image and the output image)

Adjust pixels instead of weights.

Loss_content(c, o)
Loss_style(s, o)

### TensorFlow

  TensorFlow
  TensorBoard


### Keras
