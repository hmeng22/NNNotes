# Naive Neural Network
# nn231 using numpy.

import numpy as np

def activations(t):
    return {
        'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x), (0,  1), .45),
        'tanh': (lambda x: np.tanh(x), lambda x: 1 - x**2, (0, -1), 0.005),
        'ReLU': (lambda x: x * (x > 0), lambda x: x > 0, (0, 10000), 0.0005),
    }.get(t, 'sigmoid')

# inputs and outputs dataset
# | a | b | a xor b |
# | 0 | 0 |    0    |
# | 0 | 1 |    1    |
# | 1 | 0 |    1    |
# | 1 | 1 |    0    |
training_set_length = 4
# scaling data
training_set_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_set_outputs = np.array([[0], [1], [1], [0]])

class nn231_1():

    def __init__(self):
        self.layer_1_input, self.layer_2_hidden, self.layer_3_output = 2, 3, 1

    def train(self):
        # number of iterations
        epochs = 60000

        # activation
        (a, a_derivative, (mina, maxa), L) = activations('sigmoid')

        # weights
        weights_12 = np.random.randn(self.layer_1_input, self.layer_2_hidden)
        weights_23 = np.random.randn(self.layer_2_hidden, self.layer_3_output)
        
        # activation on the output layer
        for t in range(epochs):
            # feed forward
            outputs_2 = a(np.dot(training_set_inputs, weights_12))
            outputs_3 = a(np.dot(outputs_2, weights_23))

            # back propagation
            delta_outputs_3 = (training_set_outputs - outputs_3) * a_derivative(outputs_3)
            delta_outputs_2 = delta_outputs_3.dot(weights_23.T) * a_derivative(outputs_2)

            weights_23 += outputs_2.T.dot(delta_outputs_3)
            weights_12 += training_set_inputs.T.dot(delta_outputs_2)

        print(outputs_3)

nn231_1().train()



class nn231_2():

    def __init__(self):
        self.layer_1_input, self.layer_2_hidden, self.layer_3_output = 2, 3, 1    
        
    def train(self):
        epochs = 20000

        # learning rate L
        (a, a_derivative, (mina, maxa), L) = activations('sigmoid')

        weights_12 = np.random.randn(self.layer_1_input, self.layer_2_hidden)
        weights_23 = np.random.randn(self.layer_2_hidden, self.layer_3_output)

        # No activation on the output layer, faster and accurate
        for t in range(epochs):
            outputs_2 = a(np.dot(training_set_inputs, weights_12))
            outputs_3 = np.dot(outputs_2, weights_23)

            delta_outputs_3 = (training_set_outputs - outputs_3) * L
            delta_outputs_2 = delta_outputs_3.dot(weights_23.T) * a_derivative(outputs_2)

            weights_23 += outputs_2.T.dot(delta_outputs_3)
            weights_12 += training_set_inputs.T.dot(delta_outputs_2)

        print(outputs_3)
        
nn231_2().train()



class nn231_3():

    def __init__(self):
        self.layer_1_input, self.layer_2_hidden, self.layer_3_output = 2, 3, 1  
        
    def train(self):
        epochs = 3000

        # mini batch size
        mini_batch_size = 1

        (a, a_derivative, (mina, maxa), L) = activations('sigmoid')
        
        # Normalize initial parameters
        w_bound_12 = np.sqrt(2.0 / (self.layer_1_input))
        w_bound_23 = np.sqrt(2.0 / (self.layer_2_hidden))
        weights_12 = np.random.uniform(size=(self.layer_1_input, self.layer_2_hidden), low=-w_bound_12, high=w_bound_12)
        weights_23 = np.random.uniform(size=(self.layer_2_hidden, self.layer_3_output), low=-w_bound_23, high=w_bound_23)

        # biases
        biases_2 = np.ones((1, self.layer_2_hidden))
        biases_3 = np.ones((1, self.layer_3_output))

        cost_history = []

        # mini-batch Stochastic Gradient Descent
        for t in range(epochs):
            epoch_cost_history = []

            mini_batches = [(training_set_inputs[k:k + mini_batch_size], training_set_outputs[k:k + mini_batch_size])
                            for k in range(0, training_set_length, mini_batch_size)]

            # train each single mini_batch
            for (mini_batch_inputs, mini_batch_outputs) in mini_batches:
                outputs_2 = a(np.dot(mini_batch_inputs, weights_12) + biases_2)
                outputs_3 = a(np.dot(outputs_2, weights_23) + biases_3)
                epoch_cost_history.append(np.sum((mini_batch_outputs - outputs_3)**2))

                delta_outputs_3 = (mini_batch_outputs - outputs_3) * a_derivative(outputs_3)
                delta_outputs_2 = delta_outputs_3.dot(weights_23.T) * a_derivative(outputs_2)

                mini_batches_length = len(mini_batch_inputs)

                weights_23 += outputs_2.T.dot(delta_outputs_3) * (L / mini_batches_length)
                weights_12 += mini_batch_inputs.T.dot(delta_outputs_2) * (L / mini_batches_length)

                biases_3 += delta_outputs_3 * (L / mini_batches_length)
                biases_2 += delta_outputs_2 * (L / mini_batches_length)

            mse = np.average(epoch_cost_history)
            cost_history.append(mse)

        print(mse)
        
nn231_3().train()
