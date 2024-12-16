import numpy as np
import os
import mnist
from PIL import Image
import sys

def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))

def sig_prim(x: float):
    y = sigmoid(x)
    return y * (1-y)

def tanh(x: float):
    return 2 * sigmoid(2 * x) - 1

def softmax(z: np.array):
    denominator = sum(np.exp(z))
    return np.exp(z) / denominator

def softmax_prim(z: np.array):
    a = softmax(z)
    return a * (1 - a)

def mse_prim(output: float, y_input: float):
    return output - y_input

def cross_entropy_prim(output: float, y_input: float):
    return -(y_input / (output + 1.e-15) - (1 - y_input) / (1 - output + 1.e-15))

def log_likelihood_prim(output: float, y_input: float): #not used, a single term in the cross entropy corresponding to some y_i = 1
    return -y_input / (output + 1.e-15)

class Network:
    def __init__(self, size: np.array, learning_rate: float, l: float = 0):
        self.layers = len(size)
        self.size = size
        self.bias = [np.zeros(size[i]) for i in range(1, self.layers)]
        self.weights = [np.zeros((size[i+1], size[i])) for i in range(0, self.layers - 1)]
        self.eta = learning_rate
        self.l = l

    def gaussian_bias(self, mean: float, var: float):
        self.bias = [np.random.normal(mean, var, (self.size[i])) if i < self.layers - 1 else np.zeros(self.size[i]) for i in range(1, self.layers)]

    def gaussian_weights(self, mean: float, var: float):
        self.weights = [np.random.normal(mean, var, (self.size[i+1], self.size[i])) for i in range(0, self.layers - 1)]

    def forward_output(self, input: np.array, activation: callable, smax: bool = False): #Add softmax
        output = input
        zs = [input]
        for i in range(0, self.layers - 2):
            zs.append(np.matmul(self.weights[i], output) + self.bias[i])
            output = activation(np.matmul(self.weights[i], output) + self.bias[i])
        zs.append(np.matmul(self.weights[self.layers - 2], output) + self.bias[self.layers - 2])
        if smax == True:
            output = softmax(np.matmul(self.weights[self.layers - 2], output) + self.bias[self.layers - 2])
        else:
            output = activation(np.matmul(self.weights[self.layers - 2], output) + self.bias[self.layers - 2])
        return [zs, output]
    
    def forward_sigmoid(self, input: np.array, smax: bool = False):
        return self.forward_output(input, sigmoid, smax)
    
    def output_error(self, output: np.array, y_input: np.array, z_values: np.array, cost: callable, smax: bool = False):
        if smax == True:
            return cost(output, y_input) * softmax_prim(z_values[-1])
        return cost(output, y_input) * sig_prim(z_values[-1])
    
    def backpropagate(self, output_error: np.array, z_values: np.array):
        deltas = [output_error]
        for i in range(1, self.layers - 1):
            deltas.append(np.matmul(np.transpose(self.weights[-i]), deltas[-1]) * sig_prim(z_values[-i - 1]))
        return deltas
    
    def partial_weights(self, outputs: list, deltas: list):
        partial_w = []
        for i in range(self.layers-1):
            partial_w.append(np.matmul(np.reshape(deltas[-1-i], (-1, 1)), np.reshape(outputs[i],(-1,len(outputs[i])))))
        return partial_w
    
    def gradient_descent_single(self, input: np.array, y_input: np.array, cost: callable, smax: bool = False):
        outputs = self.forward_sigmoid(input, smax)
        output_error = self.output_error(outputs[1], y_input, outputs[0], cost, smax)
        deltas = self.backpropagate(output_error, outputs[0])
        partial_weights = self.partial_weights([sigmoid(x) for x in outputs[0]], deltas)
        partial_biases = [deltas[i] if i < self.layers - 1 else np.zeros(self.size[i]) for i in range(1, self.layers)]
        return [partial_weights, partial_biases]

    def gds_mini(self, input: np.array, y_inputs: np.array, cost: callable, smax: bool = False):
        length = len(input)
        for i in range(length):
            gradient_single = self.gradient_descent_single(input[i], y_inputs[i], cost, smax)
            if i == 0:
                partial_weights = gradient_single[0]
                partial_biases = gradient_single[1]
            else:
                partial_weights = [x + y for x,y in zip(partial_weights, gradient_single[0])]
                partial_biases = [x + y for x,y in zip(partial_biases, gradient_single[1])]
        for i in range(self.layers - 1):
            self.weights[i] -= self.eta * self.l * self.weights[i] + self.eta * partial_weights[i] / length #first term is L2 regularisation term, set to zero by default, add L1 regularisation with np.sign(self.weights[i])
            self.bias[i] -=  self.eta * partial_biases[i] / length 
            
    def testing(self, input: np.array, y_inputs: np.array):
        correct = 0
        for i in range(len(input)):
            for j in range(len(input[i])):
                result = test.forward_sigmoid(input[i][j], True)[1]
                result = np.argmax(result)
                if result == np.argmax(y_inputs[i][j]):
                    correct += 1
        print(f'Correct: {correct}')

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(2)
    np.set_printoptions(precision=3, suppress = True)
    while True:
        command  = int(input("command: "))
        if command == 0:
            test = Network([800,20,3], 0.1, 0)
            test.gaussian_bias(0,0.1)
            test.gaussian_weights(0,0.2)
            for i in range(1):
                if i % 3 == 0:
                    test_input = [np.random.normal(-1,0.02,800)]
                    test_y = [np.array([1,0,0])]
                elif i % 3 == 1:
                    test_input = [np.random.normal(0,0.02,800)]
                    test_y = [np.array([0,1,0])]
                elif i % 3 == 2:
                    test_input = [np.random.normal(1,0.02,800)]
                    test_y = [np.array([0,0,1])]
                test.gds_mini(test_input, test_y, cross_entropy_prim, smax = True)
                print(i, test.forward_sigmoid(np.random.normal(-1,0.2,800), True)[1], test.forward_sigmoid(np.random.normal(0,0.02,800), True)[1], test.forward_sigmoid(np.random.normal(1,0.2,800), True)[1])
        elif command == 1:
            test = Network([784,30,10], 0.1, 0.00005)
            test.gaussian_bias(0,0.1)
            test.gaussian_weights(0,0.2)
        
            mn = mnist.mnistOpener(60000, 10, 0, 1000)
            data = mn.split_data()
            for j in range(10):
                p = np.random.permutation(5000)
                print(j)
                for i in range(5000):
                    test.gds_mini(data[0][p[i]], data[3][p[i]], cross_entropy_prim, smax = True)
                test.testing(data[2], data[5])
            #for i in range(10):
            #    print(test.forward_sigmoid(data[2][0][i], True)[1])
            #print(data[5][0])
        elif command == 2:
            for i in range(10):
                im = Image.open(f'{i}.png').convert('L')
                pixels = np.array(im.getdata()) / 255
                print(test.forward_sigmoid(pixels, True)[1])
        elif command == -1:
            break