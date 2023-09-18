"""
Logistic regression model that I made in the past, re-tuned for this new task.
Can be found documented in my other files.
"""

import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression:
    def __init__(self, training_data,split):
        split = int(len(training_data)*split)

        self.training_data = training_data[:split]  
        self.test_data = training_data[split:]
            
        self.thetas = np.random.randn(1,3)
        self.default_parameter_test = False
        self.default_parameter_train = False

    def plot_training_data(self):
        if self.default_parameter_train:
            self.__remove_default_parameter(self.training_data)
            self.default_parameter_train = False

        for point in self.training_data:
            if point[2]:
                plt.plot(point[0], point[1], marker="o", markersize=10, 
                         markerfacecolor="green", markeredgecolor="green")
            else:
                plt.plot(point[0], point[1], marker="x", markersize=10,
                          markerfacecolor="green", markeredgecolor="green")

    def plot_test_data(self):
        if self.default_parameter_test:
            self.__remove_default_parameter(self.test_data)
            self.default_parameter_test = False

        for point in self.test_data:
            if point[2]:
                plt.plot(point[0], point[1], marker="o", markersize=10,
                          markerfacecolor="blue", markeredgecolor="blue")
            else:
                plt.plot(point[0], point[1], marker="x", markersize=10,
                          markerfacecolor="blue", markeredgecolor="blue")
                
    def logistic_regression(self, epochs = 15, learning_rate = 0.01):
        if not self.default_parameter_train:
            self.__add_default_parameter(self.training_data)
            self.default_parameter_train = True

        len_td = len(self.training_data)

        self.costs = []
        self.costs.append(self.__compute_quadratic_cost(self.training_data))

        nabla_thetas = np.zeros((1,3))
        
        for epoch in range(epochs):
            #Here we calculate the partial derivatives of the log-likelihood with respect to each parameter x, y and z
            nabla_thetas[0][0] = sum([ (prediction - self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-x) 
                                for x,y,z,prediction in self.training_data])
            nabla_thetas[0][1] = sum([ (prediction - self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-y) 
                                for x,y,z,prediction in self.training_data])
            nabla_thetas[0][2] = sum([ (prediction - self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z]))))*(-z) 
                                for x,y,z,prediction in self.training_data])
           

            #Here we update the gradient descent algorithm with the partial derivatives calculated above
            self.thetas = [t-learning_rate*n_t/len_td for t,n_t in zip(self.thetas, nabla_thetas)]
            self.costs.append(self.__compute_quadratic_cost(self.training_data))

            
    def __compute_quadratic_cost(self, data):
        if not self.default_parameter_train:
            self.__add_default_parameter(self.training_data)
            self.default_parameter_train = True

        average_cost = 0
        for x,y,z,correct_value in data:
            output = self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z])))
            cost = (output - correct_value) ** 2
            average_cost += cost
        
        return average_cost / (len(data) * 2)


    def __sigmoid(self, input):
        return 1.0 / ( 1.0 + np.exp(-input) )
    
    def __add_default_parameter(self, data):
        for item in data:
            item.insert(2,1)

    def __remove_default_parameter(self, data):
        for item in data:
            item.pop(2)

    def predict_test_data(self):
        """ Method used to evaluate the model. """
        if not self.default_parameter_test:
            self.__add_default_parameter(self.test_data)
            self.default_parameter_test = True


        guessed_right = 0
        for point in self.test_data:

            x = point[0]
            y = point[1]
            z = point[2]
            correct_value = point[3]

            prediction = self.__sigmoid(np.matmul(self.thetas,np.array([x,y,z])))
            d = prediction
            if prediction < 0.5:
                prediction = 0
            else:
                prediction = 1
            if correct_value == prediction:
                guessed_right +=1

        print("Accuracy on test data: {}% with logistic regression.".format(guessed_right / len(self.test_data) * 100))


    def plot_cost_function(self):
        plt.figure()
        plt.plot(self.costs)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")
        plt.title("Logistic regression loss")
    