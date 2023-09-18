"""
Script that implements the Gaussian Discriminant Analysis model and compares its results with a logistic regression
model. Plots the associated learned distributions for each class, alogside the decision boundaries for each class
"""
import numpy as np
import matplotlib.pyplot as plt
import logisit_model

class GaussianDiscriminantAnalysis:
    """
    Initialize data and everything that is needed for the class to work.
    Different values can be modifiet here for experimentation, but mainly the samples per class are the ones
    that matter most. If the overall number of samples if modified, considering modifiyng also the split parameters
    for this and the logistic regression model.
    """
    def __init__(self, split = 0.9):
        samples_per_class0 = 200
        samples_per_class1 = 200

        split = int((samples_per_class0 + samples_per_class1) * split)

        mean_class_0 = np.array([-1.5,-1.5])
        covariance_class_0 = np.array([[1, 0.], [0., 1]])
        self.features_class_0 = np.random.multivariate_normal(mean_class_0,covariance_class_0,samples_per_class0)

        mean_class_1 = np.array([1.0,2])
        covariance_class_1 = np.array([[1, -0.5], [-0.5, 1]])
        self.features_class_1 = np.random.multivariate_normal(mean_class_1,covariance_class_1,samples_per_class1)

        features_class_0_zeroed = np.hstack((self.features_class_0, np.zeros((samples_per_class0, 1))))
        features_class_1_zeroed = np.hstack((self.features_class_1, np.ones((samples_per_class1, 1))))

        self.training_data = np.vstack((features_class_0_zeroed,features_class_1_zeroed))
        np.random.shuffle(self.training_data)
        self.whole_training_data = [item for item in self.training_data] # Will be used by the other model
        self.test_data = self.training_data[split:]
        self.training_data = self.training_data[:split]

    def plot_test_points(self):
        for point in self.test_data:
            x = point[0]
            y = point[1]
            label = point[2]
            if label == 1.0:
                plt.scatter(x,y,marker="x" ,color = "orange")
            else:
                plt.scatter(x,y,marker="o" ,color = "red")

    def plot_data(self):
        plt.scatter(self.features_class_0[:,0],self.features_class_0[:,1],marker="o",color = "red", label = "Class 0")
        plt.scatter(self.features_class_1[:,0],self.features_class_1[:,1],marker="x" ,color = "orange", label = "Class 1")     
        plt.legend()

    def plot_gaussian_distribution(self, mean_vector, covariance_matrix):
        m = np.array([[mean_vector[0]],[mean_vector[1]]])  
        cov = covariance_matrix

        cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
        cov_det = np.linalg.det(cov)  # determinant of covariance matrix

        # Plotting
        x = np.linspace(-6, 6)
        y = np.linspace(-6, 6)
        X,Y = np.meshgrid(x,y)
        
        coefficient = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
        Z = coefficient * np.e ** (-0.5 * (cov_inv[0,0]*(X-m[0])**2 + (cov_inv[0,1] + cov_inv[1,0])*(X-m[0])*(Y-m[1]) + cov_inv[1,1]*(Y-m[1])**2))
        plt.contour(X,Y,Z,levels = 10, linewidths = 2)

    def estimate_parameters(self):
        """Estimates means and covariance matrix for both classes."""
        len_train_data = len(self.training_data)
        self.bernoulli_parameter = sum(self.training_data[:,2] == 1.0) / len_train_data
        
        miu01 = sum(self.training_data[i][0] for i in range(len_train_data) if self.training_data[i][2] == 0.0 )
        miu02 = sum(self.training_data[i][1] for i in range(len_train_data) if self.training_data[i][2] == 0.0 )

        self.miu_0 = np.array([miu01,miu02]) / sum(self.training_data[:,2] == 0.0)

        miu11 = sum(self.training_data[i][0] for i in range(len_train_data) if self.training_data[i][2] == 1.0 )
        miu12 = sum(self.training_data[i][1] for i in range(len_train_data) if self.training_data[i][2] == 1.0 )

        self.miu_1 = np.array([miu11,miu12]) / sum(self.training_data[:,2] == 1.0)

        unclassed_data = np.delete(self.training_data,2,1)

        self.cov_matrix = np.zeros((2,2))
        for i in range(len_train_data):
            if self.training_data[i][2] == 0.0:
                self.cov_matrix += np.outer(unclassed_data[i] - self.miu_0,unclassed_data[i]-self.miu_0)
            else:
                self.cov_matrix += np.outer(unclassed_data[i] - self.miu_1,unclassed_data[i]-self.miu_1)

        self.cov_matrix /= len_train_data

    def show_learned_gaussians(self):
        self.plot_gaussian_distribution(self.miu_0,self.cov_matrix)
        self.plot_gaussian_distribution(self.miu_1,self.cov_matrix)


    def predict_test_data(self):
        """Predicts new data using Bayses' theorem."""

        unclassed_data = np.delete(self.test_data,2,1)

        len_test_data = len(self.test_data)

        inv_cov = np.linalg.inv(self.cov_matrix)

        det_cov = np.linalg.det(self.cov_matrix)

        miu0 = np.array([[self.miu_0[0]],[self.miu_0[1]]])

        miu1 = np.array([[self.miu_1[0]],[self.miu_1[1]]])


        guessed_right = 0
        for i in range(len_test_data):
            brn1 = self.bernoulli_parameter
            brn0 = 1 - self.bernoulli_parameter
            exponent0 = ((unclassed_data[i]-miu0.transpose()) @ inv_cov @ (unclassed_data[i]-miu0.transpose()).transpose())[0][0]
            pxgiveny0 = (1/(2*np.pi*np.sqrt(det_cov)))*np.exp((-1/2)*(exponent0))

            exponent1 = ((unclassed_data[i]-miu1.transpose()) @ inv_cov @ (unclassed_data[i]-miu1.transpose()).transpose())[0][0]
            pxgiveny1 = (1/(2*np.pi*np.sqrt(det_cov)))*np.exp((-1/2)*(exponent1))

            confidence1 = (pxgiveny1*brn1) / (pxgiveny1*brn1 + pxgiveny0*brn0) # Relations from Bayes' theorem
            confidence0 = (pxgiveny0*brn0) / (pxgiveny1*brn1 + pxgiveny0*brn0)
    

            true_value = self.test_data[i][2]

            if confidence1 > confidence0 :
                guess = 1.0
            else:
                guess = 0.0
            
            if guess == true_value:
                guessed_right += 1
            else:
                # print(unclassed_data[i])
                # print(confidence0,confidence1)
                # print(true_value)
                # print("Next point")
                pass

        print("Accuracy on test data: {}% with Gaussian Discriminant Analysis.".format(100*guessed_right/len_test_data))


              

    def plot_decision_boundry(self):
        """Plots the decision boundary. See the readme file for more details on the maths."""
        miu0 = np.array([[self.miu_0[0]],[self.miu_0[1]]])
        miu1 = np.array([[self.miu_1[0]],[self.miu_1[1]]])

        inverse_covariance = np.linalg.inv(self.cov_matrix)

        c_term = -0.5 * (np.dot(np.dot(miu1.transpose(),inverse_covariance),miu1)[0][0]
                        - np.dot(np.dot(miu0.transpose(),inverse_covariance),miu0)[0][0])
  
        c_term += np.log(self.bernoulli_parameter/(1-self.bernoulli_parameter))


        delta_miu0 = miu1[0][0] - miu0[0][0]
        delta_miu1 = miu1[1][0] - miu0[1][0]

        x_axis = np.linspace(-6,6,10)
        boundry = (1.0 / (inverse_covariance[1][0]*delta_miu0 + inverse_covariance[1][1]*delta_miu1)) *(
            -x_axis*(inverse_covariance[0][0]*delta_miu0 + inverse_covariance[0][1]*delta_miu1) - c_term
        )

        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.plot(x_axis,boundry, color = "blue", label = "GDA decision boundary")
        plt.legend()
        



def convert_data(gda_data):
    """Covnerts data so the other model can use it."""
    lr_data = []
    for arr in gda_data:
        lr_data.append([arr[0],arr[1],int(arr[2])])

    return lr_data


if __name__ == "__main__":
    np.random.seed(28)
    gda = GaussianDiscriminantAnalysis()
    gda.plot_data()

    regression = logisit_model.Logistic_Regression(training_data=convert_data(gda.whole_training_data),
                                                                      split=0.9)
    
    gda.estimate_parameters()
    gda.show_learned_gaussians()
    gda.plot_decision_boundry()
    gda.predict_test_data()

    regression.logistic_regression(300,1) 
    regression.predict_test_data()

    # Plotting decision boundry for logistic regression

    x_axis = np.linspace(-6,6,10) 
    plt.xlim(-4,6)
    plt.ylim(-4,4)
    plt.plot(x_axis,-(1.0/regression.thetas[0][1])*(regression.thetas[0][0]*x_axis+regression.thetas[0][2]),
             color = "green", label = "LR decision boundary")
    plt.legend()


    # Plotting test data separately to check the closely view the performance of the 2 models
    plt.figure()
    gda.plot_test_points()
    plt.plot(x_axis,-(1.0/regression.thetas[0][1])*(regression.thetas[0][0]*x_axis+regression.thetas[0][2]),
             color = "green", label = "LR decision boundary")
    gda.plot_decision_boundry()



