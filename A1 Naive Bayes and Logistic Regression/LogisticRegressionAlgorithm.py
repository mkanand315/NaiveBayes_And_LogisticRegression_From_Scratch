import numpy
from numpy import linalg as LNG 

class LogisticRegressionModel:

    def __init__(self):
        self.weights = None                       # The weights the model will be learning
    
    ### HELPER FUNCTIONS: Sigmoid, Compute Gradient ###

    """
    The sigmoid functiton returns the result of the logistic function as in Chapter 6.

    func_input is the dot product of w(transpose)X; func_input = numpy.dot(X, w)
    """
    def sigmoid(self, func_input):
        return 1 / (1 + numpy.exp(-func_input))

    """
    The compute_gradient functiton returns the result of evaluating the gradient as in chapter 8
    """
    def compute_gradient(self, featuresX, predictions_y, y):
        return numpy.dot(featuresX.T, (predictions_y - y)) / y.size

    
    ### Fit, Predict and Save Parameter Functions Below ###

    """
    The fit function takes the training data as input (X and y). 
    It is used for training.

    Hyperparameters:
    - X is all of the features  (n*m)
    - y is the corresponding label vector (n*1)
    - max_iters = 5000; maximum number of gradient descent iterations
    - lr = 0.01; learning rate
    - tol = 0.0005; tolerance (epsilon) for stopping

    1) Initialize weights vector, with taking to account bias term as first term of X
    2) Compute the dot product of the features and weights vector (initialized to zero)
    3) Iterate max_iter times to determine the optimal vector using gradient decent

    Citations: 
    - https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    - https://numpy.org/doc/stable/reference/generated/numpy.ones.html

    Note:
    Using tol of 0.0005 gives accuracy of 83.36
    Using tol of 0.0001 gives accuracy of 84.6
    """

    def fit(self, X, y, max_iters, lr, tol):
        # Creating the bias vector and concatenating it to each row in X 
        bias_term_vector = numpy.ones((X.shape[0], 1))
        featuresX = numpy.concatenate((bias_term_vector, X), axis=1)

        # Initializing the weights vector as 0 (initial guess)
        self.weights = numpy.zeros(featuresX.shape[1])
        weights_next = numpy.zeros(featuresX.shape[1])

        # Loop through and determine optimal weight vector with gradient descent
        for i in range(max_iters):
            # Computing the required values to perform gradient descent and update the weight vector
            input_for_sigmoid = numpy.dot(featuresX, self.weights)
            predictions_y = self.sigmoid(input_for_sigmoid)
            gradient = self.compute_gradient(featuresX, predictions_y, y)
            weights_next = self.weights - (lr * gradient)
            
            # Stopping criterion as in equation 8.2 of the notes
            if (LNG.norm(weights_next - self.weights) < tol):
                self.weights = weights_next
                break
            else:
                self.weights = weights_next
            

    """
    The predict function will take a matrix with n training examples and will
    output n predictions for the estimated y values using the stored model parameters.

    1) Convert X so that it includes a bias term for each row
    2) Pass X.T(weights) to sigmoid function
    3) Use a threshold of 0.5 for predicting the label (0 or 1).
    """
    def predict(self, X):
        # Creating the bias vector and concatenating it to each row in X 
        bias_term_vector = numpy.ones((X.shape[0], 1))
        featuresX = numpy.concatenate((bias_term_vector, X), axis=1)

        # Looping through each row of X to predict a label for the feature vector
        for i in range(len(X)):
            predicted_probability = self.sigmoid(numpy.dot(featuresX, self.weights))
            if (predicted_probability.mean() >= 0.5):
                print(1)
            else:
                print(0)
        
    """
    The save_parameters will save the model parameters after they have been fitted. 
    I am saving them to the corresponding tsv files as per the requirements.
    """
    def save_parameters(self):
        numpy.savetxt("weights.tsv", self.weights, delimiter="\t")