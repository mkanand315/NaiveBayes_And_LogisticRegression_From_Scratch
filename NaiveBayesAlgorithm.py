import numpy

class BernoulliNaiveBayesModel:

    """
    BernoulliNaiveBayesModel class initialization 
    for the attributes
    """
    def __init__(self):
        self.class_priors = None                       # The computed Class Priors by the Maximum Likelihood Expression, P(y)
        self.distinct_feature_likelihoods = None       # The computed Feature Likelihoods by the Maximum Likelihood Expression, P(x_i | y)
        self.feature_class_counts =  None              # The count statistics for each of the classes for the features

    """
    The fit function takes the training data as input (X and y). 
    It is used for training.

    X is all of the features  (n*m)
    y is the corresponding label vector (n*1)

    1) Calculate the Class Priors  -- Create sub-arrays by filtering input array X.
    2) Calculate the feature likelihoods -- Loop over sub-arrays in filtered X input. Sum the columns of each sub-array

    Citations:
    - https://www.w3schools.com/python/trypython.asp?filename=demo_numpy_shape
    - https://geoffruddock.com/naive-bayes-from-scratch-with-numpy/
    """
    def fit(self, X, y): 
        # Total number of data points (rows in input array X).                                          
        num_of_datapoints = X.shape[0]

        # Filtering X by class of the features, k, where k = {0 , 1}.
        X_filtered_by_class =  numpy.array([X[y == k] for k in numpy.unique(y)])

        # This will divide the input array into sub-arrays, one for each class k. Note, k=0 comes first in resulting list
        X_filtered_by_class = numpy.array([X[y == k] for k in numpy.unique(y)]) 

        # Calculating the class priors.
        self.class_priors = numpy.array([len(X_class_of_k) / num_of_datapoints for X_class_of_k in X_filtered_by_class])                                         

        # Calculating the count statistics. This iterates over the filtered X input 
        # and counts the number of times the jth feature appears for each of the class k's, ie. x[j] = 1
        self.feature_class_counts = numpy.array([class_k_array.sum(axis=0) for class_k_array in X_filtered_by_class])

        # Calculating P(x_i|y) based on the count statistics for each of the features. 
        # Counting the number of times each feature appears for the specified class using the row sum.
        # Re-shaping as (-1,1) gives each entry as it's own 
        self.distinct_feature_likelihoods = self.feature_class_counts / self.feature_class_counts.sum(axis=1).reshape(-1, 1)
        
        return self



    """
    The predict function will take a matrix with n training examples and will
    output n predictions for the estimated y values using the stored model parameters.

    1) loop through the validation features (rows of X) and compute the log odds expression
    2) Compare log odds ratio value against natural decision boundry
    3) Repeat for next row of X

    Citations: 
    - https://thepythonguru.com/python-builtin-functions/max/
    
    """
    def predict(self, X):
        for i in range(len(X)):
            tempComputation = {}
            tempSubArray = X[i]             # Represents 1 feature vector from the n that were input
            for k in range (0,2):           # for k= {0,1} since we only have two binary classes for the labels
                # Here, I am using the log odds ratio expression to compute P(y=1|X) and P(y=0|X). I am computing the log class prior and log P(y|X)
                tempComputation[k] = numpy.log(self.class_priors[k]) + numpy.sum(numpy.log(self.distinct_feature_likelihoods[k][tempSubArray == 1])) + \
                                   numpy.sum(numpy.log(1- self.distinct_feature_likelihoods[k][tempSubArray == 0]))
            # Using the natural decision boundry of 0 to predict the label. Checking if P(y=1|X) / P(y=0|X) > 0 or not.
            if ((tempComputation.get(1) / tempComputation.get(0)) > 0):
                print(1)
            else:
                print(0)

    
    """
    The save_parameters will save the model parameters after they have been fitted. 
    I am saving them to the corresponding tsv files as per the requirements.
    """
    def save_parameters(self):
        numpy.savetxt("class_priors.tsv", numpy.flip(self.class_priors), delimiter="\t")
            
        for i in range(len(self.distinct_feature_likelihoods)):
            if i == 0:
                numpy.savetxt("negative_feature_likelihoods.tsv", self.distinct_feature_likelihoods[i] , delimiter="\t")
            elif i == 1:
                numpy.savetxt("positive_feature_likelihoods.tsv", self.distinct_feature_likelihoods[i] , delimiter="\t")
            