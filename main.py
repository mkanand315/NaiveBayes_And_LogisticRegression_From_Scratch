import numpy
from NaiveBayesAlgorithm import BernoulliNaiveBayesModel
from LogisticRegressionAlgorithm import LogisticRegressionModel

### Loading and Parsing Train and Validation Data ###

data = numpy.loadtxt("train_dataset.tsv", delimiter="\t")
X_validation_data = numpy.loadtxt("validation_dataset.tsv", delimiter="\t")

X = data[:,:-1]
y = data[:,-1]
X_validation_features = X_validation_data[:,:-1]

### Question 1: Naive Bayes ###

NB = BernoulliNaiveBayesModel()

NB.fit(X,y)

NB.save_parameters()

NB.predict(X_validation_features)

### Question 2: Logistic Regression ###

LR = LogisticRegressionModel()

LR.fit(X, y, max_iters=5000, lr=0.01, tol=0.0005)

LR.save_parameters()

LR.predict(X_validation_features)