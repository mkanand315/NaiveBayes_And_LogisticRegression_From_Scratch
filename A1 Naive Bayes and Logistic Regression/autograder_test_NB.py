import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as metrics
from sklearn.utils import shuffle


def load_data(path, header):
    df = pd.read_csv(path, header=header, delimiter="\t")
    return df


def prediction(xtest, prior_prob_pos, prior_prob_neg, count_pos, count_neg):

	ypred = np.zeros(xtest.shape[0])
	log_likelihood_pos = 0
	log_likelihood_neg = 0
	total_pos = 0
	total_neg = 0

	log_likelihood_pos = np.matmul(xtest,np.log(count_pos)) + np.matmul((np.ones(xtest.shape)-xtest),np.log(np.ones(count_pos.shape) - count_pos))

	log_likelihood_neg = np.matmul(xtest,np.log(count_neg)) + np.matmul((np.ones(xtest.shape)-xtest),np.log(np.ones(count_neg.shape) - count_neg))

	total_pos = np.log(prior_prob_pos)*np.ones(log_likelihood_pos.shape) + log_likelihood_pos
	total_neg = np.log(prior_prob_neg)*np.ones(log_likelihood_neg.shape) + log_likelihood_neg


	for i in range(total_pos.shape[0]):

		if total_pos[i] >= total_neg[i]:

			ypred[i] = 1

		elif total_pos[i] < total_neg[i]:

			ypred[i] = 0

	return ypred


def accuracy(ytest, ypred):
		
	return 100*np.mean(ytest == ypred)



if __name__ == "__main__":
    # load the data from the file
	prior_data = np.array(load_data("class_priors.tsv", None))
	count_pos = np.array(load_data("positive_feature_likelihoods.tsv",None))
	count_neg = np.array(load_data("negative_feature_likelihoods.tsv",None))
	
	prior_prob_pos = prior_data[0][0]
	prior_prob_neg = prior_data[1][0]


	val = load_data("validation_dataset.tsv",None)
	

	xval = np.array(val.iloc[:,:-1])
	yval = np.array(val.iloc[:,-1])


	ypred = prediction(xval, prior_prob_pos, prior_prob_neg, count_pos, count_neg)


	accuracy1 = accuracy(ypred, yval)

	print("Accuracy of Naive Bayes model in % is: ", accuracy1)

