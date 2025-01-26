import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-300))



X_train = np.genfromtxt("pendigits_sta16_train.csv", delimiter = ",", dtype = int)
y_train = np.genfromtxt("pendigits_label_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("pendigits_sta16_test.csv", delimiter = ",", dtype = int)
y_test = np.genfromtxt("pendigits_label_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    
    # In order to calculate the number of counts of each class, separately
    unique_class, count = np.unique(y, return_counts=True)
    class_priors = count / len(y)

    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)


# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    
    classes = np.unique(y)
    K = len(classes)
    D = X.shape[1]
    
    # Initializing the zero matrix KxD
    P = np.zeros((K, D))
    
    # Success prob
    for i, c in enumerate(classes):
        
        X_of_c = X[y == c] # Samples of c

        P[i, :] = X_of_c.mean(axis=0)

    return(P)

P = estimate_success_probabilities(X_train, y_train)
print(P)


# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    
    K = P.shape[0]  
    N, D = X.shape  # Number of data points and features
    
    score_values = np.zeros((N, K))

    for c in range(K):
        log_p = safelog(class_priors[c])
        
        # safelog usage
        log_p_on = safelog(P[c])         
        log_p_off = safelog(1 - P[c])    
        

        score_values[:, c] = np.sum(X * log_p_on + (1 - X) * log_p_off, axis=1) + log_p

    return score_values

scores_train = calculate_score_values(X_train, P, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
print(scores_test)


# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    
    y_prediction = np.argmax(scores, axis=1)
    
    K = scores.shape[1]
    
    
    # Initializing a K x K confusion matrix
    confusion_matrix = np.zeros((K, K), dtype=int)  
    
    # Subtract 1 for zero-indexing
    y_truth = y_truth - 1  # I assumed labels in y_truth are 1-indexed

    for true_label, prediction_label in zip(y_truth, y_prediction):
        confusion_matrix[true_label, prediction_label] += 1
        
    confusion_matrix = confusion_matrix.T
    
    return confusion_matrix

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))
confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
