import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities1 = np.genfromtxt("hw06_predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("hw06_predicted_probabilities2.csv", delimiter = ",")

# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated thresholds of size (N + 1,)
def calculate_threholds(predicted_probabilities):

    intermediate_values = np.concatenate(([0.0], np.sort(np.unique(predicted_probabilities)), [1.0]))  
    
    thresholds = np.asarray([(intermediate_values[i] + intermediate_values[i + 1]) / 2 for i in range(len(intermediate_values) - 1)])


    return thresholds


thresholds1 = calculate_threholds(predicted_probabilities1)
print(thresholds1)

thresholds2 = calculate_threholds(predicted_probabilities2)
print(thresholds2)

# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the thresholds of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    
    # I changed the way of calculating thresholds in order to match with the given output
    sorted_predictions = np.sort(predicted_probabilities)
    
    thresholds = np.concatenate(([-np.inf], sorted_predictions, [np.inf]))

    total_samples = len(true_labels)
    fp_rates = np.zeros(len(thresholds))
    tp_rates = np.zeros(len(thresholds))
    positive_count = np.sum(true_labels == 1)
    negative_count = np.sum(true_labels == -1)


    for idx, threshold in enumerate(thresholds):

        predictions = np.where(predicted_probabilities > threshold, 1, -1)
        true_positives = np.sum((predictions == 1) & (true_labels == 1))
        false_positives = np.sum((predictions == 1) & (true_labels == -1))
        fp_rates[idx] = false_positives / negative_count if negative_count > 0 else 0
        tp_rates[idx] = true_positives / positive_count if positive_count > 0 else 0
        
    return fp_rates, tp_rates


fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, thresholds1)
print(fp_rates1[495:505])
print(tp_rates1[495:505])

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, thresholds2)
print(fp_rates2[495:505])
print(tp_rates2[495:505])

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()
plt.show()
fig.savefig("hw06_roc_curves.pdf", bbox_inches = "tight")

# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
   
    area = 0.0
    for index in range(len(fp_rates) - 1):
        
        delta_fp = fp_rates[index] - fp_rates[index + 1]  
        
        avg_tp = (tp_rates[index + 1] + tp_rates[index]) / 2 
        
        area += delta_fp * avg_tp

    auroc = area
    
    return auroc


auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))