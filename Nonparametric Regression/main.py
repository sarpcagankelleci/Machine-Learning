import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw03_data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start = minimum_value, stop = maximum_value, step = 0.001)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    
    N = x_query.shape[0]
    y_hat = np.zeros(N)
    
    for query_idx, query_point in enumerate(x_query):

        bin_indices = np.where((left_borders <= query_point) & (query_point <= right_borders))[0]
        if bin_indices.size == 0:
            continue
        
        current_bin_index = bin_indices[0]
        current_left_edge = left_borders[current_bin_index]
        current_right_edge = right_borders[current_bin_index]
        
        numerator = np.sum(np.asarray([
            y_val for x_val, y_val in zip(x_train, y_train)
            if current_left_edge < x_val <= current_right_edge
        ]))
        denominator = np.sum(np.asarray([
            1 for x_val in x_train
            if current_left_edge < x_val <= current_right_edge
        ]))
        
        y_hat[query_idx] = numerator / denominator if denominator > 0 else 0
    
    # your implementation ends above
    return(y_hat)
    
    
bin_width = 0.35
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below   
    y_hat = np.zeros_like(x_query)
    
    for query_idx, query_point in enumerate(x_query):

        kernel_weights = np.abs((query_point - x_train) / bin_width) < 0.5
        
        weighted_sum = np.sum(kernel_weights * y_train)
        
        total_kernel_weight = np.sum(kernel_weights)
        
        y_hat[query_idx] = weighted_sum / total_kernel_weight if total_kernel_weight > 0 else 0
        
    # your implementation ends above
    return(y_hat)



bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below   
    y_hat = np.zeros_like(x_query)
    
    for query_idx, query_point in enumerate(x_query):

        gaussian = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((query_point - x_train) / bin_width) ** 2)
        
        weighted_sum = np.sum(gaussian * y_train)
        
        total_gaussian_weight = np.sum(gaussian)
        
        y_hat[query_idx] = weighted_sum / total_gaussian_weight if total_gaussian_weight > 0 else 0

    # your implementation ends above
    return(y_hat)


bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
