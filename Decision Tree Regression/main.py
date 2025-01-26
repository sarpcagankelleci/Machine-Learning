import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.5
maximum_value = 5.1
step_size = 0.001
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    need_split = {}
    is_terminal = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    # your implementation starts below 
    
    num_features = X_train.shape[1] 
    num_samples = len(y_train)  

    node_indices[1] = np.arange(num_samples)
    is_terminal[1] = False
    need_split[1] = True

    while True:

        nodes_to_split = [node for node, split in need_split.items() if split]
        if not nodes_to_split:  
            break  
    
        for current_node in nodes_to_split:

            current_indices = node_indices[current_node]
            need_split[current_node] = False  # Mark this node as "done" (processed)
    

            # If this node has very few data points (<= P) or all targets are identical (variance=0)
            if len(current_indices) <= P or np.var(y_train[current_indices]) == 0:

                is_terminal[current_node] = True
                node_means[current_node] = np.mean(y_train[current_indices])
                # I decided to stop splitting here because the node is "pure" or too small
                continue  
    

            lowest_score, optimal_split, optimal_feature = float('inf'), None, None
    
            # I am going to check all features one by one to find the best split
            for feature_idx in range(num_features):  

                sorted_values = np.sort(np.unique(X_train[current_indices, feature_idx]))
            
                candidate_splits = (sorted_values[:-1] + sorted_values[1:]) / 2

                for threshold in candidate_splits:
                    
                    # I split the data into two groups: left and right
                    left_group = current_indices[X_train[current_indices, feature_idx] > threshold]
                    right_group = current_indices[X_train[current_indices, feature_idx] <= threshold]
    
                    # If one of the groups is empty, I skip this split 
                    if len(left_group) == 0 or len(right_group) == 0:
                        continue
    

                    weighted_score = (len(left_group) * np.var(y_train[left_group]) + 
                                      len(right_group) * np.var(y_train[right_group])) / len(current_indices)
    
                    # If this split is better than the previous best split, I update my records
                    if weighted_score < lowest_score:
                        lowest_score, optimal_split, optimal_feature = weighted_score, threshold, feature_idx
    

            node_features[current_node] = optimal_feature 
            node_splits[current_node] = optimal_split  
    
            left_group = current_indices[X_train[current_indices, optimal_feature] > optimal_split]
            right_group = current_indices[X_train[current_indices, optimal_feature] <= optimal_split]
    
            # I now create two child nodes for this split
            node_indices[2 * current_node] = left_group  
            node_indices[2 * current_node + 1] = right_group 
    

            is_terminal[2 * current_node] = False
            is_terminal[2 * current_node + 1] = False
            need_split[2 * current_node] = True
            need_split[2 * current_node + 1] = True


    # your implementation ends above
    return is_terminal, node_features, node_splits, node_means


# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    
    num_queries = X_query.shape[0] 
    y_hat = np.repeat(0.0, num_queries)  

    for sample_idx in range(num_queries):
        current_node = 1  
        
        while not is_terminal[current_node]:
            
            
            split_feature = node_features[current_node]
            split_value = node_splits[current_node]
            
            
            if X_query[sample_idx, split_feature] > split_value:
                current_node = 2 * current_node  
                
            else:
                current_node = 2 * current_node + 1  

        y_hat[sample_idx] = node_means[current_node]
    
    # your implementation ends above
    return(y_hat)


# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    
    end_nodes = [n for n, term in is_terminal.items() if term]
    end_nodes = sorted(end_nodes)

    def trace_path(leaf_id):
        
        conditions = []
        node_id = leaf_id
        while node_id > 1:
            parent_id = node_id // 2
            go_left = (node_id % 2 == 0)
            parent_split = node_splits.get(parent_id, None)
            
            if parent_split is not None:
                
                
                if go_left:
                    conditions.append("x1 <= {:.2f}".format(parent_split))
                else:
                    conditions.append("x1 > {:.2f}".format(parent_split))
            node_id = parent_id
        return conditions[::-1]

    for tid in end_nodes:
        rule_list = trace_path(tid)
        val = node_means[tid]
        print("Node {:02d}: {} => {}".format(tid, rule_list, val))

    # your implementation ends above

P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)
