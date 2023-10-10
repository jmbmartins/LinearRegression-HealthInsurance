import random

import numpy as np
import pandas as pd
import sns

# Load the dataset
data = pd.read_csv("insurance.csv")

# Extract features and target variable
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Convert categorical features into one-hot encoding
X_categorical = pd.get_dummies(data[categorical_features], drop_first=True)
X_numerical = data[numerical_features]

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)


'''
# Normalize numerical features (optional but recommended for gradient descent)
X_numerical = (X_numerical - X_numerical.mean()) / X_numerical.std()
'''


X = pd.concat([X_categorical, X_numerical], axis=1)
y = data['charges']

# Number of iterations and learning rate
num_iterations = 1000
learning_rate = 0.0001


# Define the hypothesis function
def h(theta, x):
    return sum(theta[i] * x[i] for i in range(len(theta)))


# Define the cost function
def cost_function(theta, X, y):
    m = len(X)
    error = 0
    for i in range(m):
        error += (h(theta, X.iloc[i]) - y.iloc[i]) ** 2
    return error / (2 * m)


# Gradient Descent
def gradient_descent(theta, X, y, lr):
    m = len(X)
    for iteration in range(num_iterations):
        errors = [h(theta, X.iloc[i]) - y.iloc[i] for i in range(m)]
        for j in range(len(theta)):
            gradient = sum(errors[i] * X.iloc[i][j] for i in range(m))
            theta[j] -= (lr * gradient) / m
        cost = cost_function(theta, X, y)
        print("Iteration", iteration + 1, "Cost:", cost)
    return theta


# Initialize theta with random values
def initialize_theta(num_features):
    return [random.uniform(-1, 1) for _ in range(num_features)]


'''
# GRADIENT DESCENT TEST

# Add a bias term (intercept) to the features
X.insert(0, 'bias', 1)

# Initialize theta with random values
initial_theta = initialize_theta(len(X.columns))

# Perform gradient descent
final_theta = gradient_descent(initial_theta, X, y, learning_rate)

print("Final theta:", final_theta)
'''


# Function to perform k-fold cross-validation
def k_fold_cross_validation(X, y, k, learning_rate):
    n = len(X)
    fold_size = n // k
    mse_scores = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = pd.concat([X[:start], X[end:]])
        y_train = pd.concat([y[:start], y[end:]])

        initial_theta = initialize_theta(len(X_train.columns))
        final_theta = gradient_descent(initial_theta, X_train, y_train, learning_rate)

        y_pred = [h(final_theta, x) for _, x in X_test.iterrows()]
        mse = sum((y_pred[i] - y_test.iloc[i]) ** 2 for i in range(len(y_test))) / len(y_test)
        mse_scores.append(mse)

    return mse_scores



# Perform k-fold cross-validation TEST
k = 5  # You can change the number of folds as needed
mse_scores = k_fold_cross_validation(X, y, k, learning_rate)

# Calculate and print the average MSE
average_mse = np.mean(mse_scores)
print("Average Mean Squared Error (MSE) over", k, "folds:", average_mse)


'''
# Analyze the differences in performance between the models obtained for the different folds
for i, mse in enumerate(mse_scores):
    print(f"Fold {i + 1} MSE: {mse}")
    
# Calculate statistics
min_mse = np.min(mse_scores)
max_mse = np.max(mse_scores)
range_mse = max_mse - min_mse
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

# Print statistics
print(f"Minimum MSE: {min_mse}")
print(f"Maximum MSE: {max_mse}")
print(f"Range of MSE: {range_mse}")
print(f"Mean MSE: {mean_mse}")
print(f"Standard Deviation of MSE: {std_mse}")

'''
