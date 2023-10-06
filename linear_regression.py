import pandas as pd

# Load the dataset
data = pd.read_csv("insurance.csv")

# Extract features and target variable
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Convert categorical features into one-hot encoding
X_categorical = pd.get_dummies(data[categorical_features], drop_first=True)
X_numerical = data[numerical_features]
X = pd.concat([X_categorical, X_numerical], axis=1)
y = data['charges']

# Normalize numerical features (optional but recommended for gradient descent)
X_numerical = (X_numerical - X_numerical.mean()) / X_numerical.std()

# Initialize theta with zeros
initial_theta = [0] * (len(X.columns) + 1)  # One extra for bias term

# Number of iterations and learning rate
num_iterations = 1000
learning_rate = 0.0001

# Define the hypothesis function for multiple features
def h(theta, x):
    return sum(theta[i] * x[i] for i in range(len(theta)))

# Define the cost function for multiple features
def cost_function(theta, X, y):
    m = len(X)
    error = 0
    for i in range(m):
        error += (h(theta, X.iloc[i]) - y.iloc[i]) ** 2
    return error / (2 * m)

# Gradient Descent for multiple features
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

# Add a bias term (intercept) to the features
X.insert(0, 'bias', 1)

# Perform gradient descent
final_theta = gradient_descent(initial_theta, X, y, learning_rate)

print("Final theta:", final_theta)
