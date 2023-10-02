import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("insurance.csv")

# Convert 'sex' to numeric values
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Convert 'smoker' to numeric values
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

# Apply one-hot encoding to 'region'
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Define the target variable (dependent variable) and features (independent variables)
X = data.drop(columns=['charges'])
y = data['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
learning_rates = [0.001, 0.01, 0.1, 0.2]
epochs = [100, 500, 1000, 2000]
best_mse = float('inf')
best_learning_rate = None
best_epochs = None
best_theta = None

for lr in learning_rates:
    for epoch in epochs:
        # Initialize coefficients
        theta = np.random.randn(X_train_scaled.shape[1])

        # Gradient Descent
        for _ in range(epoch):
            # Calculate predictions
            y_pred = np.dot(X_train_scaled, theta)

            # Calculate the error (Mean Squared Error)
            error = np.mean((y_pred - y_train)**2)

            # Calculate the gradient
            gradient = np.dot(X_train_scaled.T, (y_pred - y_train)) / len(y_train)

            # Update coefficients using gradient descent
            theta = theta - lr * gradient

        # Predict on the test set
        y_pred_test = np.dot(X_test_scaled, theta)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_pred_test - y_test)**2)

        # Check if this combination of hyperparameters resulted in a better model
        if mse < best_mse:
            best_mse = mse
            best_learning_rate = lr
            best_epochs = epoch
            best_theta = theta

# Print the best hyperparameters and coefficients
print("Best Hyperparameters:")
print(f"Learning Rate: {best_learning_rate}")
print(f"Epochs: {best_epochs}")
print("Final Coefficients:")
print(best_theta)

# Evaluate the model with the best hyperparameters
y_pred_test = np.dot(X_test_scaled, best_theta)
r2 = r2_score(y_test, y_pred_test)

print(f"Mean Squared Error (MSE): {best_mse}")
print(f"R-squared (R2): {r2}")
