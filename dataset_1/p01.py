import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("insurance.csv")

# Display the first few rows to get an overview of the data
print(data.head())


# Convert 'sex' to numeric values
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Convert 'smoker' to numeric values
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Summary statistics of the numeric columns
print(data.describe())


# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)


# Create subplots for bar plots and density estimates
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# List of features to visualize
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'charges', 'region_northwest', 'region_southeast', 'region_southwest']

# Plot histograms for each feature
for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    if feature == 'charges':
        # Use a histogram for the 'charges' feature
        sns.histplot(data[feature], ax=axes[row, col], kde=True)
    else:
        # Use a bar plot for categorical features, density estimate for numeric features
        if data[feature].dtype == 'float64' or data[feature].dtype == 'int64':
            sns.histplot(data[feature], ax=axes[row, col], kde=True)
        else:
            sns.countplot(data[feature], ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

# Remove empty subplots if necessary
if len(features) < 9:
    for i in range(len(features), 9):
        fig.delaxes(axes[i // 3, i % 3])

plt.tight_layout()
plt.show()


