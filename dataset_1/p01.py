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
print(data.head())


# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)


# Create subplots for bar plots and density estimates
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))


# Define the list of features you want to plot (excluding 'charges' which is the target variable)
features_to_plot = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']
'''
# Set the style for Seaborn plots
sns.set(style="whitegrid")

for feature in features_to_plot:
    plt.figure(figsize=(12, 5))

    # Create a subplot for the bar plot
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x=feature, bins=30, kde=False)
    plt.title(f'{feature} - Bar Plot')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Create a subplot for the density estimate plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=data, x=feature, fill=True)
    plt.title(f'{feature} - Density Estimate')
    plt.xlabel(feature)
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()
'''

'''
# Observing the scatter plots between pairs of features.
sns.pairplot(data=data[features_to_plot])
plt.show()
'''


# Create scatter plots between 'charges' and each feature
for feature in features_to_plot:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x=feature, y='charges')
    plt.title(f'Scatter Plot: {feature} vs. charges')
    plt.xlabel(feature)
    plt.ylabel('charges')
    plt.show()
