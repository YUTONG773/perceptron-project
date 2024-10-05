#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from collections import Counter

# Read txt file, using space as delimiter
data = pd.read_csv('diabetes_scale.txt', sep=' ', header=None)

# Save data to CSV
data.to_csv('diabetes_scale.csv', index=False)

# Read CSV file
data = pd.read_csv('diabetes_scale.csv', header=None)

# Function to extract feature values
def split_feature_value(cell):
    try:
        return float(cell.split(':')[1])
    except:
        return cell

# Process features from the second column onward
for col in range(1, data.shape[1]):
    data[col] = data[col].apply(split_feature_value)

# Show first few rows of the data
print(data.head())

# Labels in the first column, features in the rest
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Ensure data type is float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Check for columns with all NaN values
nan_columns = np.isnan(X_train).all(axis=0)
print(f"All NaN column indices: {np.where(nan_columns)}")

# Remove columns with all NaN values
X_train = X_train[:, ~nan_columns]
X_test = X_test[:, ~nan_columns]

# Replace NaN values with zero
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test, nan=0)

# Check for columns with zero variance
variances = np.var(X_train, axis=0)
print(f"Zero variance column indices: {np.where(variances == 0)}")

# Remove zero variance columns
X_train = X_train[:, variances != 0]
X_test = X_test[:, variances != 0]

# Feature selection: remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

# Perform undersampling to balance classes
# assuming -1 is the majority class
counter = Counter(y_train)
print(f"Original class distribution: {counter}")

# Get the number of minority class samples to match
minority_class = min(counter, key=counter.get)
minority_class_count = counter[minority_class]

# Randomly sample from the majority class
majority_class_indices = np.where(y_train == -1)[0]
np.random.shuffle(majority_class_indices)
keep_majority_indices = majority_class_indices[:minority_class_count]

# Get indices of the minority class
minority_class_indices = np.where(y_train == 1)[0]

# Combine majority and minority class indices
new_indices = np.concatenate([keep_majority_indices, minority_class_indices])

# Update training data
X_train = X_train[new_indices]
y_train = y_train[new_indices]

# Check new class distribution
counter = Counter(y_train)
print(f"New class distribution after undersampling: {counter}")

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add polynomial features
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Define Perceptron model
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=200):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1] + 1)
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Compute weighted sum"""
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """Return predicted label"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Initialize and train Perceptron
ppn = Perceptron(learning_rate=0.01, n_iter=200)
ppn.fit(X_train_poly, y_train)

# Predict on test set
y_pred = ppn.predict(X_test_poly)

# Compute and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.2f}')

# Visualize number of updates
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Number of updates')
plt.title('Perceptron - Error reduction over iterations')
plt.show()


# In[2]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[3]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, zero_division=1))

