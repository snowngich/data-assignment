import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Data Loading and Cleaning
# Load the Iris dataset from sklearn
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
iris_df = pd.DataFrame(data.data, columns=data.feature_names)
iris_df['species'] = data.target

# Map target numbers to species names
species_mapping = {i: name for i, name in enumerate(data.target_names)}
iris_df['species'] = iris_df['species'].map(species_mapping)

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())

# Check for data types and missing values
print("\nData types:")
print(iris_df.dtypes)

print("\nMissing values:")
print(iris_df.isnull().sum())

# Clean the dataset (No missing values in Iris dataset, so no cleaning required)

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics:")
print(iris_df.describe())

# Group by species and compute the mean of numerical columns
print("\nMean values by species:")
print(iris_df.groupby('species').mean())

# Task 3: Data Visualization
sns.set(style="whitegrid")

# 1. Line chart (example: trends in mean petal length per species)
mean_petal_length = iris_df.groupby('species')['petal length (cm)'].mean()
plt.figure(figsize=(8, 5))
plt.plot(mean_petal_length, marker='o', linestyle='-', color='b')
plt.title("Mean Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Petal Length (cm)")
plt.grid(True)
plt.show()

# 2. Bar chart (example: mean sepal width per species)
mean_sepal_width = iris_df.groupby('species')['sepal width (cm)'].mean()
plt.figure(figsize=(8, 5))
mean_sepal_width.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Mean Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Width (cm)")
plt.xticks(rotation=0)
plt.show()

# 3. Histogram (example: distribution of petal length)
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['petal length (cm)'], kde=True, color='purple')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (example: sepal length vs. petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=iris_df['sepal length (cm)'], y=iris_df['petal length (cm)'], hue=iris_df['species'], palette='deep')
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
