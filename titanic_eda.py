
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('tested.csv')

# Basic info
print(df.info())

# Descriptive statistics
print(df.describe())

# Value counts for survival
print(df['Survived'].value_counts())


# Pairplot
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']].dropna(), hue='Survived')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


for column in ['Age', 'Fare']:
     sns.histplot(df[column].dropna(), kde=True)
     plt.title(f"Histogram of {column}")
     plt.show()


for column in ['Age', 'Fare']:
     sns.boxplot(x='Survived', y=column, data=df)
     plt.title(f"Boxplot of {column} by Survival")
     plt.show()


sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title("Scatterplot of Age vs Fare by Survival")
plt.show()
