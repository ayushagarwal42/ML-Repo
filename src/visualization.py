# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data(df):
    # Pairplot of data features with hue based on the 'Outcome' column
    plt.figure(figsize=(10, 6))
    sns.pairplot(df, hue='Outcome', diag_kind='kde', markers=['o', 's'], palette='husl')
    plt.title("Pairplot of Data Features")
    plt.show()

    # Countplot to visualize the distribution of the 'Outcome' column
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Outcome')
    plt.title("Distribution of Outcome")
    plt.show()

    # You can add more visualization functions as needed
