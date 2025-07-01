import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

sns.set(style="whitegrid")

def load_data(filepath):
    """Load dataset from CSV or similar."""
    return pd.read_csv(filepath)

def dataset_overview(df):
    print("Shape:", df.shape)
    print("\nColumns and Data Types:\n", df.dtypes)
    print("\nFirst 5 Rows:\n", df.head())

def summary_statistics(df):
    print("\nSummary Statistics:\n", df.describe())

def plot_numerical_distribution(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_categorical_distribution(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Category Distribution - {col}")
        plt.xticks(rotation=45)
        plt.show()

def correlation_heatmap(df, numeric_columns):
    corr = df[numeric_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def missing_value_analysis(df):
    print("\nMissing Values:\n", df.isnull().sum())
    msno.matrix(df)
    plt.show()

def boxplot_outlier_detection(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col)
        plt.title(f"Outlier Detection - {col}")
        plt.show()
