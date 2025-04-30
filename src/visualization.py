import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    sns.countplot(data=df, x='Churn')
    plt.title("Churn Distribution")
    plt.show()

def plot_tenure_vs_churn(df):
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
    plt.title("Customer Tenure vs. Churn")
    plt.show()

def plot_monthly_charges(df):
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title("Monthly Charges by Churn Status")
    plt.show()

def plot_correlation_heatmap(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
