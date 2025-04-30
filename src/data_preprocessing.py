import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    # Drop unrelated or identifier columns
    df.drop('customerID', axis=1, inplace=True)

    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Convert data types
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Convert categorical variables to numerical
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Convert categorical variables to dummy/indicator variables
    df = pd.get_dummies(df, columns=[
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ])

    # Scale numerical features
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Separate features and target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def log_processed_data(df, output_path='data/processed/churn-processed.csv'):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data logged to {output_path}")