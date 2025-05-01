import mlflow
from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_preprocessing import load_data, preprocess_data, log_processed_data
from src.visualization import *
from src.model import split_data, train_all_models, plot_roc_curves

def main():
    mlflow.set_experiment("Telco-Churn-Experiments")

    with mlflow.start_run():
        # 1. Load Data
        df = load_data(DATA_PATH)

        # Log raw data shape
        mlflow.log_param("raw_rows", df.shape[0])
        mlflow.log_param("raw_columns", df.shape[1])

        # 2. Visualize before processing (optional to log plots later)
        plot_churn_distribution(df)
        plot_tenure_vs_churn(df)
        plot_monthly_charges(df)
        plot_correlation_heatmap(df)

        # 3. Preprocess
        X, y = preprocess_data(df)
        log_processed_data(X, y)

        mlflow.log_param("features_after_preprocessing", X.shape[1])

        # 4. Split
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 5. Train models & evaluate
        models = train_all_models(X_train, X_test, y_train, y_test)

        # Log metrics
        for model_name, model_info in models.items():
            mlflow.log_metric(f"{model_name}_accuracy", model_info['accuracy'])
            mlflow.log_metric(f"{model_name}_f1", model_info['f1_score'])
            # Save model artifact
            mlflow.sklearn.log_model(model_info['model'], artifact_path=f"{model_name}_model")

        # 6. Plot ROC
        plot_roc_curves(models, X_test, y_test)

if __name__ == "__main__":
    main()
