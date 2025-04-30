from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_preprocessing import load_data, preprocess_data, log_processed_data
from src.visualization import *
from src.model import split_data, train_all_models, plot_roc_curves

# 1. Load Data
df = load_data(DATA_PATH)

# 2. Visualize before processing
plot_churn_distribution(df)
plot_tenure_vs_churn(df)
plot_monthly_charges(df)
plot_correlation_heatmap(df)

# 3. Preprocess
X, y = preprocess_data(df)
log_processed_data(df, PROCESSED_DATA_PATH)    # log_processed_data(X, y)


# 4. Train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

# 5. Train and evaluate models
models = train_all_models(X_train, X_test, y_train, y_test)

# 6. Plot ROC-AUC
plot_roc_curves(models, X_test, y_test)
