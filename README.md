# Customer Churn Prediction

This project predicts customer churn in a telecom company using supervised machine learning techniques. It walks through data cleaning, exploratory data analysis (EDA), model training, and evaluation.

## 📁 Project Structure

``` bash
churn-prediction/
│
├── data/                   # Raw and processed data files
│   ├── processed/
│   └── raw/
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for exploration
├── outputs/                # Generated reports, plots, or artifacts
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── utils.py
│   └── visualization.py
│
├── .gitignore
├── Dockerfile
├── main.py                 # Streamlit application
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 📊 Features

- **Data Preprocessing**: Handles missing values, categorical encoding, and scaling.
- **EDA**: Visualizes patterns in customer data.
- **Modeling**: Logistic Regression, Decision Trees, Random Forest, and more.
- **Evaluation**: Accuracy, ROC AUC, Precision-Recall metrics.
- **Interactive App**: Simple interface to explore churn prediction using Streamlit.

## 📈 Model Performance

We use several metrics to evaluate model performance:

- Accuracy
- ROC AUC
- Confusion Matrix
- Precision & Recall

## 📚 Requirements

See `requirements.txt` for Python packages used in this project.

## 👤 Author

- **Ahmed Rezk**
- [Your GitHub Profile](https://github.com/ahmedrezkgabr)

## 📄 License

This project is for educational purposes. No license has been specified.
