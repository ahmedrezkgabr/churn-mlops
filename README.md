
# 📦 Customer Churn Prediction System

This project is an end-to-end machine learning solution for predicting customer churn in a telecom company. It features a complete pipeline that includes data exploration, preprocessing, model training, experiment tracking with MLflow, model serving via a FastAPI inference API, and optional interaction via a Streamlit dashboard.

---

## 📁 Project Structure

```bash
churn-prediction/
│
├── data/                   # Raw and processed data files
│   ├── processed/
│   └── raw/
├── models/                 # Trained and saved model artifacts
├── notebooks/              # Jupyter notebooks for EDA, training, and evaluation
├── outputs/                # Reports, evaluation metrics, or plots
├── src/                    # Core Python modules
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── utils.py
│   └── visualization.py
│
├── api.py                  # FastAPI backend for model inference
├── main.py                 # Streamlit interface (optional)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

---

## Components & Pipeline Overview

1. **Data Exploration (notebooks/)**
   - Understand feature distributions, correlations, missing values.
   - Visualizations for churned vs. non-churned customers.

2. **Preprocessing (src/data_preprocessing.py)**
   - Missing value handling
   - Categorical encoding
   - Feature scaling

3. **Model Training (src/model.py + notebook)**
   - Models trained:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Multi-layer Perceptron (MLP)

4. **Evaluation**
   - Accuracy, F1 score, ROC AUC
   - Plots: Confusion Matrix, ROC curves, etc.

5. **MLflow Experiment Tracking**
   - Logs metrics, models, artifacts.
   - Compares performance across multiple models.

6. **Inference API (FastAPI - `src/api.py`)**
   - Accepts JSON input
   - Preprocesses request
   - Loads best model
   - Returns prediction

---

## Tools & Libraries Used

- **Python 3.10**
- **Pandas**, **NumPy**, **scikit-learn** – Data processing and modeling
- **XGBoost** – Advanced gradient boosting model
- **MLflow** – Experiment tracking and model registry
- **FastAPI** – Inference API
- **Streamlit** – Optional frontend
- **Docker** – Containerized development and deployment
- **matplotlib**, **seaborn** – Visualizations

---

## How to Use the Project

### 1. Setup Environment

```bash
git clone https://github.com/ahmedrezkgabr/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 📊 2. Run Jupyter Notebook

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

Explore the data, visualize insights, and train models.

### 3. Train and Log Models

```bash
python main.py
```

- Trains multiple models
- Logs metrics and models to MLflow
- Saves best-performing model

### 4. Run FastAPI Inference Server

```bash
uvicorn api:app --reload
```

**Sample cURL Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

**sample_input.json**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No phone service",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "Contract": "One year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": "845.5"
}
```

### 5. Run with Docker (Optional)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Future Work

- Add unit and integration **tests**
- Automate the entire pipeline with **CI/CD**
- Deploy to **cloud (e.g., AWS/GCP)**
- Add **real-time monitoring** and logging
- Incorporate **automated model retraining**

---

## 👤 Author

- **Ahmed Rezk**
- [GitHub](https://github.com/ahmedrezkgabr)

---

## 📄 License

This project is for **educational purposes**. No license has been specified.
