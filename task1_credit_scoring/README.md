# Task 1: Credit Scoring System

An end-to-end Machine Learning pipeline that predicts the probability of credit default based on user financial profiles. It includes synthetic data generation, model training (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), and a real-time web dashboard.

## Features
- **Data Pipeline:** Generates synthetic credit data (Age, Income, Debt, Loan Amount, Credit History, etc.)
- **Model Comparison:** Evaluates 4 classification models and automatically selects the best one (based on ROC-AUC).
- **FastAPI Backend:** Serves the best-trained model for real-time inference.
- **Glassmorphism Dashboard:** A sleek, interactive UI with a Three.js 3D background to test predictions live.

## Project Structure
```
task1_credit_scoring/
├── data/                  # Generated datasets (credit_data.csv)
├── outputs/               # Saved models, scalers, plots, and JSON results
├── src/
│   ├── data_generator.py  # Script to generate synthetic credit data
│   ├── models.py          # Machine learning model definitions
│   └── visualizer.py      # Plotting functions (ROC curves, Confusion Matrices)
├── dashboard/
│   ├── index.html         # Frontend Dashboard UI
│   ├── app.js             # Frontend logic and API integration
│   └── style.css          # Styling (Glassmorphism, animations)
├── main.py                # Main ML pipeline (Train, evaluate, save)
└── server.py              # FastAPI server for real-time predictions
```

## Step-by-Step Setup & Execution

### 1. Train the Models
Before running the dashboard, you must run the machine learning pipeline to generate data, train the models, and save the best model weights.
```bash
cd task1_credit_scoring
python main.py
```
*This will create the `outputs/` folder containing `best_model.pkl`, `scaler.pkl`, and performance graphs.*

### 2. Start the Backend API
Run the FastAPI server to handle live inference requests from the dashboard.
```bash
python server.py
```
*The server will start at `http://localhost:8001`.*

### 3. Open the Dashboard
Simply double-click `dashboard/index.html` to open it in your web browser. 
- Scroll down to the **Live Prediction** section.
- Enter financial details (Income, Debt, etc.).
- Click "Run Live Prediction" to see the neural network instantly evaluate the credit score!
