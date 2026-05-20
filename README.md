# CodeAlpha Machine Learning Tasks

This repository contains my submissions for the CodeAlpha Machine Learning Internship. It features three comprehensive, end-to-end Machine Learning pipelines, complete with real-time inference backends and beautiful front-end interactive dashboards.

## Overview of Projects

### [Task 1: Credit Scoring System](./task1_credit_scoring)
An end-to-end pipeline that predicts the probability of credit default based on user financial profiles.
- **Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Backend:** FastAPI server serving a trained Logistic Regression model.
- **Frontend:** Glassmorphism dashboard with a live inference form.

### [Task 2: Emotion Recognition from Speech](./task2_emotion_recognition)
A deep learning project that classifies human emotion from raw audio recordings.
- **Models Evaluated:** CNN, Bi-LSTM, MLP.
- **Features Extracted:** 222 acoustic features (MFCCs, Chroma, Mel Spectrogram, etc.) via `librosa`.
- **Backend:** FastAPI server that decodes `.webm` browser recordings to `.wav` and runs inference via the Bi-LSTM neural network.
- **Frontend:** Interactive dashboard with a live microphone recording interface.

### [Task 3: Handwritten Character Recognition](./task3_handwritten_recognition)
A computer vision project that recognizes handwritten digits using Convolutional Neural Networks on the MNIST dataset.
- **Models Evaluated:** Standard CNN, Deep CNN with Dropout and Batch Normalization.
- **Backend:** FastAPI server featuring advanced image thresholding and scaling to map canvas data to the CNN's expected input space.
- **Frontend:** Futuristic dashboard with an integrated HTML5 drawing canvas.

---

## Global Setup & Requirements

### Prerequisites
Make sure you have Python 3.10+ installed.

### 1. Install Dependencies
Install all required libraries for the 3 projects globally:
```bash
pip install -r requirements.txt
```

### 2. Run the Projects
Each project operates independently. To run a project, navigate into its folder, run the training pipeline, and start the FastAPI server.

For example, to run Task 1:
```bash
# 1. Enter the project directory
cd task1_credit_scoring

# 2. Train the models and save the weights
python main.py

# 3. Start the live prediction backend
python server.py
```
After starting the server, open the `dashboard/index.html` file in any web browser to interact with the neural network in real-time!

> **Note:** Detailed, step-by-step instructions for each project are located in their respective `README.md` files inside their folders.
