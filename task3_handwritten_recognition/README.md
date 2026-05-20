# Task 3: Handwritten Character Recognition

A deep learning computer vision project that recognizes handwritten digits. It builds, trains, and compares a standard Convolutional Neural Network (CNN) against a Deep CNN using TensorFlow and Keras on the MNIST dataset.

## Features
- **Computer Vision Model:** Trains a Deep CNN (with Batch Normalization, Dropout, and multiple Convolutional layers) to achieve ~99.5% test accuracy.
- **Image Preprocessing API:** Contains an advanced backend algorithm that thresholds, scales, and transforms freehand canvas drawings (black/colored backgrounds) into the exact 28x28 grayscale matrices expected by the neural network.
- **Interactive UI Canvas:** Features a futuristic web dashboard with an embedded drawing pad, allowing users to draw digits and see the CNN's real-time prediction and confidence map.

## Project Structure
```
task3_handwritten_recognition/
├── data/                  # Local cache for the MNIST dataset (npz format)
├── outputs/               # Saved model weights (deep_cnn.weights.h5) and training plots
├── src/
│   ├── data_loader.py     # MNIST dataset loader
│   ├── models.py          # Keras CNN architecture definitions
│   └── visualizer.py      # Matplotlib charting for training curves & confusion matrix
├── dashboard/
│   ├── index.html         # Frontend Dashboard UI with HTML5 Drawing Canvas
│   ├── app.js             # Canvas drawing logic and FastAPI networking
│   └── style.css          # UI Styling
├── main.py                # Main ML pipeline (Train, evaluate, save weights)
└── server.py              # FastAPI server for image processing & prediction
```

## Step-by-Step Setup & Execution

### 1. Train the Deep Neural Network
Run the main script to download the MNIST dataset, train both CNN models, generate performance graphs, and save the trained weights.
```bash
cd task3_handwritten_recognition
python main.py
```
*This will create the `outputs/` folder containing `deep_cnn.weights.h5`.*

### 2. Start the Backend API
Run the FastAPI server. This server will listen for base64 encoded images from the drawing canvas, apply necessary image thresholding, and feed it into the CNN.
```bash
python server.py
```
*The server will start at `http://localhost:8003`.*

### 3. Open the Dashboard & Draw
Double-click `dashboard/index.html` to open it in your web browser. 
- You will see a large square canvas labeled "Draw a Digit".
- Use your mouse to draw any number from 0-9.
- Click **Analyze**. The API will instantly process the image and output the predicted character and its confidence percentage!
