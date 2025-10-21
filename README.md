# 🧠 End-to-End ANN Digit Recognition Project (MNIST Dataset)

 A complete Deep Learning project to predict handwritten digits (0–9) using an Artificial Neural Network (ANN)

# 📘 Project Overview

This project focuses on building, training, and deploying an Artificial Neural Network (ANN) on the MNIST dataset, one of the most popular datasets in computer vision and deep learning.

The goal is to recognize handwritten digits (0–9) from grayscale 28×28 images.

This project covers the entire ML lifecycle — from data loading and preprocessing to model training, evaluation, and deployment using Streamlit.

# 🔄 End-to-End Workflow

1️⃣ Data Loading & Understanding

Used MNIST dataset (70,000 grayscale digit images, 28×28 pixels).

Each image represents digits 0–9.

Split data into training and test sets for supervised learning.

2️⃣ Data Preprocessing

Normalized pixel values (0–255) → (0–1) for faster convergence.

Flattened each image (28×28 → 784 features).

One-hot encoded target labels for multiclass classification.

3️⃣ Model Building (ANN)

Created a Sequential ANN model using Keras:

Input Layer: 784 neurons

Hidden Layers: Dense layers with ReLU activation

Output Layer: 10 neurons (Softmax)

Optimizer: Adam

Loss Function: Categorical Crossentropy

4️⃣ Model Training

Trained model on 60,000 training samples and validated on test set.

Used batch training with epochs tuning to avoid overfitting.

Visualized accuracy/loss curves using Matplotlib.

5️⃣ Model Evaluation

Evaluated on 10,000 test images:

Accuracy: 98–99%

Loss: Very low (<0.1)

Generated confusion matrix and classification report.

6️⃣ Model Saving

Saved the trained best model as:

model.save("model5.h5")


for future prediction and deployment.

7️⃣ Streamlit Deployment

Developed a user-friendly Streamlit interface where:

Users upload an image (digit in .png/.jpg format).

The image is grayscale converted, normalized, and flattened.

The model predicts the digit (0–9) with confidence score.

# 🧠 Model Summary

Layer	        Output Shape	Parameters

Dense (ReLU)	 (None, 128)	    100,480

Dense (ReLU)	 (None, 64)	     8,256

Dense (Softmax)	 (None, 10)	       650

Total Parameters	            109,386	


# 📈 Results

Metric	               Value

Training Accuracy	   99.1%

Validation Accuracy	   98.4%

Test Accuracy	       98.2%

Loss	                0.07

Model	            model5.h5



# 💻 Streamlit App Usage

Run the app locally:

streamlit run app.py


# App Features

Upload handwritten digit images (JPG or PNG)

Model normalizes and predicts instantly

Displays predicted digit and confidence level

Clean, interactive UI with automatic preprocessing


# 🧠 Key Learnings

Implemented ANN from scratch using Keras

Gained deep understanding of model tuning and optimization

Performed data normalization & flattening for image-based input

Evaluated metrics like accuracy, precision, and recall

Created an interactive Streamlit web app for digit recognition

# 🚀 Future Enhancements

Extend ANN to CNN for better spatial accuracy

Add digit drawing canvas for real-time prediction

Deploy on Streamlit Cloud / Hugging Face Spaces

Integrate Grad-CAM for model explainability

# 👨‍💻 Author

👤 [SURYAVHI DAS]
Senior Data Science Learner | Deep Learning Enthusiast

📧 [dassuryavhi123@gmail.com]

🔗 LinkedIn:www.linkedin.com/in/suryavhi-das-a95094351
 

# 🌟 Acknowledgements

TensorFlow/Keras for deep learning framework

MNIST Dataset for digit recognition research

Streamlit for model deployment

Intellipaat Mentors for guidance on Deep Learning projects