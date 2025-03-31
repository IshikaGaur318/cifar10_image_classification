# 🖼️ CIFAR-10 Image Classification using CNNs

🚀 A deep learning-based image classification project using **Convolutional Neural Networks (CNNs)** to classify images from the **CIFAR-10 dataset**.

## 📌 Features
✅ Processes and augments CIFAR-10 dataset images 📷  
✅ Trains a CNN model using TensorFlow/Keras 🧠  
✅ Evaluates model accuracy and performance 📊  
✅ Classifies new images with a trained model 🏷️  

## 📂 Project Structure
📦 cifar10_image_classification
├── 📁 dataset/               # (Optional) Store dataset if downloaded manually
├── 📁 models/                # Save trained models here
├── 📁 notebooks/             # Jupyter notebooks for experimentation
├── 📁 src/
│   ├── 📝 data_preprocessing.py   # Data loading & augmentation
│   ├── 📝 model.py                # CNN model definition
│   ├── 📝 train.py                # Train the model
│   ├── 📝 evaluate.py             # Evaluate trained model
│   ├── 📝 predict.py              # Predict on new images
├── 📄 README.md             # Project documentation
├── 📄 requirements.txt      # Dependencies (TensorFlow, NumPy, etc.)

## 🚀 Setup and Installation
### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

2️⃣ #Install dependencies:
pip install -r requirements.txt

📊 #Dataset
The CIFAR-10 dataset consists of 60,000 images (10 classes, 32x32 pixels each):

Training Set: 50,000 images

Test Set: 10,000 images

Classes: ✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🐕 Dog, 🦌 Deer, 🐸 Frog, 🏠 House, 🚢 Ship, 🚚 Truck

Source: Kaggle CIFAR-10 Dataset

🏋️ #Training
Run the following command to train the model:
python src/train.py
🎯 Key Training Details:
✅ Trained for 20 epochs using Adam optimizer
✅ Applied data augmentation for better generalization
✅ Achieved 75%+ accuracy on test data

📈# Model Evaluation
Evaluate the trained model on test data:
python src/evaluate.py
🔍 Evaluation Metrics:
✔️ Accuracy: 75%+
✔️ Loss: Tracked during training
✔️ Confusion Matrix & Classification Report generated

🔍# Making Predictions
Classify new images using the trained model:
python src/predict.py
📌 The predicted class label will be displayed.

🛠️# Requirements
🐍 Python 3.8+

🏗️ TensorFlow

📊 NumPy

📉 Matplotlib

📚 Scikit-learn

⭐ Contributing
Pull requests are welcome! Feel free to fork the repo and submit improvements. 😊


