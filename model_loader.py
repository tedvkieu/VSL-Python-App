import os
print("Current working directory:", os.getcwd())
import numpy as np
from keras.models import load_model

# Load model và label chỉ 1 lần
model = load_model("models/model_lstm_20_6.keras")  # Sửa đuôi .keras
labels = np.load("models/labels_lstm_20_6.npy")