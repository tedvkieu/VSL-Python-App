from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import pandas as pd
import os
from datetime import datetime
import base64
import io
from PIL import Image
import threading
import time

app = Flask(__name__)

# Load model và label
try:
    model = load_model("models/model_lstm_20_6.keras")
    labels = np.load("models/labels_lstm_20_6.npy")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    labels = None

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Global variables
sequence = []
collecting = False
last_prediction = ""
hand_missing_counter = 0
missing_threshold = 10
prediction_confidence = 0.0

def extract_both_hands_landmarks(results):
    """Trích xuất landmarks từ cả hai tay"""
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            if handedness.classification[0].label == 'Left':
                left_hand = coords
            else:
                right_hand = coords
                
    return left_hand + right_hand

def save_to_csv(sequence_data, prediction):
    """Lưu dữ liệu vào file CSV"""
    try:
        # Tạo thư mục data nếu chưa có
        os.makedirs('data', exist_ok=True)
        
        # Tạo tên file dựa trên thời gian hiện tại
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/hand_data_{timestamp}.csv"
        
        # Tạo DataFrame từ sequence
        df = pd.DataFrame(sequence_data)
        
        # Thêm cột prediction
        df['prediction'] = prediction
        
        # Lưu vào file CSV
        df.to_csv(filename, index=False)
        print(f"Đã lưu dữ liệu vào file: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None

def process_frame(frame_data):
    """Xử lý frame từ camera"""
    global sequence, collecting, last_prediction, hand_missing_counter, prediction_confidence
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Extract landmarks
        keypoints = extract_both_hands_landmarks(results)
        
        # Prepare landmarks data for client-side drawing
        landmarks_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_data = {
                    'hand_type': handedness.classification[0].label,
                    'landmarks': []
                }
                for lm in hand_landmarks.landmark:
                    hand_data['landmarks'].append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    })
                landmarks_data.append(hand_data)
        
        # Process hand detection logic
        if any(k != 0 for k in keypoints):
            # Phát hiện tay -> bắt đầu thu frame
            sequence.append(keypoints)
            collecting = True
            hand_missing_counter = 0
        else:
            if collecting:
                hand_missing_counter += 1
                if hand_missing_counter >= missing_threshold:
                    # Tay biến mất -> thực hiện dự đoán
                    if len(sequence) > 0 and model is not None:
                        input_data = np.expand_dims(sequence, axis=0)
                        prediction = model.predict(input_data, verbose=0)[0]
                        max_index = np.argmax(prediction)
                        max_label = labels[max_index]
                        confidence = prediction[max_index]
                        
                        print(f"Dự đoán: {max_label} ({confidence:.3f})")
                        
                        if confidence > 0.9 and max_label != "non-action":
                            last_prediction = max_label
                            prediction_confidence = confidence
                            # Lưu dữ liệu vào CSV
                            save_to_csv(sequence, max_label)
                        else:
                            last_prediction = ""
                            prediction_confidence = 0.0
                    
                    # Reset trạng thái
                    sequence.clear()
                    collecting = False
                    hand_missing_counter = 0
        
        return {
            'success': True,
            'landmarks': landmarks_data,
            'prediction': last_prediction,
            'confidence': float(prediction_confidence),
            'collecting': collecting,
            'sequence_length': len(sequence)
        }
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """API endpoint để xử lý frame"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data'})
        
        result = process_frame(frame_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_sequence', methods=['POST'])
def reset_sequence():
    """Reset sequence data"""
    global sequence, collecting, last_prediction, hand_missing_counter
    sequence.clear()
    collecting = False
    last_prediction = ""
    hand_missing_counter = 0
    return jsonify({'success': True})

@app.route('/get_status')
def get_status():
    """Lấy trạng thái hiện tại"""
    return jsonify({
        'prediction': last_prediction,
        'confidence': float(prediction_confidence),
        'collecting': collecting,
        'sequence_length': len(sequence),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)