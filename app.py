import json
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import base64
import io
from PIL import Image

from flask import Flask, jsonify, send_from_directory, request, render_template # Added render_template

# Attempt to import from local files
from main import PoseEstimator, calculate_angle, mp_pose, COLOR_CORRECT, COLOR_INCORRECT
from PoseCheck import PoseCheck
# Preprocessing functions will be copied directly into this file later

app = Flask(__name__)

CONFIG_FILE = 'exercise_config.json'
IMAGE_DIR = 'images' # This is relative to app.py, used by send_from_directory
# For static files like CSS/JS, Flask automatically looks for a 'static' folder.
# For templates, Flask automatically looks for a 'templates' folder.

MODEL_WEIGHTS_FILE = 'final_model_weights.pth'
CLASS_MAPPING_FILE = 'class_mapping.json'

# Global variables
exercise_config = None
pose_estimator_global = None
pose_check_model_global = None
class_mapping_global = None
index_to_class_global = None # For mapping model output index to class name

# --- Preprocessing functions (copied from preprocessing.py for simplicity) ---
def get_center_point(landmarks, body_part_from, body_part_to):
    landmark_from = landmarks.landmark[body_part_from.value]
    landmark_to = landmarks.landmark[body_part_to.value]
    center_x = (landmark_from.x + landmark_to.x) / 2
    center_y = (landmark_from.y + landmark_to.y) / 2
    center_z = (landmark_from.z + landmark_to.z) / 2
    return {'x': center_x, 'y': center_y, 'z': center_z, 'visibility': (landmark_from.visibility + landmark_to.visibility)/2}

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    if landmarks is None: return 0
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]

    if not (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and \
            left_hip.visibility > 0.5 and right_hip.visibility > 0.5):
        max_dist = 0
        for i in range(len(landmarks.landmark)):
            for j in range(i + 1, len(landmarks.landmark)):
                p1, p2 = landmarks.landmark[i], landmarks.landmark[j]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                    if dist > max_dist: max_dist = dist
        return max_dist * torso_size_multiplier if max_dist > 0 else 1.0

    center_x_shoulders = (left_shoulder.x + right_shoulder.x) / 2
    center_y_shoulders = (left_shoulder.y + right_shoulder.y) / 2
    center_x_hips = (left_hip.x + right_hip.x) / 2
    center_y_hips = (left_hip.y + right_hip.y) / 2
    torso_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)
    torso_height = np.sqrt((center_x_shoulders - center_x_hips)**2 + (center_y_shoulders - center_y_hips)**2)

    if torso_width == 0 or torso_height == 0:
        max_dist = 0
        for i in range(len(landmarks.landmark)):
            for j in range(i + 1, len(landmarks.landmark)):
                p1, p2 = landmarks.landmark[i], landmarks.landmark[j]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                    if dist > max_dist: max_dist = dist
        return max_dist * torso_size_multiplier if max_dist > 0 else 1.0
    pose_size = np.sqrt(torso_width**2 + torso_height**2)
    return pose_size * torso_size_multiplier if pose_size > 0 else 1.0

def normalize_pose_landmarks(landmarks):
    if landmarks is None: return None
    pose_center_x = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    pose_center_y = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    pose_size = get_pose_size(landmarks)
    if pose_size == 0: pose_size = 1.0
    normalized_landmarks = []
    for landmark in landmarks.landmark:
        norm_x = (landmark.x - pose_center_x) / pose_size
        norm_y = (landmark.y - pose_center_y) / pose_size
        norm_z_scaled = landmark.z / pose_size
        normalized_landmarks.extend([norm_x, norm_y, norm_z_scaled])
    return normalized_landmarks

def landmarks_to_embedding(landmarks_normalized):
    if landmarks_normalized is None: return None
    return np.array(landmarks_normalized)

def preprocess_data(landmarks_mp):
    if landmarks_mp is None: return None
    normalized_landmarks = normalize_pose_landmarks(landmarks_mp)
    if normalized_landmarks is None: return None
    embedding = landmarks_to_embedding(normalized_landmarks)
    return embedding
# --- End of Preprocessing functions ---

def load_app_state():
    global exercise_config, pose_estimator_global, pose_check_model_global, class_mapping_global, index_to_class_global
    try:
        with open(CONFIG_FILE, 'r') as f: exercise_config = json.load(f)
    except Exception as e: print(f"ERROR loading {CONFIG_FILE}: {e}"); exercise_config = {}

    pose_estimator_global = PoseEstimator(exercise_config if exercise_config else {})
    if not exercise_config: print("Warning: PoseEstimator initialized with empty config.")

    try:
        with open(CLASS_MAPPING_FILE, 'r') as f: class_mapping_global = json.load(f)
        index_to_class_global = {v: k for k, v in class_mapping_global.items()}
    except Exception as e:
        print(f"ERROR loading {CLASS_MAPPING_FILE}: {e}")
        class_mapping_global, index_to_class_global = {}, {}

    input_size = 99
    num_classes = len(index_to_class_global) if index_to_class_global else 1
    if num_classes == 1 and class_mapping_global : num_classes = len(class_mapping_global) # if index_to_class failed but class_mapping loaded
    if num_classes == 1: print("ERROR: Cannot determine number of classes for PoseCheck model.")

    pose_check_model_global = PoseCheck(input_size=input_size, num_classes=num_classes)
    try:
        if os.path.exists(MODEL_WEIGHTS_FILE):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pose_check_model_global.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
            pose_check_model_global.eval()
            print(f"PoseCheck model loaded from {MODEL_WEIGHTS_FILE} onto {device}.")
        else:
            print(f"ERROR: Model weights file {MODEL_WEIGHTS_FILE} not found.")
            pose_check_model_global = None
    except Exception as e:
        print(f"Error loading model weights: {e}")
        pose_check_model_global = None

@app.route('/')
def home():
    return jsonify({'message': 'YogaFix API is running!'})

@app.route('/get_exercise_config', methods=['GET'])
def get_exercise_config_route():
    if exercise_config is None: return jsonify({"error": "Configuration not loaded"}), 500
    return jsonify(exercise_config)

@app.route('/ui') # New route for the UI
def ui_home():
    return render_template('index.html')

# Note: The '/images/<path:filename>' route is for serving images specified in exercise_config.json
# For static assets like CSS and JS referenced in index.html (e.g. static/style.css),
# Flask's `url_for('static', filename='style.css')` works automatically if you have a 'static' folder
# at the same level as your 'templates' folder. The IMAGE_DIR global is for dynamic images from config.
@app.route('/images/<path:filename>')
def get_image(filename):
    # This serves images from the IMAGE_DIR directory (e.g., 'images/mountain_pose.png')
    # Ensure IMAGE_DIR is correctly defined and accessible.
    # If images in config are like "pose_1.png", and IMAGE_DIR is "images", it works.
    # If image_url in config is "images/pose_1.png", then send_from_directory should get 'images' as base.
    # The current setup assumes image_url in config is like "images/pose_1.png",
    # and the route is /images/images/pose_1.png which is not ideal.
    # Let's adjust how images are served or how they are referenced.
    # If config stores "mountain_pose.png" and IMAGE_DIR is "images", then this is fine.
    # The JS code currently does `poseImageDisplay.src = `/${currentPoseData.image_url}`;`
    # If image_url is "images/mountain.png", this becomes "/images/mountain.png" which matches this route.
    return send_from_directory(os.getcwd(), filename) # Serve from basedir if filename includes 'images/'

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    base64_image_string = data['image']
    current_pose_name_req = data.get('current_pose_name')

    try:
        image_bytes = base64.b64decode(base64_image_string)
        pil_image = Image.open(io.BytesIO(image_bytes))
        frame_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

    landmarks_mp_serializable, predicted_pose_name_cnn = None, "Error or no landmarks"
    angle_details_expert, all_joints_correct_expert = {}, False
    error_message = None

    if pose_estimator_global is None: return jsonify({'error': 'PoseEstimator not initialized'}), 500

    results = pose_estimator_global.detector.process(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
    landmarks_mp = results.pose_landmarks

    if landmarks_mp:
        landmarks_mp_serializable = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in landmarks_mp.landmark]

        if pose_check_model_global and index_to_class_global:
            preprocessed_landmarks = preprocess_data(landmarks_mp)
            if preprocessed_landmarks is not None:
                try:
                    input_tensor = torch.tensor(preprocessed_landmarks, dtype=torch.float32).unsqueeze(0)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    input_tensor = input_tensor.to(device)
                    with torch.no_grad():
                        outputs = pose_check_model_global(input_tensor)
                        _, predicted_idx = torch.max(outputs, 1)
                        predicted_pose_name_cnn = index_to_class_global.get(predicted_idx.item(), "Unknown pose index")
                except Exception as e:
                    error_message = f"Error during model inference: {str(e)}"
                    predicted_pose_name_cnn = "Inference error"
            else: predicted_pose_name_cnn = "Preprocessing failed"
        else:
            predicted_pose_name_cnn = "Model or class mapping not loaded"
            if not pose_check_model_global: error_message = (error_message or "") + " PoseCheck model unavailable."
            if not index_to_class_global: error_message = (error_message or "") + " Class mapping unavailable."

        if current_pose_name_req and current_pose_name_req != "unknown":
            criteria = pose_estimator_global.exercise_config.get('poses', {}).get(current_pose_name_req, {}).get('criteria')
            if criteria:
                angle_details_expert, all_joints_correct_expert, _ = pose_estimator_global.get_angle_details(
                    landmarks_mp, current_pose_name_req, use_visibility_filter=True
                )
            else:
                angle_details_expert = {"message": f"No criteria found for {current_pose_name_req}"}
        else:
            angle_details_expert = {"message": "No current_pose_name provided or is 'unknown'"}
    else:
        error_message = (error_message or "") + " No landmarks detected in image."
        predicted_pose_name_cnn = "No landmarks detected"

    return jsonify({
        'landmarks_mp': landmarks_mp_serializable,
        'predicted_pose_name_cnn': predicted_pose_name_cnn,
        'angle_details_expert': angle_details_expert,
        'all_joints_correct_expert': all_joints_correct_expert,
        'error_message': error_message
    })

if __name__ == '__main__':
    load_app_state()
    if pose_check_model_global is None or pose_estimator_global is None or class_mapping_global is None :
        print("CRITICAL ERROR: Essential components failed to load. API may not function correctly.")
    app.run(debug=True, host='0.0.0.0', port=os.environ.get("PORT", 5000))
