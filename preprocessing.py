import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from enum import Enum
import json

class BodyPart(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    class_names=df.pose_label
    unique_classes = df.pose_label.unique()
    class_to_index = {name: idx for idx, name in enumerate(unique_classes)}
    index_to_class = {idx: name for name, idx in class_to_index.items()}

    # Numeric label column
    df['class_no'] = class_names.map(class_to_index)

    # Extract X (features), y (one-hot), and numeric class mapping
    X = df.drop(columns=['pose_label'])
    X= X.drop(columns=['class_no'])  # Drop the class_no column to get the landmarks

    # y = pd.get_dummies(class_names).astype(int) #not doing now
    y=df['class_no']



    return X, y, class_names,index_to_class,class_to_index

def get_center_point(landmarks, left_idx, right_idx):
    left = landmarks[:, left_idx, :]
    right = landmarks[:, right_idx, :]
    center = (left + right) * 0.5
    return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP.value, BodyPart.RIGHT_HIP.value)
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER.value, BodyPart.RIGHT_SHOULDER.value)
    torso_size = torch.norm(shoulders_center - hips_center, dim=1)
    
    pose_center = hips_center.unsqueeze(1)
    pose_center = pose_center.expand(-1, landmarks.size(1), -1)
    d = landmarks - pose_center
    max_dist = torch.max(torch.norm(d, dim=2), dim=1).values
    pose_size = torch.max(torso_size * torso_size_multiplier, max_dist)
    return pose_size

def normalize_pose_landmarks(landmarks):
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP.value, BodyPart.RIGHT_HIP.value)
    pose_center = pose_center.unsqueeze(1).expand(-1, landmarks.size(1), -1)
    landmarks = landmarks - pose_center
    pose_size = get_pose_size(landmarks).unsqueeze(1).unsqueeze(2)
    landmarks = landmarks / pose_size
    return landmarks

def landmarks_to_embedding(landmarks_and_scores):
    
    landmarks_and_scores = landmarks_and_scores.reshape(-1, 33, 4)
    # landmarks = normalize_pose_landmarks(landmarks_and_scores[:, :, :3])
    landmarks = normalize_pose_landmarks(landmarks_and_scores)
    embedding = landmarks.view(landmarks.size(0), -1)
    return embedding

def preprocess_data(X_df):
    assert X_df.shape[1] == 132
    data = torch.tensor(X_df.values, dtype=torch.float32)
    embeddings = landmarks_to_embedding(data)
    return embeddings

# Load and preprocess datasets
X, y, class_names,index_to_class,class_to_index = load_csv('pose_landmarks_data_labeled.csv')



processed_X_train = pd.DataFrame(preprocess_data(X))
processed_X_train.to_csv('processed_train_X_data.csv',  index=False)
y.to_csv('processed_train_y_data.csv', index=False)
class_names.to_csv('processed_train_class_names.csv', index=False)
# Save to JSON
with open('class_mapping.json', 'w') as f:
    json.dump(index_to_class, f)


# processed_X_val = preprocess_data(X_val)
# processed_X_test = preprocess_data(X_test)
