# YogaFix



---

**1. `config.yaml`**

```yaml
# --- Data Collection ---
video_source_dir: "videos/input"
frame_output_dir: "data/raw_frames"
frame_sample_rate: 10 # fps

# --- SMPL Conversion ---
smpl_output_dir: "data/smpl_params"
vibe_romp_confidence_threshold: 0.7 # Example threshold

# --- Pose Pair Generation ---
pose_pair_output_dir: "data/pose_pairs"
perturbation_factors: # Example factors for joint angle noise (in radians)
  knee_bend: 0.5
  shoulder_raise: 0.3
  elbow_bend: 0.4
num_perturbations_per_pose: 5

# --- Text Label Generation ---
# (Specific rules might be coded directly, but key params can be here)
angle_threshold_degrees: 15.0

# --- SMPL to MediaPipe Conversion ---
smpl_mediapipe_map_model: "models/smpl_to_mediapipe_mapper.joblib" # Path to save/load the mapping model

# --- Dataset Structuring ---
structured_dataset_file: "data/yoga_feedback_dataset.jsonl"
pose_names:
  - "Warrior II"
  - "Triangle Pose"
  - "Downward-Facing Dog"
  - "Tree Pose"
  - "Plank Pose"
  # Add all relevant yoga pose names

# --- Model Architecture ---
model_output_dir: "models/feedback_model"
embedding_dim: 128
hidden_dim: 256
num_encoder_layers: 3
num_decoder_layers: 3
nhead: 8 # For Transformer
vocab_size: 1000 # Placeholder - determine after tokenization
max_seq_len: 50  # Max length of feedback text

# --- Training ---
batch_size: 32
learning_rate: 0.0001
num_epochs: 50
lr_decay_step: 10
lr_decay_gamma: 0.5
log_dir: "logs/training"
save_checkpoint_epoch: 5

# --- Application ---
live_feedback_cooldown_seconds: 3.0
tts_engine: "pyttsx3" # or "google" etc.
show_visual_feedback: True
target_pose_for_live_app: "Warrior II" # Example target pose for the app

# --- MediaPipe ---
mediapipe_model_complexity: 1
min_detection_confidence: 0.5
min_tracking_confidence: 0.5
```

---

**2. `utils.py`**

```python
import yaml
import numpy as np
import cv2
import torch
import mediapipe as mp
import logging
import os
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# --- Pose Normalization ---
def normalize_pose(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes pose landmarks (e.g., MediaPipe) to be centered and scaled.
    Assumes landmarks shape is (N_joints, 3) or (N_joints * 3,).
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape((-1, 3))
    
    if landmarks.shape[0] == 0:
        return landmarks # Return empty if no landmarks

    # 1. Center the pose at the midpoint of hips (or a reference point)
    # Using midpoint of shoulders and hips as a robust center estimate
    left_shoulder_idx = 11 # MediaPipe indices
    right_shoulder_idx = 12
    left_hip_idx = 23
    right_hip_idx = 24
    
    # Check if key indices are present (handle potential visibility issues)
    valid_indices = [i for i in [left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx] if i < len(landmarks)]
    if not valid_indices: # If no key points, maybe use centroid
        center = landmarks.mean(axis=0)
    else:
        key_points = landmarks[valid_indices]
        center = key_points.mean(axis=0)

    normalized_landmarks = landmarks - center

    # 2. Scale the pose based on a characteristic distance (e.g., shoulder width or torso height)
    if left_shoulder_idx < len(landmarks) and right_shoulder_idx < len(landmarks):
      shoulder_dist = np.linalg.norm(normalized_landmarks[left_shoulder_idx] - normalized_landmarks[right_shoulder_idx])
    else:
        shoulder_dist = None

    if left_hip_idx < len(landmarks) and right_hip_idx < len(landmarks):
       hip_dist = np.linalg.norm(normalized_landmarks[left_hip_idx] - normalized_landmarks[right_hip_idx])
    else:
        hip_dist = None

    # Use average of shoulder and hip distance if both available, otherwise use whichever is available, or fallback
    valid_dists = [d for d in [shoulder_dist, hip_dist] if d is not None and d > 1e-6]

    if valid_dists:
        scale_dist = np.mean(valid_dists)
    else:
        # Fallback: max distance from center (less robust)
        scale_dist = np.max(np.linalg.norm(normalized_landmarks, axis=1))

    if scale_dist > 1e-6:
         normalized_landmarks /= scale_dist
    else:
        logging.warning("Could not determine a valid scale for normalization.")


    return normalized_landmarks.flatten() # Return flattened array for model input

# --- Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, connections: List[Tuple[int, int]] = None):
    """Draws landmarks and connections on the image."""
    if landmarks is None or landmarks.shape[0] == 0:
        return image

    # Reshape if flattened
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape((-1, 3)) # Assume (N, 3)

    # Convert normalized coordinates back to pixel coordinates for drawing
    h, w, _ = image.shape
    pixel_landmarks = []
    for lm in landmarks:
        # Assuming landmarks input here are relative x,y,z NOT normalized yet
        # If they were normalized, this needs adjustment or use original landmarks
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        pixel_landmarks.append((cx, cy))

    # Draw keypoints
    for idx, point in enumerate(pixel_landmarks):
        cv2.circle(image, point, 5, (0, 255, 0), cv2.FILLED)
        # cv2.putText(image, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


    # Draw connections (use MediaPipe standard connections if none provided)
    if connections is None:
        connections = mp_pose.POSE_CONNECTIONS

    if connections:
      mp_drawing.draw_landmarks(
          image,
          # Create a dummy LandmarkList object for drawing utility
          # Requires normalized x, y, visibility
          # This part needs careful handling depending on what `landmarks` contains
          # Assuming `landmarks` are the direct output from MediaPipe results for now
          _create_mediapipe_landmark_list(landmarks),
          connections,
          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
          )

    return image


def _create_mediapipe_landmark_list(landmarks_np: np.ndarray):
    """Helper to create a MediaPipe LandmarkList from a numpy array."""
    from mediapipe.framework.formats import landmark_pb2
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    if landmarks_np.ndim == 1:
        landmarks_np = landmarks_np.reshape((-1, 3))

    for i in range(landmarks_np.shape[0]):
        landmark = landmark_list.landmark.add()
        landmark.x = landmarks_np[i, 0]
        landmark.y = landmarks_np[i, 1]
        # Use z as visibility proxy if available and scaled appropriately, otherwise assume visible
        landmark.z = landmarks_np[i, 2] # Might need adjustment based on normalization
        landmark.visibility = 1.0 # Assume visible for now
    return landmark_list

def draw_feedback(image: np.ndarray, text: str):
    """Draws feedback text on the image."""
    h, w, _ = image.shape
    # Position text at the bottom center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (w - text_size[0]) // 2
    text_y = h - 30 # 30 pixels from the bottom

    # Add a semi-transparent background rectangle for better readability
    overlay = image.copy()
    cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    alpha = 0.6 # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Put the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


# --- Model Loading ---
def load_pytorch_model(model_path: str, model_instance: torch.nn.Module, device: str = 'cpu') -> torch.nn.Module:
    """Loads a trained PyTorch model checkpoint."""
    try:
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        # Adjust based on how checkpoint was saved (might contain 'model_state_dict', 'epoch', etc.)
        if 'model_state_dict' in checkpoint:
            model_instance.load_state_dict(checkpoint['model_state_dict'])
        else:
             model_instance.load_state_dict(checkpoint)
        model_instance.to(device)
        model_instance.eval() # Set to evaluation mode
        logging.info(f"Model loaded from {model_path} and set to evaluation mode.")
        return model_instance
    except Exception as e:
        logging.error(f"Error loading PyTorch model from {model_path}: {e}")
        raise

# --- Tokenizer Placeholder ---
# In a real scenario, use libraries like Hugging Face's tokenizers or build your own
class SimpleTokenizer:
    def __init__(self, vocab=None, max_vocab_size=1000):
        self.word_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        self.max_vocab_size = max_vocab_size
        if vocab:
            self.build_vocab(vocab)

    def build_vocab(self, sentences: List[str]):
        word_counts = {}
        for sentence in sentences:
            for word in self._tokenize_text(sentence):
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort words by frequency, limit vocab size
        sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        for word, _ in sorted_words:
            if word not in self.word_to_idx and self.vocab_size < self.max_vocab_size:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
        logging.info(f"Built vocabulary with {self.vocab_size} words.")


    def _tokenize_text(self, text: str) -> List[str]:
         # Simple whitespace and punctuation splitting
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace()) # Keep only letters, numbers, spaces
        return text.split()

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                  for word in self._tokenize_text(text)]
        encoded = [self.word_to_idx['<SOS>']] + tokens + [self.word_to_idx['<EOS>']]
        # Pad or truncate
        padded = encoded[:max_len] + [self.word_to_idx['<PAD>']] * (max_len - len(encoded))
        return padded

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for idx in token_ids:
            word = self.idx_to_word.get(idx, '<UNK>')
            if word == '<EOS>' or word == '<PAD>': # Stop decoding at EOS or PAD
                break
            if word != '<SOS>': # Skip SOS token
                 words.append(word)
        return ' '.join(words)

# --- SMPL Constants (Placeholder - adapt if using specific SMPL library) ---
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]
SMPL_JOINT_MAP = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}

# --- MediaPipe Constants ---
MEDIAPIPE_JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
    'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
    'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]
MEDIAPIPE_JOINT_MAP = {name: i for i, name in enumerate(MEDIAPIPE_JOINT_NAMES)}

if __name__ == '__main__':
    # Example Usage
    config = load_config()
    print("Config loaded:", config)

    # Example normalization (dummy data)
    dummy_landmarks = np.random.rand(33, 3) * np.array([640, 480, 1]) # Example pixel coords + depth
    # Convert to relative coords for normalization function (if needed)
    # dummy_landmarks_relative = dummy_landmarks / np.array([640, 480, 1])
    # normalized_flat = normalize_pose(dummy_landmarks_relative.flatten())
    # print("Normalized (flattened) shape:", normalized_flat.shape)

    # Example Tokenizer
    sentences = ["Straighten your left leg.", "Bend your right knee more."]
    tokenizer = SimpleTokenizer(max_vocab_size=config.get('vocab_size', 1000))
    tokenizer.build_vocab(sentences)
    encoded = tokenizer.encode("Straighten your right knee.", max_len=config.get('max_seq_len', 50))
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    # Example drawing
    # blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # drawn_image = draw_landmarks(blank_image, dummy_landmarks) # Needs proper landmark format
    # cv2.imshow("Example Drawing", drawn_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
```

---

**3. `data_collection.py`**

```python
import os
import cv2
import logging
from pytube import YouTube
# Or use yt_dlp if preferred:
# import yt_dlp
from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_youtube_video(url: str, output_path: str):
    """Downloads a YouTube video using pytube."""
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            logging.error(f"No suitable stream found for {url}")
            return None
        logging.info(f"Downloading '{yt.title}'...")
        downloaded_file = stream.download(output_path=output_path)
        logging.info(f"Download complete: {downloaded_file}")
        return downloaded_file
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return None

# Alternative using yt-dlp
# def download_youtube_video_yt_dlp(url: str, output_path: str):
#     """Downloads a YouTube video using yt-dlp."""
#     ydl_opts = {
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
#         'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
#         'merge_output_format': 'mp4',
#     }
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(url, download=True)
#             filename = ydl.prepare_filename(info)
#             logging.info(f"Download complete: {filename}")
#             return filename
#     except Exception as e:
#         logging.error(f"Error downloading {url} with yt-dlp: {e}")
#         return None


def extract_frames(video_path: str, output_dir: str, sample_rate: int, pose_name: str):
    """Extracts frames from a video file."""
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / sample_rate))
    if frame_interval <= 0:
        frame_interval = 1
        logging.warning(f"Adjusted frame interval to 1 (requested sample rate {sample_rate} >= video fps {video_fps})")


    frame_count = 0
    saved_count = 0
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    pose_output_dir = os.path.join(output_dir, pose_name, video_basename)
    os.makedirs(pose_output_dir, exist_ok=True)

    logging.info(f"Extracting frames from {video_path} every {frame_interval} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(pose_output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    logging.info(f"Extracted {saved_count} frames for pose '{pose_name}' from {video_path}")


if __name__ == "__main__":
    config = load_config()

    # --- Configuration ---
    VIDEO_URLS = {
        "Warrior II": [
            "https://www.youtube.com/watch?v=example_warrior2_url1",
             "https://www.youtube.com/watch?v=example_warrior2_url2",
             # Add actual YouTube URLs for Warrior II pose videos
        ],
        "Triangle Pose": [
            "https://www.youtube.com/watch?v=example_triangle_url1",
            # Add actual YouTube URLs for Triangle Pose videos
        ],
        # Add more poses and their corresponding video URLs
    }
    VIDEO_OUTPUT_DIR = config['video_source_dir']
    FRAME_OUTPUT_DIR = config['frame_output_dir']
    FRAME_SAMPLE_RATE = config['frame_sample_rate']

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    # --- Download Videos ---
    downloaded_videos = {} # Store path per pose
    for pose, urls in VIDEO_URLS.items():
        downloaded_videos[pose] = []
        pose_video_dir = os.path.join(VIDEO_OUTPUT_DIR, pose)
        os.makedirs(pose_video_dir, exist_ok=True)
        for url in urls:
            # Simple check if file might already exist (based on URL - needs better handling)
            # A more robust check would involve checking filename patterns from pytube/yt-dlp
            # For now, just attempt download each time or implement manual check
            logging.info(f"Processing URL for {pose}: {url}")
            # Use pytube (or yt-dlp)
            downloaded_file_path = download_youtube_video(url, pose_video_dir)
            # downloaded_file_path = download_youtube_video_yt_dlp(url, pose_video_dir) # Alternative

            if downloaded_file_path:
                downloaded_videos[pose].append(downloaded_file_path)
            else:
                logging.warning(f"Skipping frame extraction for failed download: {url}")


    # --- Extract Frames ---
    for pose, video_paths in downloaded_videos.items():
        for video_path in video_paths:
             extract_frames(video_path, FRAME_OUTPUT_DIR, FRAME_SAMPLE_RATE, pose)

    logging.info("-" * 30)
    logging.info("MANUAL STEP REQUIRED:")
    logging.info("Please review the extracted frames in:")
    logging.info(f"  {FRAME_OUTPUT_DIR}")
    logging.info("1. Remove frames that are blurry, transitional, or incorrect poses.")
    logging.info("2. Ensure each subfolder (e.g., 'Warrior II') contains only clean, correct examples of that pose.")
    logging.info("This curated data is crucial for the next steps.")
    logging.info("-" * 30)
```

---

**4. `convert_to_smpl.py`**

```python
import os
import glob
import numpy as np
import logging
import json
from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Placeholder for VIBE/ROMP Integration ---
# This part requires installing and running VIBE or ROMP.
# Their output format needs to be parsed correctly.
# Below is a conceptual placeholder function.

def run_pose_estimator(input_frame_dir: str, output_smpl_dir: str, confidence_threshold: float):
    """
    Placeholder function simulating running VIBE/ROMP on extracted frames.

    In reality, you would:
    1. Install VIBE (https://github.com/mkocabas/VIBE) or
       ROMP (https://github.com/Arthur151/ROMP).
    2. Prepare the input frame structure they expect.
    3. Run their inference script (e.g., demo.py or inference.py).
    4. Parse their output files (often .pkl or .json containing SMPL params).

    This function just creates dummy output files for demonstration.
    """
    logging.info(f"Simulating VIBE/ROMP processing for frames in: {input_frame_dir}")
    os.makedirs(output_smpl_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(input_frame_dir, "*.png")))
    if not frame_files:
        logging.warning(f"No frames found in {input_frame_dir}")
        return

    output_data = {}
    high_conf_count = 0

    for i, frame_path in enumerate(frame_files):
        frame_basename = os.path.splitext(os.path.basename(frame_path))[0]

        # Simulate VIBE/ROMP output for one person per frame
        # VIBE output often includes: pred_cam, orig_cam, pose (72), shape (10), joints3d
        # ROMP output can be similar, potentially with confidence scores.
        dummy_confidence = np.random.uniform(0.5, 1.0) # Simulate confidence

        if dummy_confidence >= confidence_threshold:
            smpl_pose = np.random.randn(72) * 0.2 # 72D pose (axis-angle)
            smpl_shape = np.random.randn(10) * 0.5 # 10D shape
            smpl_cam = np.array([0.9, 0.0, 0.0])  # Example camera translation (s, tx, ty)

            # Store data similar to how VIBE/ROMP might structure it per frame
            output_data[frame_basename] = {
                'pose': smpl_pose.tolist(),
                'shape': smpl_shape.tolist(),
                'cam': smpl_cam.tolist(),
                'confidence': dummy_confidence, # Add confidence if available
                'frame_path': frame_path # Keep track of original frame
            }
            high_conf_count += 1
        else:
             logging.debug(f"Skipping frame {frame_basename} due to low confidence ({dummy_confidence:.2f})")


    # Save the collected data for the sequence (e.g., video clip)
    # VIBE often saves one file per video; ROMP might save per frame or sequence
    # Let's save one npz file per processed video/directory for simplicity here
    output_npz_path = os.path.join(output_smpl_dir, f"{os.path.basename(input_frame_dir)}_smpl.npz")

    # Filter data for saving (only high confidence frames)
    smpl_poses_list = [data['pose'] for data in output_data.values()]
    smpl_shapes_list = [data['shape'] for data in output_data.values()]
    smpl_cams_list = [data['cam'] for data in output_data.values()]
    frame_paths_list = [data['frame_path'] for data in output_data.values()]
    confidences_list = [data['confidence'] for data in output_data.values()] # Store confidence

    if not smpl_poses_list:
         logging.warning(f"No high-confidence frames found for {input_frame_dir}. Skipping save.")
         return

    np.savez(output_npz_path,
             pose=np.array(smpl_poses_list),
             shape=np.array(smpl_shapes_list),
             cam=np.array(smpl_cams_list),
             frame_paths=np.array(frame_paths_list),
             confidences=np.array(confidences_list)) # Save confidence values

    logging.info(f"Saved {high_conf_count} high-confidence SMPL parameters (out of {len(frame_files)} frames) to {output_npz_path}")

    # Alternatively, save as JSON (can be very large)
    # output_json_path = os.path.join(output_smpl_dir, f"{os.path.basename(input_frame_dir)}_smpl.json")
    # try:
    #     with open(output_json_path, 'w') as f:
    #         json.dump(output_data, f, indent=4)
    #     logging.info(f"Saved filtered SMPL parameters to {output_json_path}")
    # except Exception as e:
    #     logging.error(f"Failed to save JSON {output_json_path}: {e}")


if __name__ == "__main__":
    config = load_config()

    FRAME_OUTPUT_DIR = config['frame_output_dir'] # Input for this script
    SMPL_OUTPUT_DIR = config['smpl_output_dir'] # Output for this script
    CONFIDENCE_THRESHOLD = config['vibe_romp_confidence_threshold']
    POSE_NAMES = config['pose_names'] # Use pose names from config

    os.makedirs(SMPL_OUTPUT_DIR, exist_ok=True)

    logging.info("Starting SMPL parameter conversion process...")

    # Iterate through each pose directory in the frame output directory
    for pose_name in POSE_NAMES:
        pose_frame_dir = os.path.join(FRAME_OUTPUT_DIR, pose_name)
        if not os.path.isdir(pose_frame_dir):
            logging.warning(f"Frame directory not found for pose: {pose_name}. Skipping.")
            continue

        logging.info(f"Processing pose: {pose_name}")

        # Iterate through each video's frame subdirectory within the pose directory
        video_frame_dirs = [d for d in glob.glob(os.path.join(pose_frame_dir, '*')) if os.path.isdir(d)]
        if not video_frame_dirs:
            logging.warning(f"No video subdirectories found in {pose_frame_dir}")
            continue

        pose_smpl_output_dir = os.path.join(SMPL_OUTPUT_DIR, pose_name)
        os.makedirs(pose_smpl_output_dir, exist_ok=True)

        for video_dir in video_frame_dirs:
            # Run the (placeholder) pose estimator for each video's frames
            run_pose_estimator(video_dir, pose_smpl_output_dir, CONFIDENCE_THRESHOLD)

    logging.info("SMPL parameter conversion simulation complete.")
    logging.info(f"Results (potentially simulated) saved in: {SMPL_OUTPUT_DIR}")
    logging.info("NOTE: This script uses a PLACEHOLDER for VIBE/ROMP. You need to integrate the actual tool.")
```

---

**5. `generate_pose_pairs.py`**

```python
import os
import glob
import numpy as np
import random
import logging
from utils import load_config, SMPL_JOINT_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perturb_smpl_pose(smpl_pose_aa: np.ndarray, perturbation_type: str, factor: float) -> Tuple[np.ndarray, str]:
    """
    Applies a synthetic perturbation to an SMPL pose (axis-angle format).
    Returns the perturbed pose and a description of the perturbation.
    Assumes smpl_pose_aa is a 72D vector (24 joints * 3 axis-angle params).
    """
    perturbed_pose = smpl_pose_aa.copy()
    description = "Unknown perturbation"

    # SMPL joints relevant to common yoga mistakes (indices from SMPL_JOINT_MAP)
    left_knee_idx = SMPL_JOINT_MAP['left_knee'] * 3
    right_knee_idx = SMPL_JOINT_MAP['right_knee'] * 3
    left_elbow_idx = SMPL_JOINT_MAP['left_elbow'] * 3
    right_elbow_idx = SMPL_JOINT_MAP['right_elbow'] * 3
    left_shoulder_idx = SMPL_JOINT_MAP['left_shoulder'] * 3
    right_shoulder_idx = SMPL_JOINT_MAP['right_shoulder'] * 3
    # Add more joints as needed (e.g., hips, spine)

    if perturbation_type == "knee_bend":
        # Introduce bend primarily around one axis (e.g., x-axis for knee flexion)
        knee_idx = random.choice([left_knee_idx, right_knee_idx])
        perturbed_pose[knee_idx] += factor # Add to axis-angle param (simplistic, might need axis selection)
        side = "left" if knee_idx == left_knee_idx else "right"
        description = f"Increased bend in {side} knee"
    elif perturbation_type == "shoulder_raise":
        # Modify shoulder rotation (e.g., around z-axis for elevation/depression)
        shoulder_idx = random.choice([left_shoulder_idx, right_shoulder_idx])
        perturbed_pose[shoulder_idx + 2] += factor # Modify z-axis rotation
        side = "left" if shoulder_idx == left_shoulder_idx else "right"
        description = f"Raised {side} shoulder"
    elif perturbation_type == "elbow_bend":
        # Similar to knee bend
        elbow_idx = random.choice([left_elbow_idx, right_elbow_idx])
        perturbed_pose[elbow_idx] += factor # Add to axis-angle param
        side = "left" if elbow_idx == left_elbow_idx else "right"
        description = f"Increased bend in {side} elbow"
    # Add more perturbation types:
    # - spine curvature change (spine1, spine2, spine3)
    # - hip rotation (left_hip, right_hip)
    # - foot/ankle orientation (left_ankle, right_ankle, left_foot, right_foot)
    else:
        logging.warning(f"Unknown perturbation type: {perturbation_type}")
        # Apply random noise as fallback
        noise = (np.random.rand(72) - 0.5) * factor * 0.1 # Small random noise
        perturbed_pose += noise
        description = f"Applied random noise factor {factor*0.1:.2f}"


    # Clip pose parameters to a reasonable range if needed (optional)
    # perturbed_pose = np.clip(perturbed_pose, -np.pi, np.pi) # Simple clipping example

    return perturbed_pose, description


if __name__ == "__main__":
    config = load_config()

    SMPL_INPUT_DIR = config['smpl_output_dir'] # Input: Where SMPL params are stored
    POSE_PAIR_OUTPUT_DIR = config['pose_pair_output_dir'] # Output: Where pairs will be saved
    PERTURBATION_CONFIG = config['perturbation_factors']
    NUM_PERTURBATIONS = config['num_perturbations_per_pose']
    POSE_NAMES = config['pose_names']

    os.makedirs(POSE_PAIR_OUTPUT_DIR, exist_ok=True)

    logging.info("Starting Pose Pair (A/B) generation...")

    total_pairs_generated = 0

    for pose_name in POSE_NAMES:
        pose_smpl_dir = os.path.join(SMPL_INPUT_DIR, pose_name)
        if not os.path.isdir(pose_smpl_dir):
            logging.warning(f"SMPL directory not found for pose: {pose_name}. Skipping.")
            continue

        logging.info(f"Generating pairs for pose: {pose_name}")
        pose_pair_pose_dir = os.path.join(POSE_PAIR_OUTPUT_DIR, pose_name)
        os.makedirs(pose_pair_pose_dir, exist_ok=True)

        smpl_files = glob.glob(os.path.join(pose_smpl_dir, "*.npz"))
        if not smpl_files:
             logging.warning(f"No SMPL .npz files found in {pose_smpl_dir}")
             continue

        pose_pairs_list = [] # Collect pairs for this pose

        for smpl_file in smpl_files:
            try:
                data = np.load(smpl_file, allow_pickle=True)
                poses_a = data['pose'] # Shape: (N_frames, 72)
                shapes = data['shape'] # Shape: (N_frames, 10)
                frame_paths = data.get('frame_paths', [f"frame_{i}" for i in range(len(poses_a))]) # Use get for backward compatibility

                logging.debug(f"Loaded {len(poses_a)} poses (PoseA) from {smpl_file}")

                for i in range(len(poses_a)):
                    pose_a = poses_a[i]
                    shape = shapes[i] # Keep shape consistent for A and B
                    frame_id = os.path.basename(frame_paths[i]) if i < len(frame_paths) else f"frame_{i}"
                    original_video_name = os.path.splitext(os.path.basename(smpl_file))[0].replace('_smpl', '')

                    # Generate multiple perturbations for each PoseA
                    for j in range(NUM_PERTURBATIONS):
                        # Choose a random perturbation type and factor
                        pert_type = random.choice(list(PERTURBATION_CONFIG.keys()))
                        pert_factor = PERTURBATION_CONFIG[pert_type]
                        # Add randomness to factor? e.g. * random.uniform(0.8, 1.2)
                        pert_factor_rand = pert_factor * random.uniform(0.7, 1.3)

                        pose_b, pert_desc = perturb_smpl_pose(pose_a, pert_type, pert_factor_rand)

                        pair_info = {
                            'pose_name': pose_name,
                            'original_video': original_video_name,
                            'frame_id': frame_id,
                            'pose_a_smpl': pose_a,
                            'pose_b_smpl': pose_b,
                            'shape_smpl': shape, # Store shape as it might be needed
                            'perturbation_type': pert_type,
                            'perturbation_description': pert_desc,
                            'perturbation_factor': pert_factor_rand,
                        }
                        pose_pairs_list.append(pair_info)
                        total_pairs_generated += 1

            except Exception as e:
                logging.error(f"Error processing SMPL file {smpl_file}: {e}")


        # Save all pairs for this pose into a single file
        if pose_pairs_list:
             output_path = os.path.join(pose_pair_pose_dir, f"{pose_name}_pairs.npz")
             # Need to carefully structure saving dicts in npz, might be better as jsonl or pickle
             # Let's save components separately for easier loading in npz
             np.savez(output_path,
                      pose_a_smpl=[p['pose_a_smpl'] for p in pose_pairs_list],
                      pose_b_smpl=[p['pose_b_smpl'] for p in pose_pairs_list],
                      shape_smpl=[p['shape_smpl'] for p in pose_pairs_list],
                      metadata=[{k: v for k, v in p.items() if 'pose' not in k and 'shape' not in k} for p in pose_pairs_list]
                     )
             logging.info(f"Saved {len(pose_pairs_list)} pose pairs for {pose_name} to {output_path}")


    logging.info(f"Pose pair generation complete. Total pairs generated: {total_pairs_generated}")
    logging.info(f"Pairs saved in subdirectories under: {POSE_PAIR_OUTPUT_DIR}")

```

---

**6. `generate_labels.py`**

```python
import os
import glob
import numpy as np
import logging
from typing import Dict, Tuple
from utils import load_config, SMPL_JOINT_MAP
# Potential dependency: SMPL model library (e.g., smplx) to get joint positions/rotations
# For simplicity, we'll approximate angle differences directly from axis-angle
# Or use a simplified kinematic tree if needed. Let's start simple.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pose Comparison Logic (Inspired by PoseScript corrective_data.py) ---

def get_angle_difference(pose_a_aa: np.ndarray, pose_b_aa: np.ndarray, joint_name: str) -> float:
    """
    Estimates the rotational difference for a specific joint between two SMPL poses.
    This is a simplification using the magnitude of the axis-angle difference.
    A more accurate method would involve converting axis-angle to rotation matrices
    and calculating the geodesic distance (angle of the relative rotation).
    """
    if joint_name not in SMPL_JOINT_MAP:
        logging.warning(f"Joint '{joint_name}' not found in SMPL map.")
        return 0.0

    joint_idx = SMPL_JOINT_MAP[joint_name] * 3
    aa_a = pose_a_aa[joint_idx : joint_idx + 3]
    aa_b = pose_b_aa[joint_idx : joint_idx + 3]

    # Simple difference in axis-angle vectors magnitude (approximation)
    diff_aa = aa_b - aa_a
    angle_diff_rad = np.linalg.norm(diff_aa)

    # More accurate (requires rotation library like scipy.spatial.transform):
    # from scipy.spatial.transform import Rotation as R
    # try:
    #     rot_a = R.from_rotvec(aa_a)
    #     rot_b = R.from_rotvec(aa_b)
    #     relative_rot = rot_b * rot_a.inv()
    #     angle_diff_rad = relative_rot.magnitude() # Geodesic distance
    # except Exception as e:
    #     # Handle potential issues with zero rotations etc.
    #     logging.debug(f"Rotation calculation error for joint {joint_name}: {e}")
    #     angle_diff_rad = np.linalg.norm(diff_aa) # Fallback


    return np.degrees(angle_diff_rad) # Return difference in degrees

# --- Feedback Generation Rules ---

# These rules are highly specific and need careful design based on common errors
# for each yoga pose. This is a simplified, generic example.
# Reference: https://github.com/naver/posescript/blob/main/src/text2pose/posefix/corrective_data.py

FEEDBACK_RULES = {
    "generic": [
        {"joint": "left_knee", "threshold": 15.0, "condition": "more_bent", "text": "Straighten your left leg more at the knee."},
        {"joint": "left_knee", "threshold": 15.0, "condition": "less_bent", "text": "Bend your left knee slightly more."},
        {"joint": "right_knee", "threshold": 15.0, "condition": "more_bent", "text": "Straighten your right leg more at the knee."},
        {"joint": "right_knee", "threshold": 15.0, "condition": "less_bent", "text": "Bend your right knee slightly more."},
        {"joint": "left_elbow", "threshold": 20.0, "condition": "more_bent", "text": "Straighten your left arm at the elbow."},
        {"joint": "left_elbow", "threshold": 20.0, "condition": "less_bent", "text": "Allow a slight bend in your left elbow."}, # Less common correction
        {"joint": "right_elbow", "threshold": 20.0, "condition": "more_bent", "text": "Straighten your right arm at the elbow."},
        {"joint": "right_elbow", "threshold": 20.0, "condition": "less_bent", "text": "Allow a slight bend in your right elbow."},
        {"joint": "left_shoulder", "threshold": 10.0, "condition": "raised", "text": "Relax your left shoulder down, away from your ear."},
        {"joint": "right_shoulder", "threshold": 10.0, "condition": "raised", "text": "Relax your right shoulder down, away from your ear."},
         # Add rules for spine, hips, head/neck alignment etc.
    ],
    "Warrior II": [
        # Specific Warrior II corrections override generic ones if applicable
        {"joint": "right_knee", "threshold": 15.0, "condition": "more_bent", "text": "Ensure your front (right) knee is stacked over the ankle, not bent too much."},
        {"joint": "right_knee", "threshold": 10.0, "condition": "less_bent", "text": "Deepen the bend in your front (right) knee, aiming for a 90-degree angle."},
        {"joint": "left_knee", "threshold": 10.0, "condition": "more_bent", "text": "Keep your back (left) leg straight and strong."},
        {"joint": "left_shoulder", "threshold": 10.0, "condition": "raised", "text": "Draw your left shoulder blade down your back in Warrior II."},
        {"joint": "right_shoulder", "threshold": 10.0, "condition": "raised", "text": "Draw your right shoulder blade down your back in Warrior II."},
        # Could add torso orientation, arm height etc.
    ],
    "Triangle Pose": [
         {"joint": "left_knee", "threshold": 10.0, "condition": "more_bent", "text": "Keep your front (left) leg straight but avoid locking the knee in Triangle Pose."},
         {"joint": "right_knee", "threshold": 10.0, "condition": "more_bent", "text": "Ensure your back (right) leg is straight and engaged."},
         # Add torso rotation, arm extension rules
    ],
    # Add specific rules for other poses...
}

def generate_feedback_for_pair(pose_a_smpl: np.ndarray, pose_b_smpl: np.ndarray, pose_name: str, angle_threshold: float) -> Tuple[str, List[str]]:
    """
    Compares PoseA and PoseB and generates textual feedback based on rules.
    """
    feedback_texts = []
    corrections_made = [] # Keep track of what was corrected

    # Use pose-specific rules if available, otherwise fallback to generic
    rules = FEEDBACK_RULES.get(pose_name, []) + FEEDBACK_RULES["generic"]
    applied_joints = set() # Avoid multiple corrections for the same joint from generic/specific overlap

    for rule in rules:
        joint = rule["joint"]
        threshold = rule.get("threshold", angle_threshold) # Use rule-specific or global threshold
        condition = rule["condition"]
        text_template = rule["text"]

        if joint in applied_joints:
            continue

        angle_diff = get_angle_difference(pose_a_smpl, pose_b_smpl, joint)

        correction_applied = False
        # Determine the nature of the difference (e.g., more bent, less bent, raised)
        # This requires a more sophisticated comparison than just angle magnitude.
        # For simplicity, let's assume the perturbation type hints at the condition.
        # A better approach involves comparing actual joint angles/positions derived from SMPL.

        # --- Simplified Logic based on angle difference magnitude ---
        if angle_diff > threshold:
            # This doesn't know the *direction* of the error (more bent vs less bent)
            # We need more info from the perturbation or direct angle calculation.
            # Let's assume for now threshold crossing implies the 'error' condition defined in the rule
            # This is a MAJOR simplification.
             if condition in ["more_bent", "less_bent", "raised", "lowered"]: # Check if the condition might match magnitude difference
                 feedback_texts.append(text_template)
                 corrections_made.append(f"{joint}_{condition}")
                 applied_joints.add(joint)
                 correction_applied = True


        # --- TODO: Implement more robust condition checking ---
        # Example: Requires SMPL forward kinematics to get joint angles
        # 1. Use an SMPL library (like smplx) to get 3D joint positions/rotations from pose_a, pose_b
        # 2. Calculate relevant joint angles (e.g., knee angle, elbow angle, shoulder elevation)
        # 3. Compare angle_a vs angle_b to determine the condition (more_bent, less_bent, etc.)
        # 4. Apply feedback if angle_b deviates significantly from angle_a in the specified way.


    if not feedback_texts:
        # If no specific rule triggered by significant difference, maybe provide general encouragement
        # or state that the pose looks okay based on the comparison.
        # return "Keep holding the pose, looking good.", [] # Or maybe no feedback if difference is small
        return "", [] # No feedback if differences are below thresholds

    # Combine multiple feedback points if necessary
    final_feedback = " ".join(feedback_texts)

    return final_feedback, corrections_made


if __name__ == "__main__":
    config = load_config()

    POSE_PAIR_INPUT_DIR = config['pose_pair_output_dir'] # Input: Pose pairs
    STRUCTURED_DATA_FILE = config['structured_dataset_file'] # Output component (will be used later)
    POSE_NAMES = config['pose_names']
    ANGLE_THRESHOLD = config['angle_threshold_degrees'] # Default threshold

    logging.info("Starting feedback label generation...")

    all_labeled_data = [] # We will store results here first, then save in structure_dataset.py

    for pose_name in POSE_NAMES:
        pose_pair_dir = os.path.join(POSE_PAIR_INPUT_DIR, pose_name)
        if not os.path.isdir(pose_pair_dir):
            logging.warning(f"Pose pair directory not found for pose: {pose_name}. Skipping.")
            continue

        pair_files = glob.glob(os.path.join(pose_pair_dir, "*.npz"))
        if not pair_files:
            logging.warning(f"No pose pair .npz files found in {pose_pair_dir}")
            continue

        logging.info(f"Generating labels for pose: {pose_name}")

        for pair_file in pair_files:
            try:
                data = np.load(pair_file, allow_pickle=True)
                poses_a = data['pose_a_smpl']
                poses_b = data['pose_b_smpl']
                shapes = data['shape_smpl']
                metadata_list = data['metadata'] # List of dictionaries

                logging.debug(f"Loaded {len(poses_a)} pairs from {pair_file}")

                for i in range(len(poses_a)):
                    pose_a = poses_a[i]
                    pose_b = poses_b[i]
                    shape = shapes[i]
                    metadata = metadata_list[i] # Get metadata for this pair

                    feedback_text, corrections = generate_feedback_for_pair(
                        pose_a, pose_b, pose_name, ANGLE_THRESHOLD
                    )

                    if feedback_text: # Only keep pairs where feedback was generated
                        labeled_entry = {
                            'pose_name': pose_name,
                            'pose_a_smpl': pose_a.tolist(), # Convert to list for JSON compatibility if needed later
                            'pose_b_smpl': pose_b.tolist(),
                            'shape_smpl': shape.tolist(),
                            'metadata': metadata, # Contains original video, frame_id, perturbation info
                            'feedback_text': feedback_text,
                            'corrections_made': corrections # Store which rules were triggered
                        }
                        all_labeled_data.append(labeled_entry)

            except Exception as e:
                logging.error(f"Error processing pair file {pair_file}: {e}", exc_info=True)


    logging.info(f"Generated feedback labels for {len(all_labeled_data)} pose pairs.")

    # --- NOTE ---
    # The generated labels are currently stored in the `all_labeled_data` list.
    # The next script (`smpl_to_mediapipe.py`) will convert poses, and then
    # `structure_dataset.py` will combine everything (MP poses + labels) into the final dataset file.
    # For now, we can optionally save this intermediate labeled data if needed for debugging.
    INTERMEDIATE_LABEL_FILE = "data/intermediate_labeled_smpl_data.jsonl" # Optional save
    if all_labeled_data:
         try:
            import jsonlines
            os.makedirs(os.path.dirname(INTERMEDIATE_LABEL_FILE), exist_ok=True)
            with jsonlines.open(INTERMEDIATE_LABEL_FILE, mode='w') as writer:
                writer.write_all(all_labeled_data)
            logging.info(f"Saved intermediate labeled data (with SMPL poses) to {INTERMEDIATE_LABEL_FILE}")
         except ImportError:
            logging.warning("`jsonlines` library not found. Skipping saving intermediate labeled data.")
         except Exception as e:
             logging.error(f"Could not save intermediate labeled data: {e}")

    logging.info("Feedback label generation process finished.")
    logging.warning("Label quality depends heavily on the accuracy of `generate_feedback_for_pair` and the defined RULES.")
```

---

**7. `smpl_to_mediapipe.py`**

```python
import os
import numpy as np
import logging
import joblib
from sklearn.linear_model import LinearRegression
# Requires SMPL library (e.g., smplx) to get 3D joint locations from SMPL params
# Requires paired SMPL & MediaPipe data for training the mapper
from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Placeholder for SMPL Forward Kinematics ---
# This requires a library like smplx
try:
    import torch
    import smplx
    SMPLX_SUPPORT = True
except ImportError:
    logging.warning("smplx library not found. SMPL forward kinematics will be simulated.")
    SMPLX_SUPPORT = False

SMPL_MODEL_PATH = "models/smpl" # Path to SMPL model files (e.g., basicModel_neutral_lbs_10_207_0_v1.0.0.pkl)

# --- SMPL to 3D Joints ---
# Global SMPL model instance to avoid reloading
smpl_model = None

def get_smpl_joints(smpl_pose: np.ndarray, smpl_shape: np.ndarray, smpl_model_path: str = SMPL_MODEL_PATH) -> np.ndarray:
    """
    Performs SMPL forward kinematics to get 3D joint locations.
    Input: smpl_pose (72,), smpl_shape (10,)
    Output: 3D joints (e.g., 24, 3) - using SMPL standard output
    """
    global smpl_model
    if not SMPLX_SUPPORT:
        logging.debug("Simulating SMPL joints - returning random data.")
        # Return shape compatible with SMPL output (e.g., 24 joints)
        return np.random.rand(24, 3)

    if smpl_model is None:
        try:
            # Ensure model path exists
            if not os.path.exists(os.path.join(smpl_model_path, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')):
                 logging.error(f"SMPL model file not found in {smpl_model_path}. Download from SMPL website.")
                 raise FileNotFoundError("SMPL model file missing.")

            smpl_model = smplx.SMPL(model_path=smpl_model_path, gender='neutral', create_transl=False)
            logging.info(f"Loaded SMPL model from {smpl_model_path}")
        except Exception as e:
            logging.error(f"Failed to load SMPL model: {e}")
            # Fallback to simulation if model loading fails
            logging.warning("Falling back to simulated SMPL joints due to model load error.")
            return np.random.rand(24, 3)


    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smpl_model.to(device)

        with torch.no_grad():
            pose_tensor = torch.tensor(smpl_pose, dtype=torch.float32).unsqueeze(0).to(device)
            shape_tensor = torch.tensor(smpl_shape, dtype=torch.float32).unsqueeze(0).to(device)

            smpl_output = smpl_model(betas=shape_tensor, body_pose=pose_tensor[:, 3:], global_orient=pose_tensor[:, :3])
            joints_3d = smpl_output.joints.squeeze(0).cpu().numpy() # Get (N_joints, 3)

            # SMPL output might have more joints (e.g., 45 including face). Select the standard 24 body joints if needed.
            if joints_3d.shape[0] > 24:
                joints_3d = joints_3d[:24, :] # Assuming first 24 are standard body joints

            return joints_3d # Shape (24, 3)
    except Exception as e:
        logging.error(f"Error during SMPL forward kinematics: {e}")
        return np.random.rand(24, 3) # Fallback


# --- Mapping Function ---
def train_smpl_to_mediapipe_mapper(smpl_joints_data: np.ndarray, mediapipe_landmarks_data: np.ndarray, model_save_path: str):
    """
    Trains a linear regression model to map SMPL 3D joints to MediaPipe 3D landmarks.
    Input:
        smpl_joints_data (N_samples, N_smpl_joints * 3)
        mediapipe_landmarks_data (N_samples, N_mp_joints * 3)
    Saves the trained model.
    """
    if smpl_joints_data.shape[0] != mediapipe_landmarks_data.shape[0]:
        raise ValueError("Number of samples for SMPL and MediaPipe data must match.")
    if smpl_joints_data.shape[0] < 10: # Need sufficient data
         raise ValueError("Insufficient data for training the mapper.")

    logging.info(f"Training SMPL to MediaPipe mapper with {smpl_joints_data.shape[0]} samples...")

    # Reshape data if necessary (ensure flattened N_samples, N_features)
    n_samples = smpl_joints_data.shape[0]
    smpl_flat = smpl_joints_data.reshape(n_samples, -1)
    mp_flat = mediapipe_landmarks_data.reshape(n_samples, -1)

    # Train Linear Regression model
    mapper = LinearRegression()
    mapper.fit(smpl_flat, mp_flat)

    # Evaluate (optional - simple score check)
    score = mapper.score(smpl_flat, mp_flat)
    logging.info(f"Linear Regression mapper trained. R^2 score on training data: {score:.4f}")

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(mapper, model_save_path)
    logging.info(f"Mapper model saved to {model_save_path}")
    return mapper


def apply_smpl_to_mediapipe_map(smpl_joints: np.ndarray, mapper_model) -> np.ndarray:
    """
    Applies the trained mapper to convert SMPL 3D joints to MediaPipe landmarks.
    Input: smpl_joints (N_smpl_joints, 3) or flattened (N_smpl_joints * 3,)
    Output: mediapipe_landmarks (N_mp_joints * 3,) flattened
    """
    smpl_flat = smpl_joints.flatten().reshape(1, -1) # Ensure 2D for prediction (1 sample)
    mp_flat_pred = mapper_model.predict(smpl_flat)
    return mp_flat_pred.flatten() # Return flattened landmarks


# --- Main Execution ---
if __name__ == "__main__":
    config = load_config()
    MAPPER_MODEL_PATH = config['smpl_mediapipe_map_model']
    # Path to where labeled data (SMPL poses + feedback) is stored (from generate_labels.py)
    LABELED_SMPL_DATA_PATH = "data/intermediate_labeled_smpl_data.jsonl"

    # --- Step 1: Prepare Training Data (CRUCIAL & HARD) ---
    # This step requires having a dataset where you have *both* SMPL parameters
    # *and* corresponding MediaPipe landmarks for the *same* frames, *aligned*.
    # This is non-trivial to obtain. Public datasets like AMASS or 3DPW might be
    # used as a starting point, but require processing to get both formats aligned.

    # --- Placeholder: Simulate loading paired data ---
    logging.warning("Using PLACEHOLDER paired data for SMPL->MediaPipe mapper training.")
    logging.warning("Replace this with loading your actual aligned dataset.")

    def load_placeholder_paired_data(num_samples=1000):
        # Simulate N samples
        all_smpl_joints = []
        all_mp_landmarks = []
        for _ in range(num_samples):
            # Simulate getting SMPL joints from random params
            dummy_pose = np.random.randn(72) * 0.3
            dummy_shape = np.random.randn(10) * 0.6
            smpl_joints = get_smpl_joints(dummy_pose, dummy_shape) # (24, 3)

            # Simulate corresponding MediaPipe landmarks (33, 3)
            # This relationship is complex; linear mapping is an approximation.
            # A real dataset is needed here.
            # Let's just create random data with the right shape for the placeholder.
            mp_landmarks = np.random.rand(33, 3)
            # Maybe add *some* correlation for the placeholder simulation
            if smpl_joints.shape == (24, 3): # If SMPL kinematics worked (or simulated correctly)
                 # Simplistic: copy some overlapping joints (e.g., shoulders, hips)
                 # This is NOT accurate but makes the placeholder less random.
                 smpl_lshoulder_idx = SMPL_JOINT_MAP['left_shoulder']
                 smpl_rshoulder_idx = SMPL_JOINT_MAP['right_shoulder']
                 smpl_lhip_idx = SMPL_JOINT_MAP['left_hip']
                 smpl_rhip_idx = SMPL_JOINT_MAP['right_hip']

                 mp_lshoulder_idx = 11
                 mp_rshoulder_idx = 12
                 mp_lhip_idx = 23
                 mp_rhip_idx = 24

                 if mp_lshoulder_idx < mp_landmarks.shape[0]: mp_landmarks[mp_lshoulder_idx] = smpl_joints[smpl_lshoulder_idx] * 1.1 + np.random.rand(3)*0.05
                 if mp_rshoulder_idx < mp_landmarks.shape[0]: mp_landmarks[mp_rshoulder_idx] = smpl_joints[smpl_rshoulder_idx] * 1.1 + np.random.rand(3)*0.05
                 if mp_lhip_idx < mp_landmarks.shape[0]: mp_landmarks[mp_lhip_idx] = smpl_joints[smpl_lhip_idx] * 0.9 + np.random.rand(3)*0.05
                 if mp_rhip_idx < mp_landmarks.shape[0]: mp_landmarks[mp_rhip_idx] = smpl_joints[smpl_rhip_idx] * 0.9 + np.random.rand(3)*0.05


            all_smpl_joints.append(smpl_joints.flatten()) # Flatten (72,)
            all_mp_landmarks.append(mp_landmarks.flatten()) # Flatten (99,)

        return np.array(all_smpl_joints), np.array(all_mp_landmarks)

    # Load or generate placeholder paired data
    try:
        # Ideally: Load your actual paired dataset here
        # smpl_train_data, mp_train_data = load_my_real_paired_data('path/to/my/data')
        smpl_train_data, mp_train_data = load_placeholder_paired_data(num_samples=2000) # Using placeholder
        logging.info(f"Loaded/Generated placeholder paired data: SMPL {smpl_train_data.shape}, MP {mp_train_data.shape}")
    except Exception as e:
        logging.error(f"Failed to load or generate paired data: {e}")
        # Decide how to proceed: exit, or try to load existing mapper?
        smpl_train_data, mp_train_data = None, None


    # --- Step 2: Train or Load the Mapper ---
    mapper = None
    if os.path.exists(MAPPER_MODEL_PATH):
        try:
            mapper = joblib.load(MAPPER_MODEL_PATH)
            logging.info(f"Loaded existing mapper model from {MAPPER_MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to load existing mapper model: {e}. Attempting to retrain.")
            mapper = None

    if mapper is None:
        if smpl_train_data is not None and mp_train_data is not None:
            try:
                # Ensure SMPL data used for training has the same dimensionality as output of get_smpl_joints
                # Placeholder assumes get_smpl_joints -> (24,3) -> flattened (72,)
                if smpl_train_data.shape[1] != 24 * 3:
                     logging.warning(f"Training SMPL data has unexpected shape {smpl_train_data.shape}. Expected (*, 72). Adjusting or check data source.")
                     # Attempt reshape or handle error
                     # For placeholder, we assume it's already (N, 72)

                # Ensure MP data shape is correct (N, 33*3) = (N, 99)
                if mp_train_data.shape[1] != 33 * 3:
                     logging.warning(f"Training MediaPipe data has unexpected shape {mp_train_data.shape}. Expected (*, 99). Adjusting or check data source.")
                     # Attempt reshape or handle error


                mapper = train_smpl_to_mediapipe_mapper(smpl_train_data, mp_train_data, MAPPER_MODEL_PATH)
            except Exception as e:
                logging.error(f"Failed to train the mapper: {e}", exc_info=True)
                logging.error("Cannot proceed without a mapper model.")
        else:
            logging.error("No training data available and no existing mapper found. Cannot proceed.")


    # --- Step 3: Apply Mapper (Example) ---
    # This part is mainly executed by structure_dataset.py, but we test here.
    if mapper is not None:
        logging.info("Testing the mapper with a sample SMPL pose...")
        # Get a sample SMPL pose and shape (e.g., from the labeled data)
        try:
            import jsonlines
            if os.path.exists(LABELED_SMPL_DATA_PATH):
                 with jsonlines.open(LABELED_SMPL_DATA_PATH, mode='r') as reader:
                    first_item = next(iter(reader))
                    sample_pose_a = np.array(first_item['pose_a_smpl'])
                    sample_shape = np.array(first_item['shape_smpl'])

                    # 1. Get 3D joints from SMPL params
                    smpl_joints_3d = get_smpl_joints(sample_pose_a, sample_shape) # (24, 3)

                    # 2. Apply the map
                    if smpl_joints_3d is not None and smpl_joints_3d.shape == (24,3):
                         predicted_mp_landmarks_flat = apply_smpl_to_mediapipe_map(smpl_joints_3d, mapper) # (99,)
                         logging.info(f"Successfully mapped sample SMPL joints to MediaPipe landmarks (shape: {predicted_mp_landmarks_flat.shape})")

                         # Optional: Check if shape is correct
                         if predicted_mp_landmarks_flat.shape != (99,):
                              logging.error(f"Mapping output has unexpected shape: {predicted_mp_landmarks_flat.shape}. Expected (99,)")

                    else:
                         logging.error("Could not get valid SMPL 3D joints for the sample.")

            else:
                 logging.warning(f"Intermediate labeled data file not found at {LABELED_SMPL_DATA_PATH}, skipping mapper test.")

        except ImportError:
             logging.warning("`jsonlines` not found. Skipping mapper test.")
        except StopIteration:
             logging.warning(f"Intermediate labeled data file {LABELED_SMPL_DATA_PATH} is empty. Skipping mapper test.")
        except Exception as e:
            logging.error(f"Error during mapper test: {e}")

    else:
        logging.error("Mapper model is not available. Conversion cannot be performed.")

    logging.info("SMPL to MediaPipe conversion script finished.")
    logging.warning("The quality of the conversion depends heavily on the training data and the mapper model (Linear Regression is a simplification).")

```

---

**8. `structure_dataset.py`**

```python
import os
import numpy as np
import logging
import jsonlines
import joblib
from tqdm import tqdm
from utils import load_config, normalize_pose
# Import functions from the previous script
from smpl_to_mediapipe import get_smpl_joints, apply_smpl_to_mediapipe_map, SMPL_MODEL_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    config = load_config()

    LABELED_SMPL_DATA_PATH = "data/intermediate_labeled_smpl_data.jsonl" # Input
    MAPPER_MODEL_PATH = config['smpl_mediapipe_map_model'] # Input: Trained mapper
    FINAL_DATASET_PATH = config['structured_dataset_file'] # Output: Final JSONL dataset
    SMPL_MODEL_DIR = SMPL_MODEL_PATH # Path to SMPL model files needed by get_smpl_joints

    logging.info("Starting dataset structuring: Converting SMPL pairs to MediaPipe format and combining with labels.")

    # --- Load the SMPL-to-MediaPipe Mapper ---
    if not os.path.exists(MAPPER_MODEL_PATH):
        logging.error(f"Mapper model not found at {MAPPER_MODEL_PATH}. Run smpl_to_mediapipe.py first.")
        exit()
    try:
        mapper = joblib.load(MAPPER_MODEL_PATH)
        logging.info(f"Loaded SMPL->MediaPipe mapper from {MAPPER_MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error loading mapper model: {e}")
        exit()

    # --- Load the Labeled SMPL Data ---
    if not os.path.exists(LABELED_SMPL_DATA_PATH):
        logging.error(f"Labeled SMPL data file not found at {LABELED_SMPL_DATA_PATH}. Run generate_labels.py first.")
        exit()

    try:
        with jsonlines.open(LABELED_SMPL_DATA_PATH, mode='r') as reader:
            labeled_smpl_data = list(reader)
        logging.info(f"Loaded {len(labeled_smpl_data)} entries from {LABELED_SMPL_DATA_PATH}")
    except Exception as e:
        logging.error(f"Error loading labeled SMPL data: {e}")
        exit()

    if not labeled_smpl_data:
        logging.warning("Input labeled data is empty. No dataset will be created.")
        exit()

    # --- Process and Convert Data ---
    final_dataset = []
    conversion_errors = 0
    normalization_errors = 0

    for entry in tqdm(labeled_smpl_data, desc="Structuring Dataset"):
        try:
            pose_a_smpl = np.array(entry['pose_a_smpl'])
            pose_b_smpl = np.array(entry['pose_b_smpl'])
            shape_smpl = np.array(entry['shape_smpl'])
            pose_name = entry['pose_name']
            feedback_text = entry['feedback_text']

            # 1. Convert Pose A (SMPL -> SMPL 3D Joints -> MediaPipe)
            joints_a_3d = get_smpl_joints(pose_a_smpl, shape_smpl, SMPL_MODEL_DIR)
            if joints_a_3d is None or joints_a_3d.shape[0] != 24: # Check if valid joints returned
                 logging.warning(f"Skipping entry: Failed to get valid 3D joints for PoseA (Frame: {entry.get('metadata',{}).get('frame_id', 'N/A')})")
                 conversion_errors += 1
                 continue
            pose_a_mp_flat = apply_smpl_to_mediapipe_map(joints_a_3d, mapper) # (99,)

             # 2. Convert Pose B (SMPL -> SMPL 3D Joints -> MediaPipe)
            joints_b_3d = get_smpl_joints(pose_b_smpl, shape_smpl, SMPL_MODEL_DIR) # Use same shape
            if joints_b_3d is None or joints_b_3d.shape[0] != 24:
                 logging.warning(f"Skipping entry: Failed to get valid 3D joints for PoseB (Frame: {entry.get('metadata',{}).get('frame_id', 'N/A')})")
                 conversion_errors += 1
                 continue
            pose_b_mp_flat = apply_smpl_to_mediapipe_map(joints_b_3d, mapper) # (99,)

            # 3. Normalize MediaPipe Poses (Important for model input)
            # Reshape to (33, 3) before normalization if needed by the function
            pose_a_mp_reshaped = pose_a_mp_flat.reshape(33, 3)
            pose_b_mp_reshaped = pose_b_mp_flat.reshape(33, 3)

            # Note: Normalization should ideally happen *before* training the model,
            # but doing it here ensures the stored dataset format is consistent.
            # The normalize_pose function returns a flattened array.
            try:
                 norm_pose_a_mp_flat = normalize_pose(pose_a_mp_reshaped) # Output: (99,)
                 norm_pose_b_mp_flat = normalize_pose(pose_b_mp_reshaped) # Output: (99,)
            except Exception as norm_e:
                 logging.warning(f"Skipping entry due to normalization error: {norm_e} (Frame: {entry.get('metadata',{}).get('frame_id', 'N/A')})")
                 normalization_errors +=1
                 continue


            # 4. Structure the final data row
            final_entry = {
                "pose_name": pose_name,
                "pose_a_mp_norm": norm_pose_a_mp_flat.tolist(), # Store normalized, flattened MP pose
                "pose_b_mp_norm": norm_pose_b_mp_flat.tolist(), # Store normalized, flattened MP pose
                # Optional: Store non-normalized versions too?
                # "pose_a_mp_raw": pose_a_mp_flat.tolist(),
                # "pose_b_mp_raw": pose_b_mp_flat.tolist(),
                "feedback_text": feedback_text
                # Add metadata back if needed for analysis?
                # "metadata": entry['metadata']
            }
            final_dataset.append(final_entry)

        except Exception as e:
            logging.error(f"Error processing entry: {entry.get('metadata', {}).get('frame_id', 'N/A')}. Error: {e}", exc_info=True)
            conversion_errors += 1

    logging.info(f"Processed {len(labeled_smpl_data)} entries.")
    logging.warning(f"Skipped {conversion_errors} entries due to SMPL->MP conversion errors.")
    logging.warning(f"Skipped {normalization_errors} entries due to normalization errors.")
    logging.info(f"Resulting dataset size: {len(final_dataset)} entries.")

    # --- Save the Final Dataset ---
    if final_dataset:
        try:
            output_dir = os.path.dirname(FINAL_DATASET_PATH)
            if output_dir: # Ensure directory exists only if path includes one
                 os.makedirs(output_dir, exist_ok=True)
            with jsonlines.open(FINAL_DATASET_PATH, mode='w') as writer:
                writer.write_all(final_dataset)
            logging.info(f"Final structured dataset saved to {FINAL_DATASET_PATH}")
        except Exception as e:
            logging.error(f"Failed to save the final dataset: {e}")
    else:
        logging.warning("No data to save in the final dataset.")

    logging.info("Dataset structuring complete.")

```

---

**9. `pose_feedback_model.py`**

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """ Standard Transformer Positional Encoding """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe) # Buffer makes it part of state_dict but not parameters

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PoseFeedbackModel(nn.Module):
    def __init__(self, num_pose_names: int, pose_embedding_dim: int,
                 input_pose_dim: int = 99, # 33 landmarks * 3 coords
                 d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 vocab_size: int = 1000, max_seq_len: int = 50):
        """
        Args:
            num_pose_names: Number of unique yoga poses (for embedding).
            pose_embedding_dim: Dimension for the pose name embedding.
            input_pose_dim: Dimension of the flattened input pose (e.g., 33*3=99).
            d_model: Dimension of the transformer model (embeddings, attention etc.).
            nhead: Number of attention heads in the transformer.
            num_encoder_layers: Number of layers in the transformer encoder.
            num_decoder_layers: Number of layers in the transformer decoder.
            dim_feedforward: Dimension of the feedforward network in transformer layers.
            dropout: Dropout rate.
            vocab_size: Size of the target vocabulary (for feedback text).
            max_seq_len: Maximum length of the generated feedback sequence.
        """
        super().__init__()

        self.input_pose_dim = input_pose_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # --- Input Pose Encoder ---
        # Simple MLP to project flattened pose to d_model
        self.pose_encoder_mlp = nn.Sequential(
            nn.Linear(input_pose_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )

        # --- Pose Type Embedding ---
        self.pose_name_embedding = nn.Embedding(num_pose_names, pose_embedding_dim)

        # --- Fusion Layer ---
        # Project pose embedding and fuse with encoded pose
        self.fusion_layer = nn.Linear(d_model + pose_embedding_dim, d_model)

        # --- Text Decoder Input Embedding ---
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # --- Output Layer ---
        self.output_fc = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers and embeddings
        for name, param in self.named_parameters():
            if param.dim() > 1: # Exclude biases and 1D params like LayerNorm weights
                 nn.init.xavier_uniform_(param)
            elif "embedding" in name:
                 nn.init.normal_(param, mean=0.0, std=0.02)


    def forward(self, pose_a_input: torch.Tensor, pose_name_idx: torch.Tensor,
                target_tokens: torch.Tensor, tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None):
        """
        Forward pass for training.

        Args:
            pose_a_input (Tensor): Batch of flattened, normalized PoseA landmarks. Shape: [batch_size, input_pose_dim]
            pose_name_idx (Tensor): Batch of indices for the pose names. Shape: [batch_size]
            target_tokens (Tensor): Batch of target feedback sequences (shifted right). Shape: [batch_size, seq_len]
            tgt_mask (Tensor, optional): Mask to prevent attention to future tokens. Shape: [seq_len, seq_len]
            tgt_key_padding_mask (Tensor, optional): Mask to ignore padding tokens. Shape: [batch_size, seq_len]

        Returns:
            Tensor: Output logits for each token in the sequence. Shape: [batch_size, seq_len, vocab_size]
        """
        batch_size = pose_a_input.size(0)
        seq_len = target_tokens.size(1)

        # 1. Encode Input Pose
        encoded_pose = self.pose_encoder_mlp(pose_a_input) # [batch_size, d_model]

        # 2. Get Pose Name Embedding
        pose_embed = self.pose_name_embedding(pose_name_idx) # [batch_size, pose_embedding_dim]

        # 3. Fuse Pose Representation and Pose Type Embedding
        fused_input = torch.cat((encoded_pose, pose_embed), dim=1) # [batch_size, d_model + pose_embedding_dim]
        memory = self.fusion_layer(fused_input) # [batch_size, d_model]
        # Unsqueeze memory to act as the single "encoder output" step for the decoder
        memory = memory.unsqueeze(1) # [batch_size, 1, d_model] - acts like constant context

        # 4. Prepare Decoder Input (Embed target tokens + Positional Encoding)
        # Target tokens are typically shifted right during training (teacher forcing)
        tgt_emb = self.decoder_embedding(target_tokens) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]
        tgt_emb = self.positional_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1) # Apply positional encoding [batch_size, seq_len, d_model]


        # 5. Transformer Decoder
        # Decoder expects memory as [target_seq_len, batch_size, d_model] if batch_first=False
        # Since our memory is constant context [batch_size, 1, d_model], we repeat it
        memory = memory.repeat(1, seq_len, 1) # [batch_size, seq_len, d_model] (repeating context for each step)

        # If using batch_first=True for decoder layer:
        # tgt shape: [batch_size, seq_len, d_model]
        # memory shape: [batch_size, memory_seq_len (1 here), d_model]
        # We need memory as [batch_size, 1, d_model]
        memory_for_decoder = memory[:, 0:1, :] # Use the single context step: [batch_size, 1, d_model]

        decoder_output = self.transformer_decoder(
             tgt=tgt_emb,                   # [batch_size, seq_len, d_model]
             memory=memory_for_decoder,     # [batch_size, 1, d_model]
             tgt_mask=tgt_mask,             # [seq_len, seq_len]
             tgt_key_padding_mask=tgt_key_padding_mask # [batch_size, seq_len]
             # memory_key_padding_mask=None # No padding in our simplified memory
        ) # Output shape: [batch_size, seq_len, d_model]


        # 6. Output Layer
        logits = self.output_fc(decoder_output) # [batch_size, seq_len, vocab_size]

        return logits

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a square mask for the sequence. Used for autoregressive decoding."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask # Shape: [sz, sz]


    @torch.no_grad()
    def predict(self, pose_a_input: torch.Tensor, pose_name_idx: torch.Tensor,
              tokenizer, device, max_len=50):
        """
        Generate feedback text for a given pose input during inference.
        Uses greedy decoding for simplicity.

        Args:
            pose_a_input (Tensor): Single flattened, normalized PoseA. Shape: [1, input_pose_dim]
            pose_name_idx (Tensor): Single index for the pose name. Shape: [1]
            tokenizer: Tokenizer instance with encode/decode and special tokens (<SOS>, <EOS>, <PAD>).
            device: The device ('cuda' or 'cpu').
            max_len (int): Maximum length of the generated sequence.

        Returns:
            str: Generated feedback text.
        """
        self.eval() # Set model to evaluation mode

        sos_token_id = tokenizer.word_to_idx['<SOS>']
        eos_token_id = tokenizer.word_to_idx['<EOS>']
        pad_token_id = tokenizer.word_to_idx['<PAD>']

        # Move inputs to device
        pose_a_input = pose_a_input.to(device)
        pose_name_idx = pose_name_idx.to(device)

        # --- Encode input pose and fuse --- (Same as in forward) ---
        encoded_pose = self.pose_encoder_mlp(pose_a_input)
        pose_embed = self.pose_name_embedding(pose_name_idx)
        fused_input = torch.cat((encoded_pose, pose_embed), dim=1)
        memory = self.fusion_layer(fused_input).unsqueeze(1) # [1, 1, d_model]

        # --- Autoregressive Decoding ---
        # Start with the <SOS> token
        decoder_input_ids = torch.tensor([[sos_token_id]], dtype=torch.long, device=device) # [1, 1]

        for _ in range(max_len - 1): # Max length constraint
            # Prepare decoder input embedding and positional encoding
            tgt_emb = self.decoder_embedding(decoder_input_ids) * math.sqrt(self.d_model) # [1, current_len, d_model]
            tgt_emb = self.positional_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1) # [1, current_len, d_model]

            # Generate causal mask for current length
            current_len = decoder_input_ids.size(1)
            tgt_mask = self.generate_square_subsequent_mask(current_len).to(device) # [current_len, current_len]

            # Get output from decoder
            decoder_output = self.transformer_decoder(
                 tgt=tgt_emb,           # [1, current_len, d_model]
                 memory=memory,         # [1, 1, d_model] - Broadcasts implicitly? Check docs. Let's keep it simple.
                 tgt_mask=tgt_mask      # [current_len, current_len]
             ) # Output shape: [1, current_len, d_model]

            # Get logits for the last token
            last_token_logits = self.output_fc(decoder_output[:, -1, :]) # [1, vocab_size]

            # Greedy decoding: choose the token with the highest probability
            next_token_id = torch.argmax(last_token_logits, dim=-1) # [1]

            # Append predicted token ID to the sequence
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id.unsqueeze(1)], # Append as [1, 1]
                 dim=1
            ) # Shape [1, current_len + 1]

            # Stop if <EOS> token is generated
            if next_token_id.item() == eos_token_id:
                break

        # Decode the generated token IDs (excluding <SOS>)
        output_ids = decoder_input_ids.squeeze(0).cpu().tolist()
        feedback_text = tokenizer.decode(output_ids)

        return feedback_text


# --- Example Usage (Initialization) ---
if __name__ == "__main__":
    from utils import load_config, SimpleTokenizer

    config = load_config()

    # --- Dummy Parameters ---
    NUM_POSES = len(config['pose_names'])
    POSE_DIM = 99 # 33 * 3
    VOCAB_SIZE = config['vocab_size'] # Example vocab size
    MAX_LEN = config['max_seq_len']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate Model ---
    model = PoseFeedbackModel(
        num_pose_names=NUM_POSES,
        pose_embedding_dim=config['embedding_dim'],
        input_pose_dim=POSE_DIM,
        d_model=config['hidden_dim'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['hidden_dim'] * 2, # Often 2x or 4x d_model
        dropout=0.1,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_LEN
    ).to(DEVICE)

    print(f"Model initialized successfully on {DEVICE}.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Test Forward Pass (Dummy Data) ---
    BATCH_SIZE = config['batch_size']
    dummy_pose_input = torch.randn(BATCH_SIZE, POSE_DIM).to(DEVICE)
    dummy_pose_idx = torch.randint(0, NUM_POSES, (BATCH_SIZE,)).to(DEVICE)
    # Target tokens: batch of sequences, e.g., [[1, 5, 12, 3, 0, 0], [1, 8, 2, 0, 0, 0]] (SOS, word, EOS, PAD...)
    dummy_target_tokens = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_LEN - 1)).to(DEVICE) # Exclude last prediction target
    # Prepend SOS token
    sos_tensor = torch.full((BATCH_SIZE, 1), 1, dtype=torch.long).to(DEVICE) # Assuming SOS=1
    dummy_target_tokens_with_sos = torch.cat([sos_tensor, dummy_target_tokens], dim=1) # [batch_size, max_len]

    # --- Prepare masks for Transformer Decoder ---
    # Target mask: prevents attending to future tokens
    tgt_seq_len = dummy_target_tokens_with_sos.size(1)
    transformer_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(DEVICE)

    # Padding mask: prevents attending to padding tokens (assuming PAD=0)
    # Shape: [batch_size, seq_len]. True where padded.
    padding_mask = (dummy_target_tokens_with_sos == 0)

    try:
         # The target for loss calculation should be shifted left relative to decoder input
         # Decoder input: <SOS> token1 token2 ... tokenN <EOS> <PAD>
         # Target output: token1 token2 ... tokenN <EOS> <PAD> <PAD>
         decoder_input = dummy_target_tokens_with_sos[:, :-1] # Input: <SOS> to <EOS>
         target_output = dummy_target_tokens_with_sos[:, 1:]  # Target: token1 to <PAD>
         tgt_mask_for_input = model.generate_square_subsequent_mask(decoder_input.size(1)).to(DEVICE)
         padding_mask_for_input = (decoder_input == 0)


         logits = model(dummy_pose_input, dummy_pose_idx, decoder_input,
                        tgt_mask=tgt_mask_for_input,
                        tgt_key_padding_mask=padding_mask_for_input)
         print("Forward pass successful. Output logits shape:", logits.shape) # Should be [batch_size, seq_len-1, vocab_size]

         # --- Test Prediction (Greedy Decode) ---
         print("\nTesting prediction:")
         tokenizer = SimpleTokenizer(max_vocab_size=VOCAB_SIZE) # Need a basic tokenizer
         # Build a tiny vocab for demo
         tokenizer.build_vocab(["straighten your knee", "bend elbow more", "lower shoulder"])

         sample_pose = torch.randn(1, POSE_DIM).to(DEVICE)
         sample_idx = torch.tensor([0]).to(DEVICE) # Pose index 0

         predicted_text = model.predict(sample_pose, sample_idx, tokenizer, DEVICE, max_len=MAX_LEN)
         print(f"Sample Pose Idx {sample_idx.item()} -> Predicted Text: '{predicted_text}'")

    except Exception as e:
        print(f"Error during forward pass or prediction test: {e}")
        import traceback
        traceback.print_exc()

```

---

**10. `train_model.py`**

```python
import os
import logging
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import time
# Use evaluate library for BLEU/ROUGE (preferred) or nltk
try:
    import evaluate
    rouge_metric = evaluate.load('rouge')
    bleu_metric = evaluate.load('bleu')
    METRICS_LIB = 'evaluate'
except ImportError:
    try:
         from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
         # ROUGE using NLTK is less direct, consider 'py-rouge' or stick to BLEU if 'evaluate' not available
         METRICS_LIB = 'nltk'
         logging.warning("evaluate library not found, using NLTK for BLEU. ROUGE calculation skipped.")
    except ImportError:
        logging.warning("Neither 'evaluate' nor 'nltk' found. BLEU/ROUGE metrics will be skipped.")
        METRICS_LIB = None

from utils import load_config, SimpleTokenizer, load_pytorch_model
from pose_feedback_model import PoseFeedbackModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset Class ---
class YogaFeedbackDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, config, pose_name_to_id):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = config['max_seq_len']
        self.pose_name_to_id = pose_name_to_id
        self.input_pose_dim = 99 # Hardcoded based on structure_dataset output

        logging.info(f"Loading dataset from {jsonl_path}...")
        try:
            with jsonlines.open(jsonl_path, mode='r') as reader:
                for item in tqdm(reader, desc="Loading data"):
                    try:
                        pose_a = np.array(item['pose_a_mp_norm'], dtype=np.float32)
                        # pose_b = np.array(item['pose_b_mp_norm']) # Not directly used as input
                        feedback = item['feedback_text']
                        pose_name = item['pose_name']

                        if pose_name not in self.pose_name_to_id:
                             logging.warning(f"Skipping entry with unknown pose name: {pose_name}")
                             continue

                        pose_name_id = self.pose_name_to_id[pose_name]

                        # Ensure pose_a has the correct dimension
                        if pose_a.shape[0] != self.input_pose_dim:
                            logging.warning(f"Skipping entry: Pose A has incorrect dimension {pose_a.shape[0]}, expected {self.input_pose_dim}. Frame: {item.get('metadata',{}).get('frame_id', 'N/A')}")
                            continue


                        # Tokenize feedback text
                        token_ids = self.tokenizer.encode(feedback, max_len=self.max_seq_len)

                        self.data.append({
                            'pose_a': torch.tensor(pose_a, dtype=torch.float32),
                            'pose_name_id': torch.tensor(pose_name_id, dtype=torch.long),
                            'feedback_tokens': torch.tensor(token_ids, dtype=torch.long)
                        })
                    except Exception as e:
                        logging.warning(f"Skipping problematic entry: {item.get('metadata',{}).get('frame_id', 'N/A')}. Error: {e}")

            logging.info(f"Loaded {len(self.data)} valid entries.")
        except FileNotFoundError:
            logging.error(f"Dataset file not found: {jsonl_path}")
            raise
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Collate Function for DataLoader ---
def collate_fn(batch, pad_token_id):
    poses_a = torch.stack([item['pose_a'] for item in batch])
    pose_name_ids = torch.stack([item['pose_name_id'] for item in batch])
    feedback_tokens = [item['feedback_tokens'] for item in batch]

    # Pad feedback sequences
    feedback_tokens_padded = pad_sequence(feedback_tokens, batch_first=True, padding_value=pad_token_id)

    return {
        'pose_a': poses_a,
        'pose_name_id': pose_name_ids,
        'feedback_tokens': feedback_tokens_padded
    }

# --- Evaluation Function ---
@torch.no_grad()
def evaluate_model(model, dataloader, criterion, tokenizer, device, config):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_references = []
    pad_token_id = tokenizer.word_to_idx['<PAD>']

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress_bar:
        poses_a = batch['pose_a'].to(device)
        pose_name_ids = batch['pose_name_id'].to(device)
        feedback_tokens = batch['feedback_tokens'].to(device) # [batch, seq_len]

        # Prepare inputs and targets for loss calculation (shifted)
        decoder_input_tokens = feedback_tokens[:, :-1] # <SOS> ... tokenN
        target_tokens = feedback_tokens[:, 1:]         # token1 ... <EOS> / <PAD>

        # Create masks
        tgt_mask = model.generate_square_subsequent_mask(decoder_input_tokens.size(1)).to(device)
        tgt_padding_mask = (decoder_input_tokens == pad_token_id)

        # Forward pass
        logits = model(poses_a, pose_name_ids, decoder_input_tokens,
                       tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
                       # [batch, seq_len-1, vocab_size]

        # Calculate loss - ignore padding in target
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
        # Mask out padding tokens for accurate loss calculation
        non_pad_mask = (target_tokens != pad_token_id).view(-1)
        masked_loss = (loss * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1) # Avoid division by zero

        total_loss += masked_loss.item()


        # --- Generate Predictions for Metrics ---
        # Use model.predict for generation (simpler than handling batch generation here)
        # This is slower but easier to implement for eval metrics
        for i in range(poses_a.size(0)):
            single_pose_a = poses_a[i:i+1] # Keep batch dim
            single_pose_name_id = pose_name_ids[i:i+1]
            reference_text = tokenizer.decode(feedback_tokens[i].cpu().tolist()) # Decode reference

            try:
                 # Use model's predict method for generation
                 prediction_text = model.predict(single_pose_a, single_pose_name_id, tokenizer, device, config['max_seq_len'])
                 all_predictions.append(prediction_text)
                 all_references.append([reference_text]) # BLEU expects list of references
            except Exception as e:
                logging.warning(f"Prediction failed for one sample during evaluation: {e}")
                # Append dummy values or skip? Let's append blanks to keep lists aligned
                all_predictions.append("")
                all_references.append([""])


    avg_loss = total_loss / len(dataloader)
    metrics = {"loss": avg_loss}

    # Calculate BLEU/ROUGE if libraries available and data exists
    if METRICS_LIB and all_predictions and all_references:
        try:
            if METRICS_LIB == 'evaluate':
                 bleu_score = bleu_metric.compute(predictions=all_predictions, references=all_references)
                 rouge_score = rouge_metric.compute(predictions=all_predictions, references=all_references)
                 metrics['bleu'] = bleu_score['bleu']
                 metrics['rouge'] = rouge_score # ROUGE dict (rouge1, rouge2, rougeL, rougeLsum)
            elif METRICS_LIB == 'nltk':
                 # NLTK BLEU - requires tokenized input usually
                 chencherry = SmoothingFunction().method1
                 bleu_scores = []
                 for pred, ref_list in zip(all_predictions, all_references):
                     pred_tok = pred.split() # Simple split
                     ref_tok_list = [r.split() for r in ref_list]
                     # Need list of references for sentence_bleu
                     bleu = sentence_bleu(ref_tok_list, pred_tok, smoothing_function=chencherry)
                     bleu_scores.append(bleu)
                 metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
                 # metrics['rouge'] = "N/A (NLTK)" # ROUGE needs separate setup

        except Exception as e:
            logging.error(f"Failed to compute BLEU/ROUGE metrics: {e}")


    return metrics, all_predictions[:5], all_references[:5] # Return metrics and some samples


# --- Training Loop ---
def train(config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")

    # --- Load Data & Tokenizer ---
    DATASET_PATH = config['structured_dataset_file']
    if not os.path.exists(DATASET_PATH):
         logging.error(f"Dataset file not found: {DATASET_PATH}. Please run previous steps.")
         return

    # Build tokenizer from scratch based on dataset (or load pre-built)
    logging.info("Building tokenizer vocabulary...")
    all_texts = []
    try:
        with jsonlines.open(DATASET_PATH, mode='r') as reader:
             all_texts = [item['feedback_text'] for item in reader]
    except FileNotFoundError: # Should be caught above, but double-check
         logging.error(f"Cannot build tokenizer, dataset file missing: {DATASET_PATH}")
         return
    except Exception as e:
         logging.error(f"Error reading dataset for tokenizer: {e}")
         return

    if not all_texts:
        logging.error("No text data found in dataset to build tokenizer.")
        return


    tokenizer = SimpleTokenizer(max_vocab_size=config['vocab_size'])
    tokenizer.build_vocab(all_texts)
    config['actual_vocab_size'] = tokenizer.vocab_size # Update config with actual size
    PAD_TOKEN_ID = tokenizer.word_to_idx['<PAD>']

    # Create pose name mapping
    pose_name_to_id = {name: i for i, name in enumerate(config['pose_names'])}
    NUM_POSES = len(config['pose_names'])

    # Split data (simple split for demonstration)
    # TODO: Implement proper train/validation split
    dataset = YogaFeedbackDataset(DATASET_PATH, tokenizer, config, pose_name_to_id)
    if len(dataset) == 0:
         logging.error("Dataset loaded 0 entries. Cannot train.")
         return

    # Example: 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
         logging.warning("Dataset too small for train/val split. Using all data for training.")
         # Adjust: use a small validation set if possible, or just train
         if len(dataset) > 10:
             val_size = max(1, int(0.1 * len(dataset))) # Min 10% or 1 sample for val
             train_size = len(dataset) - val_size
         else: # Very small dataset, just use for training, no validation
              train_size = len(dataset)
              val_size = 0

    logging.info(f"Dataset size: {len(dataset)}. Train: {train_size}, Validation: {val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, PAD_TOKEN_ID), num_workers=os.cpu_count()//2)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                collate_fn=lambda b: collate_fn(b, PAD_TOKEN_ID), num_workers=os.cpu_count()//2) if val_size > 0 else None

    # --- Initialize Model, Optimizer, Loss ---
    model = PoseFeedbackModel(
        num_pose_names=NUM_POSES,
        pose_embedding_dim=config['embedding_dim'],
        input_pose_dim=99, # From dataset structure
        d_model=config['hidden_dim'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['hidden_dim'] * 2,
        dropout=0.1,
        vocab_size=config['actual_vocab_size'],
        max_seq_len=config['max_seq_len']
    ).to(DEVICE)

    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")


    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_gamma'])
    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='none') # Use reduction='none' for manual masking

    # --- Training Loop ---
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)
    log_file = os.path.join(config['log_dir'], "training_log.txt")

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        num_batches = len(train_dataloader)

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Training]", leave=False)
        for batch in progress_bar:
            poses_a = batch['pose_a'].to(DEVICE)
            pose_name_ids = batch['pose_name_id'].to(DEVICE)
            feedback_tokens = batch['feedback_tokens'].to(DEVICE)

            # Prepare inputs and targets (shifted)
            decoder_input_tokens = feedback_tokens[:, :-1]
            target_tokens = feedback_tokens[:, 1:]

            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(decoder_input_tokens.size(1)).to(DEVICE)
            tgt_padding_mask = (decoder_input_tokens == PAD_TOKEN_ID) # True where padded

            optimizer.zero_grad()

            # Forward pass
            logits = model(poses_a, pose_name_ids, decoder_input_tokens,
                           tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
                           # Shape: [batch, seq_len-1, vocab_size]


            # Calculate loss - Need to mask padding in the target
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            non_pad_mask = (target_tokens != PAD_TOKEN_ID).view(-1)
            masked_loss = (loss * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1)


            # Backward pass and optimization
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            epoch_train_loss += masked_loss.item()
            progress_bar.set_postfix(loss=masked_loss.item())

        avg_train_loss = epoch_train_loss / num_batches
        scheduler.step() # Update learning rate

        # --- Validation ---
        val_metrics = {"loss": float('nan'), "bleu": 0.0, "rouge": {}}
        sample_predictions = []
        sample_references = []
        if val_dataloader:
            val_metrics, sample_predictions, sample_references = evaluate_model(
                model, val_dataloader, criterion, tokenizer, DEVICE, config
            )
            avg_val_loss = val_metrics['loss']
        else:
             avg_val_loss = float('nan') # No validation data


        epoch_duration = time.time() - start_time

        # --- Logging ---
        log_message = (
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val BLEU: {val_metrics.get('bleu', 0.0):.4f} | "
             # Add ROUGE scores if available (e.g., ROUGE-L)
            f"Val ROUGE-L: {val_metrics.get('rouge', {}).get('rougeL', 0.0):.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        logging.info(log_message)
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
            if sample_predictions:
                f.write("Sample Predictions:\n")
                for i in range(len(sample_predictions)):
                     f.write(f"  Ref: {sample_references[i][0]}\n") # Take first reference
                     f.write(f"  Pred: {sample_predictions[i]}\n")
                f.write("-" * 20 + "\n")


        # --- Save Model Checkpoint ---
        is_best = avg_val_loss < best_val_loss if not np.isnan(avg_val_loss) else False # Consider best based on val loss
        if not np.isnan(avg_val_loss) and is_best:
             best_val_loss = avg_val_loss
             save_path = os.path.join(config['model_output_dir'], "best_model.pth")
             torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config, # Save config used for this model
                'tokenizer_vocab': tokenizer.word_to_idx # Save vocab with model
             }, save_path)
             logging.info(f"Saved new best model to {save_path}")

        # Save checkpoint periodically
        if (epoch + 1) % config['save_checkpoint_epoch'] == 0:
            checkpoint_path = os.path.join(config['model_output_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config,
                'tokenizer_vocab': tokenizer.word_to_idx
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")


    logging.info("Training finished.")


if __name__ == "__main__":
    config = load_config()
    try:
        train(config)
    except Exception as e:
        logging.error("An error occurred during training.", exc_info=True)

```

---

**11. `yoga_live_feedback_app.py`**

```python
import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import logging
import os
try:
    import pyttsx3
    TTS_SUPPORT = True
except ImportError:
    logging.warning("pyttsx3 library not found. Text-to-speech feedback will be disabled.")
    TTS_SUPPORT = False

from utils import load_config, normalize_pose, draw_landmarks, draw_feedback, SimpleTokenizer, load_pytorch_model, MEDIAPIPE_JOINT_MAP
from pose_feedback_model import PoseFeedbackModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Load Configuration ---
    config = load_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.join(config['model_output_dir'], "best_model.pth") # Load the best model
    COOLDOWN_SECONDS = config['live_feedback_cooldown_seconds']
    TARGET_POSE_NAME = config['target_pose_for_live_app'] # The pose to correct against
    SHOW_VISUALS = config['show_visual_feedback']
    ENABLE_TTS = config.get('tts_engine', 'pyttsx3') == 'pyttsx3' and TTS_SUPPORT

    # --- Load Model and Tokenizer ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}. Train the model first.")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Load config and vocab from checkpoint if saved
        model_config = checkpoint.get('config', config) # Use saved config if available
        tokenizer_vocab = checkpoint.get('tokenizer_vocab', None)

        if tokenizer_vocab is None:
             logging.error("Tokenizer vocabulary not found in model checkpoint. Cannot proceed.")
             # Alternatively, rebuild tokenizer if dataset path is known and consistent
             # For now, require it in checkpoint.
             return

        tokenizer = SimpleTokenizer()
        tokenizer.word_to_idx = tokenizer_vocab
        tokenizer.idx_to_word = {v: k for k, v in tokenizer_vocab.items()}
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        model_config['actual_vocab_size'] = tokenizer.vocab_size # Ensure model uses correct vocab size

        # Ensure pose names map is available
        pose_name_to_id = {name: i for i, name in enumerate(model_config['pose_names'])}
        NUM_POSES = len(model_config['pose_names'])


        # Instantiate model with loaded config
        model = PoseFeedbackModel(
             num_pose_names=NUM_POSES,
             pose_embedding_dim=model_config['embedding_dim'],
             input_pose_dim=99, # Consistent with training data
             d_model=model_config['hidden_dim'],
             nhead=model_config['nhead'],
             num_encoder_layers=model_config['num_encoder_layers'],
             num_decoder_layers=model_config['num_decoder_layers'],
             vocab_size=model_config['actual_vocab_size'],
             max_seq_len=model_config['max_seq_len']
        )

        model = load_pytorch_model(MODEL_PATH, model, DEVICE) # Load state dict
        logging.info(f"Feedback model loaded successfully from {MODEL_PATH}")

    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
        return

    # --- Initialize MediaPipe Pose ---
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        min_detection_confidence=config['min_detection_confidence'],
        min_tracking_confidence=config['min_tracking_confidence'],
        model_complexity=config['mediapipe_model_complexity']
    )

    # --- Initialize TTS Engine ---
    tts_engine = None
    if ENABLE_TTS:
        try:
            tts_engine = pyttsx3.init()
            # Optional: Configure voice, rate, volume
            # voices = tts_engine.getProperty('voices')
            # tts_engine.setProperty('voice', voices[1].id) # Example: female voice
            tts_engine.setProperty('rate', 150) # Adjust speech rate
            logging.info("TTS engine initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize TTS engine: {e}")
            ENABLE_TTS = False # Disable TTS if init fails

    # --- Application State ---
    last_feedback_time = 0
    current_feedback_text = "Starting..."
    pose_buffer = [] # Optional: buffer frames for stability
    buffer_size = 3 # Example buffer size

    # --- Video Capture ---
    cap = cv2.VideoCapture(0) # Use webcam 0
    if not cap.isOpened():
        logging.error("Cannot open webcam.")
        return

    # --- Get Target Pose ID ---
    if TARGET_POSE_NAME not in pose_name_to_id:
        logging.error(f"Target pose '{TARGET_POSE_NAME}' not found in model's known poses: {list(pose_name_to_id.keys())}")
        # Fallback or exit? Let's fallback to the first pose.
        fallback_pose = list(pose_name_to_id.keys())[0]
        logging.warning(f"Falling back to target pose: '{fallback_pose}'")
        target_pose_id = torch.tensor([pose_name_to_id[fallback_pose]], dtype=torch.long).to(DEVICE)
    else:
         target_pose_id = torch.tensor([pose_name_to_id[TARGET_POSE_NAME]], dtype=torch.long).to(DEVICE)
         logging.info(f"Targeting corrections for pose: {TARGET_POSE_NAME}")


    # --- Main Loop ---
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Ignoring empty camera frame.")
                continue

            # Flip the frame horizontally for a later selfie-view display
            # Convert the BGR image to RGB.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect pose
            results = pose_detector.process(rgb_frame)

            # Prepare frame for drawing
            annotated_frame = frame.copy()

            if results.pose_landmarks:
                # --- Extract and Normalize Landmarks ---
                landmarks_proto = results.pose_landmarks
                landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_proto.landmark], dtype=np.float32) # Shape (33, 3)

                try:
                     normalized_landmarks_flat = normalize_pose(landmarks_np) # Output: (99,)
                     current_pose_tensor = torch.tensor(normalized_landmarks_flat, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Shape [1, 99]
                except Exception as e:
                     logging.warning(f"Normalization failed: {e}")
                     current_pose_tensor = None # Skip inference if normalization fails

                # --- Cooldown Logic & Inference ---
                current_time = time.time()
                if current_pose_tensor is not None and (current_time - last_feedback_time > COOLDOWN_SECONDS):
                    logging.debug("Cooldown passed, performing inference...")
                    last_feedback_time = current_time

                    # Perform inference
                    try:
                        generated_text = model.predict(current_pose_tensor, target_pose_id, tokenizer, DEVICE, model_config['max_seq_len'])

                        if generated_text and generated_text != current_feedback_text:
                            current_feedback_text = generated_text
                            logging.info(f"Feedback: {current_feedback_text}")

                            # Trigger TTS
                            if ENABLE_TTS and tts_engine:
                                # Ensure TTS engine isn't busy (optional, basic check)
                                if not tts_engine.isBusy():
                                     tts_engine.say(current_feedback_text)
                                     tts_engine.runAndWait() # Blocks until speech is done, consider threading for non-blocking
                                else:
                                     logging.warning("TTS engine is busy, skipping announcement.")

                    except Exception as e:
                         logging.error(f"Model prediction failed: {e}", exc_info=True)
                         current_feedback_text = "Error during prediction." # Show error on screen


                # --- Draw Landmarks (using original coordinates for visualization) ---
                if SHOW_VISUALS:
                    # Use MediaPipe's drawing utils with the protobuf landmarks for accuracy
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_frame,
                        landmarks_proto,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
            else:
                # Optional: Clear feedback if no pose detected for a while?
                # current_feedback_text = "No pose detected"
                pass


            # --- Display Feedback Text ---
            annotated_frame = draw_feedback(annotated_frame, current_feedback_text)

            # --- Show Frame ---
            cv2.imshow('Yoga Live Feedback', annotated_frame)

            # --- Exit Condition ---
            if cv2.waitKey(5) & 0xFF == ord('q'):
                logging.info("Exit requested.")
                break

    finally:
        # Release resources
        pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        if ENABLE_TTS and tts_engine:
            try:
                 # Clean up TTS engine resources if needed (depends on library)
                 pass
            except Exception as e:
                logging.error(f"Error shutting down TTS engine: {e}")

if __name__ == "__main__":
    main()
```

---

**12. `requirements.txt`**

```txt
opencv-python
numpy
pyyaml
pytube
# yt-dlp # Uncomment if using yt-dlp instead of pytube
mediapipe>=0.9.0 # Ensure a recent version for Pose API
torch>=1.8 # Or newer, specify torchvision/torchaudio if needed
scikit-learn
joblib
jsonlines
tqdm
pyttsx3 # For text-to-speech feedback (optional)
# evaluate # Preferred for BLEU/ROUGE (optional, needs transformers/datasets)
# nltk # Alternative for BLEU (optional)
# smplx # Required for accurate SMPL forward kinematics (optional, needs separate installation/setup)
scipy # Needed for rotation calculations if implementing accurate angle diff
```

---

**Explanation and Next Steps:**

1.  **File Structure:** Create a directory for your project and save each code block above into its corresponding file name (e.g., `data_collection.py`, `config.yaml`, etc.).
2.  **Dependencies:** Install the required Python packages using pip: `pip install -r requirements.txt`. Note that `smplx` often requires manual installation steps (check its repository). `evaluate` might pull in large dependencies like `transformers`.
3.  **Placeholders:**
    *   **YouTube URLs:** Replace the example URLs in `data_collection.py` with actual, relevant video links.
    *   **VIBE/ROMP Integration (`convert_to_smpl.py`):** This is the most significant placeholder. You *must* install either VIBE or ROMP, run it on the extracted frames, and modify the script to parse *their specific output format* (likely `.pkl` or `.json` files containing SMPL parameters and potentially confidence scores). The current script *simulates* this output.
    *   **SMPL Model (`smpl_to_mediapipe.py`, `generate_labels.py`):** You need to download the official SMPL model files (usually requiring registration on their website) and place them in the `models/smpl` directory (or update the path in `SMPL_MODEL_PATH`). If `smplx` isn't installed or the models aren't found, these scripts fall back to simulated/placeholder data.
    *   **SMPL<->MediaPipe Paired Data (`smpl_to_mediapipe.py`):** The script currently *simulates* paired data to train the linear mapper. You *need* a real dataset with aligned SMPL and MediaPipe landmarks for the same poses to train an effective mapper. This is a challenging data acquisition step. Using only linear regression is also a simplification; a neural mapper might yield better results but requires more complex training.
    *   **Feedback Rules (`generate_labels.py`):** The rules are very basic. You'll need to significantly expand and refine these based on anatomical knowledge and common yoga mistakes for *each specific pose*. This is critical for generating meaningful feedback. The current angle difference calculation is also approximate.
    *   **Tokenizer:** The `SimpleTokenizer` is basic. For better performance, consider using pre-trained tokenizers (like SentencePiece or Hugging Face's tokenizers) and adjusting the model's vocabulary size and embedding layer accordingly.
    *   **Model Architecture:** The `PoseFeedbackModel` uses a standard Transformer decoder. You might experiment with LSTMs or different MLP structures for the encoder based on performance. Hyperparameters in `config.yaml` should be tuned.
4.  **Running the Pipeline:**
    *   Run `data_collection.py` to download videos and extract initial frames. **Manually clean the frames.**
    *   Run VIBE/ROMP externally. Update `convert_to_smpl.py` to read its output and run it.
    *   Run `generate_pose_pairs.py`.
    *   Refine rules in `generate_labels.py` and run it.
    *   Acquire paired data, place SMPL models, and run `smpl_to_mediapipe.py` to train/save the mapper.
    *   Run `structure_dataset.py` to create the final training data.
    *   Run `train_model.py` to train the feedback model.
    *   Run `yoga_live_feedback_app.py` for the live application.
5.  **Iteration:** This is a complex system. Expect to iterate on data cleaning, pose estimation quality, feedback rule refinement, SMPL<->MediaPipe mapping accuracy, and model training/tuning.

