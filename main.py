import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
import queue
import pyttsx3
import os # For checking image file existence
import torch
from PoseCheck import PoseCheck
import torch.nn as nn
import pandas as pd

from preprocessing import preprocess_data

# --- Configuration Constants ---
CONFIG_FILE = 'exercise_config.json'
EASY_MODE_TOLERANCE = 20  # Degrees tolerance for easy mode

# Display and Timing
DESCRIPTION_DISPLAY_TIME = 5  # seconds
IMAGE_DISPLAY_TIME = 5        # seconds
POSE_HOLD_SECONDS = 4         # seconds
FPS = 30                      # Target FPS for hold counter
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# TTS
TTS_DELAY = 0.5  # seconds delay after each TTS utterance

# Colors (BGR)
COLOR_CORRECT = (0, 255, 0)       # Green
COLOR_INCORRECT = (0, 0, 255)     # Red
COLOR_LANDMARK = (230, 230, 230)  # Light Gray
COLOR_TEXT = (255, 255, 255)      # White
COLOR_TEXT_BG = (0, 0, 0)         # Black (for text background)
COLOR_PROGRESS_BAR_BG = (100, 100, 100)
COLOR_PROGRESS_BAR_FG = (0, 200, 0)

# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Utility Functions ---
def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points (in degrees)."""
    p1_np, p2_np, p3_np = np.array(p1), np.array(p2), np.array(p3)
    ba = p1_np - p2_np
    bc = p3_np - p2_np
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0: return 0.0
    cos_angle = np.clip(dot_product / (mag_ba * mag_bc), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def create_placeholder_image(width=200, height=200):
    img = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)
    cv2.putText(img, "?", (int(width*0.3), int(height*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 200, 200), 5)
    return img

# --- TTS Helper Class ---
class TTSHelper:
    def __init__(self, delay=0.5):
        self.tts_queue = queue.Queue()
        self.delay = delay
        self.engine = None
        self.thread = None
        self.stop_event = threading.Event()
        try:
            self.engine = pyttsx3.init()
            # You can configure voice, rate etc. here if needed
            # voices = self.engine.getProperty('voices')
            # self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 160)
        except Exception as e:
            print(f"Error initializing TTS engine: {e}. TTS will be disabled.")
            self.engine = None

        if self.engine:
            self.thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.thread.start()

    def _tts_worker(self):
        while not self.stop_event.is_set():
            try:
                text_to_speak = self.tts_queue.get(timeout=0.1) # Non-blocking with timeout
                if text_to_speak is None: # Sentinel to stop
                    break
                print(f"TTS Speaking: {text_to_speak}")
                self.engine.say(text_to_speak)
                self.engine.runAndWait()
                time.sleep(self.delay) # Delay after speaking
                self.tts_queue.task_done()
            except queue.Empty:
                continue # No item in queue, loop again
            except Exception as e:
                print(f"Error in TTS worker: {e}")
                if self.tts_queue.unfinished_tasks > 0:
                    self.tts_queue.task_done() # Ensure queue doesn't deadlock

    def speak(self, text):
        if self.engine and self.thread and self.thread.is_alive():
            self.tts_queue.put(text)
        elif not self.engine:
            print(f"TTS (disabled) would say: {text}")


    def stop(self):
        if self.thread and self.thread.is_alive():
            print("Stopping TTS thread...")
            self.stop_event.set()
            self.tts_queue.put(None) # Send sentinel
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                print("Warning: TTS thread did not terminate gracefully.")
        print("TTSHelper stopped.")

# --- Pose Estimator Class ---
class PoseEstimator:
    def __init__(self, config_data):
        self.joint_definitions = config_data["joint_definitions"]
        self.pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True
        h, w = frame.shape[:2]
        return results.pose_landmarks, w, h
    
    def get_complementry_joint(self, joint_name):
        """
        Returns the complementary joint name for a given joint.
        For example, if the input is "left_shoulder", it returns "right_shoulder".
        """
        if joint_name.startswith("left_"):
            return joint_name.replace("left_", "right_")
        elif joint_name.startswith("right_"):
            return joint_name.replace("right_", "left_")
        else:
            return None


    def check_pose_angles(self, landmarks_mp, pose_criteria, frame_w, frame_h, tolerance):
        """
        Checks angles against criteria.
        Returns:
            - angle_details (list): List of dicts for each checked joint:
                {'name', 'angle', 'is_correct', 'feedback', 'p1', 'p2', 'p3', 'color'}
            - all_joints_correct (bool): True if all defined criteria met.
        """
        angle_details = {}
        all_joints_correct = True
        landmarks = landmarks_mp.landmark # Easier access

        for joint_name, criterion in pose_criteria.items():
            if joint_name not in self.joint_definitions:
                print(f"Warning: Joint '{joint_name}' in pose criteria but not in joint_definitions.")
                continue

            lm_info = self.joint_definitions[joint_name]["landmarks"]
            try:
                lm_A_obj = landmarks[getattr(mp_pose.PoseLandmark, lm_info["A"])]
                lm_B_obj = landmarks[getattr(mp_pose.PoseLandmark, lm_info["B"])]
                lm_C_obj = landmarks[getattr(mp_pose.PoseLandmark, lm_info["C"])]
            except (AttributeError, IndexError) as e:
                print(f"Error getting landmarks for joint {joint_name}: {e}")
                all_joints_correct = False # Missing landmarks means incorrect
                continue

            # Visibility check (optional but good)
            vis_threshold = 0.3
            if not (lm_A_obj.visibility > vis_threshold and \
                    lm_B_obj.visibility > vis_threshold and \
                    lm_C_obj.visibility > vis_threshold):
                current_angle = -1 # Indicate low visibility
                is_correct = False 
                feedback_msg = f"{joint_name.replace('_', ' ')} not clearly visible."
                color = COLOR_INCORRECT
                comp_joint=self.get_complementry_joint(joint_name)
                if comp_joint in angle_details:
                    if angle_details[comp_joint]["is_correct"] == False:
                        all_joints_correct = False 
                # for details in angle_details:
                #     if details["name"] == get_complementry_joint(joint_name):
                #         details["is_correct"] = False
                #         details["feedback"] = feedback_msg
                #         details["color"] = color
            else:
                p1 = (lm_A_obj.x * frame_w, lm_A_obj.y * frame_h)
                p2 = (lm_B_obj.x * frame_w, lm_B_obj.y * frame_h)
                p3 = (lm_C_obj.x * frame_w, lm_C_obj.y * frame_h)

                current_angle = calculate_angle(p1, p2, p3)
                min_angle, max_angle = criterion["angle_range"]
                
                is_correct = (min_angle - tolerance) <= current_angle <= (max_angle + tolerance)
                color = COLOR_CORRECT if is_correct else COLOR_INCORRECT
                feedback_msg = ""

                if not is_correct:
                    all_joints_correct = False
                    if current_angle < (min_angle - tolerance):
                        feedback_msg = criterion["feedback"]["below_min"]
                    else:
                        feedback_msg = criterion["feedback"]["above_max"]
            
            # Store points as integers for drawing
            p1_draw = (int(lm_A_obj.x * frame_w), int(lm_A_obj.y * frame_h))
            p2_draw = (int(lm_B_obj.x * frame_w), int(lm_B_obj.y * frame_h))
            p3_draw = (int(lm_C_obj.x * frame_w), int(lm_C_obj.y * frame_h))

            angle_details[joint_name]={
                "angle": current_angle,
                "is_correct": is_correct,
                "feedback": feedback_msg,
                "p1": p1_draw, "p2": p2_draw, "p3": p3_draw, # Points for drawing
                "color": color
            }
        return angle_details, all_joints_correct

# --- Exercise Coach Class ---
class ExerciseCoach:
    def __init__(self, config_data, tts_helper):
        self.config = config_data
        self.tts = tts_helper
        self.pose_estimator = PoseEstimator(config_data)

        self.sequence = self.config["sequence"]
        self.all_poses_data = self.config["poses"]
        self.loaded_images = {} # Cache for pose images

        self.current_pose_idx = -1 # Start before the first pose
        self.current_pose_name = None
        self.current_pose_data = None
        self.current_angle_details = {} # To store for drawing

        self.phase = "IDLE" # IDLE, DESCRIPTION, IMAGE, CORRECTION
        self.phase_start_time = 0
        self.hold_counter = 0
        self.pose_hold_frames = POSE_HOLD_SECONDS * FPS

        self.active_feedback_messages = [] # For on-screen text

        self.mp_landmarks_for_pose = None # Placeholder for landmarks

        self.placeholder_img = create_placeholder_image()
        self._preload_all_images()

    def _preload_all_images(self):
        print("Preloading pose images...")
        for pose_name, pose_data in self.all_poses_data.items():
            if "image_path" in pose_data and os.path.exists(pose_data["image_path"]):
                img = cv2.imread(pose_data["image_path"])
                if img is not None:
                    # Resize to a standard preview size if needed, e.g., for IMAGE phase
                    # For now, just store it. Resizing can happen in draw phase.
                    self.loaded_images[pose_name] = img
                else:
                    print(f"Warning: Could not load image for {pose_name} from {pose_data['image_path']}.")
                    self.loaded_images[pose_name] = self.placeholder_img.copy()
            else:
                if "image_path" in pose_data:
                     print(f"Warning: Image path not found for {pose_name}: {pose_data['image_path']}")
                self.loaded_images[pose_name] = self.placeholder_img.copy()
        print("Image preloading complete.")

    # def previous_pose(self):
    #     if self.current_pose_idx > 0:
    #         self.current_pose_idx -= 1
    #         self.start_next_pose()
    #     else:
    #         print("Already at the first pose.")

    def predict_from_model(self, model, landmarks_mp,frame    ):
        """
        Predicts the pose label from the model using the landmarks.
        Returns the predicted class index.
        """
        if landmarks_mp and landmarks_mp.landmark:
            landmarks = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks_mp.landmark])
            landmarks = landmarks.reshape(1, -1)

            landmark_np=np.array(preprocess_data(pd.DataFrame(landmarks)))
            model.eval()
            index_to_class={0: 'pranamasana',
                            1: 'hasta_uttanasana',
                            2: 'padahastasana_fold',
                            3: 'ashwa_sanchalanasana_R_leg_back',
                            4: 'dandasana_plank',
                            5: 'ashtanga_namaskara_eight_limbs',
                            6: 'bhujangasana_cobra',
                            7: 'adho_mukha_svanasana_down_dog',
                            8: 'ashwa_sanchalanasana_L_leg_back',
                            9: 'Could_not_find_pose'
                            }
                            
            with torch.no_grad():
                new_sample = torch.tensor(landmark_np, dtype=torch.float32).view(1, 4, 33)
                logits = model(new_sample)
                output = torch.softmax(logits, dim=1)
                for index,class_name in index_to_class.items():
                    print(f" {class_name} : {output[0][index].item()}")
                predicted_class = torch.argmax(output, dim=1).item()

            overlay = frame.copy() 
            text = f"Predicted: {index_to_class[predicted_class]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

            # Get image dimensions
            img_height, img_width = overlay.shape[:2]

            # Compute top-right corner position
            x = img_width - text_width - 20  # 20 pixels padding from right
            y = img_height-50  # 50 pixels from bottom

            # Draw the text
            cv2.putText(overlay, text, (x, y), font, scale, COLOR_TEXT, thickness, cv2.LINE_AA)

            # cv2.putText(overlay, f"Predicted: {index_to_class[predicted_class]}", (w-100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2, cv2.LINE_AA)

            return overlay
            # return self.all_poses_data[predicted_class]
        #     return index_to_class[predicted_class]
        # else:
        #     print("No landmarks detected for prediction.")
        #     return None
        return frame # No landmarks, return original frame






    def save_landmarks(self, landmarks_mp, pose_name ,file):
        """
        Saves the landmarks and pose name to a CSV file.
        Each line contains x, y, z, visibility for every landmark of each pose example.
        Goal is to create a dataset of landmarks for training.
        """
        if landmarks_mp and landmarks_mp.landmark:
            for landmark in landmarks_mp.landmark:
                file.write(f"{landmark.x},{landmark.y},{landmark.z},{landmark.visibility},")
            file.write(f"{pose_name}\n") # Add pose name at the end of the landmarks
        else:
            print("No landmarks detected to save.")

        



    def start_next_pose(self):
        self.current_pose_idx += 1
        if self.current_pose_idx >= len(self.sequence):
            self.phase = "COMPLETED"
            self.tts.speak("Workout completed! Well done.")
    
            print("Workout completed!")
            return

        self.current_pose_name = self.sequence[self.current_pose_idx]
        self.current_pose_data = self.all_poses_data[self.current_pose_name]
        self.current_angle_details = {}
        self.active_feedback_messages.clear()

        self.phase = "DESCRIPTION"
        self.phase_start_time = time.time()
        display_name = self.current_pose_data.get('display_name', self.current_pose_name)
        description = self.current_pose_data.get('description', "No description.")
        self.tts.speak(f"Next: {display_name}. {description}")

    def update(self, frame):
        if self.phase == "IDLE":
            self.start_next_pose() # Initialize first pose
            return
        if self.phase == "COMPLETED":
            return

        current_time = time.time()

        if self.phase == "DESCRIPTION":
            if current_time - self.phase_start_time > DESCRIPTION_DISPLAY_TIME:
                self.phase = "IMAGE"
                self.phase_start_time = current_time
                # TTS for image phase is optional, description already spoken
        
        elif self.phase == "IMAGE":
            if current_time - self.phase_start_time > IMAGE_DISPLAY_TIME:
                self.phase = "CORRECTION"
                self.phase_start_time = current_time # Reset for correction phase timing if needed
                self.hold_counter = 0
                display_name = self.current_pose_data.get('display_name', self.current_pose_name)
                self.tts.speak(f"Hold {display_name}.")
        
        elif self.phase == "CORRECTION":
            landmarks_mp, w, h = self.pose_estimator.get_landmarks(frame)
            self.mp_landmarks_for_pose = landmarks_mp # Store for saving for data collection
            self.active_feedback_messages.clear() # Clear previous frame's feedback

            if landmarks_mp:
                self.current_angle_details, all_correct = self.pose_estimator.check_pose_angles(
                    landmarks_mp, self.current_pose_data["criteria"], w, h, EASY_MODE_TOLERANCE
                )
                
                if all_correct:
                    self.hold_counter += 1
                    self.active_feedback_messages.append("Looking good!")
                    if self.hold_counter >= self.pose_hold_frames:
                        self.tts.speak("Great!")
                        # time.sleep(0.5) # Short pause before next
                        self.start_next_pose()
                        return # Avoid further processing for this frame
                else:
                    self.hold_counter = 0
                    # Collect feedback for incorrect joints
                    incorrect_feedbacks = [detail["feedback"] for detail in self.current_angle_details.values() if not detail["is_correct"] and detail["feedback"]]
                    if incorrect_feedbacks:
                        self.active_feedback_messages.extend(incorrect_feedbacks[:2]) # Show max 2 specific
                        # self.tts.speak(incorrect_feedbacks[0]) # Speak first correction - can be spammy
                    else:
                        self.active_feedback_messages.append("Adjust your pose.")


            else: # No landmarks detected
                self.hold_counter = 0
                self.current_angle_details = {} # Clear old drawing data
                self.active_feedback_messages.append("Cannot see you clearly.")
                # self.tts.speak("Cannot see you.") # Can be spammy

    def draw_overlay(self, frame,DEBUG_MODE=False):
        h, w = frame.shape[:2]
        overlay = frame.copy() # Work on a copy to avoid modifying original frame if not needed elsewhere

        # --- Draw Phase-Specific Info ---
        if self.phase == "DESCRIPTION":
            display_name = self.current_pose_data.get('display_name', self.current_pose_name)
            description = self.current_pose_data.get('description', "No description.")
            y_offset = 70
            cv2.putText(overlay, f"Get Ready: {display_name}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2, cv2.LINE_AA)
            y_offset += 60
            # Wrap description text
            for i, line in enumerate(self._wrap_text(description, 80)): # Wrap at 80 chars approx
                 cv2.putText(overlay, line, (50, y_offset + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

        elif self.phase == "IMAGE":
            pose_image = self.loaded_images.get(self.current_pose_name, self.placeholder_img)
            img_h, img_w = pose_image.shape[:2]
            
            # Scale image to fit, maintaining aspect ratio (e.g., fit within half screen width/height)
            scale_factor = min((w * 0.8) / img_w, (h * 0.8) / img_h)
            disp_w, disp_h = int(img_w * scale_factor), int(img_h * scale_factor)
            
            if disp_w > 0 and disp_h > 0: # Ensure valid dimensions
                resized_pose_image = cv2.resize(pose_image, (disp_w, disp_h))
                x_offset = (w - disp_w) // 2
                y_offset = (h - disp_h) // 2
                try:
                    overlay[y_offset:y_offset+disp_h, x_offset:x_offset+disp_w] = resized_pose_image
                except ValueError as e:
                    print(f"Error placing image on overlay: {e}. Image shape: {resized_pose_image.shape}, Slice: {y_offset}:{y_offset+disp_h}, {x_offset}:{x_offset+disp_w}")
            
            display_name = self.current_pose_data.get('display_name', self.current_pose_name)
            cv2.putText(overlay, f"Preview: {display_name}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2, cv2.LINE_AA)

        elif self.phase == "CORRECTION":
            # Draw pose name
            display_name = self.current_pose_data.get('display_name', self.current_pose_name)
            cv2.putText(overlay, f"Current: {display_name}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2, cv2.LINE_AA)

            if DEBUG_MODE:

                for angle_name in self.current_angle_details.keys():
                    cv2.putText(overlay, f"{angle_name}: {self.current_angle_details[angle_name]['angle']:.1f}°",
                                (50, 100 + list(self.current_angle_details.keys()).index(angle_name) * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.current_angle_details[angle_name]["color"], 2)



            # Draw landmarks (optional - can be noisy with lines)
            # if self.current_angle_details: # landmarks were processed
            #     landmarks_mp, _, _ = self.pose_estimator.get_landmarks(frame) # Re-get for drawing, inefficient
            #     if landmarks_mp:
            #          mp_drawing.draw_landmarks(
            #              overlay, landmarks_mp, mp_pose.POSE_CONNECTIONS,
            #              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


            # Draw lines for checked angles
            
            for detail in self.current_angle_details.values():
                cv2.line(overlay, detail["p1"], detail["p2"], detail["color"], 3)
                cv2.line(overlay, detail["p2"], detail["p3"], detail["color"], 3)
                cv2.circle(overlay, detail["p2"], 6, detail["color"], -1)
                # Optional: Display angle value near joint
                # cv2.putText(overlay, f"{detail['angle']:.0f}", detail["p2"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)


            # Draw progress bar
            bar_x, bar_y, bar_w, bar_h = 50, h - 100, 300, 30
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_PROGRESS_BAR_BG, -1)
            progress = min(1.0, self.hold_counter / self.pose_hold_frames)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), COLOR_PROGRESS_BAR_FG, -1)
            cv2.putText(overlay, f"Hold: {int(progress*100)}%", (bar_x + bar_w + 10, bar_y + bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            
            # Draw feedback messages
            fb_y = h - 150
            for i, msg in enumerate(self.active_feedback_messages):
                if not msg: continue
                cv2.putText(overlay, msg, (50, fb_y - i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INCORRECT, 2)
        
        elif self.phase == "COMPLETED":
             cv2.putText(overlay, "Workout Complete!", (w//2 - 250, h//2), cv2.FONT_HERSHEY_TRIPLEX, 2, COLOR_CORRECT, 3)


        # --- Draw FPS (always) ---
        # (Add your FPS calculation and drawing logic here if desired)

        # Blend overlay with original frame for a nice effect (optional)
        # cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame) 
        # For simplicity, we'll just return the overlay directly
        return overlay

    def _wrap_text(self, text, line_length):
        """Simple text wrapper."""
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= line_length:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())
        return lines


# --- Main Application ---
def main():
    print("Starting Exercise Coach...")
    model=PoseCheck(num_classes=10) # Initialize model with 4 classes
    model.load_state_dict(torch.load('final_model_weights.pth'))
    # model.load_state_dict(torch.load('final_model_weights_softmax.pth'))
    model.eval()


    # Load Configuration
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{CONFIG_FILE}'.")
        return

    # Initialize TTS
    tts_helper = TTSHelper(delay=TTS_DELAY)

    # Initialize Coach
    coach = ExerciseCoach(config_data, tts_helper)

    # OpenCV Video Capture
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video camera.")
        tts_helper.stop()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    
    # coach.start_next_pose() # Start the first pose sequence

    frame_count = 0
    start_time = time.time()
    display_fps = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1) # Mirror effect

        # Process and update coach state
        frame=coach.predict_from_model(model, coach.mp_landmarks_for_pose,frame) # Predict the pose from the model
        coach.update(frame)
        # Draw overlays onto the frame
        output_frame = coach.draw_overlay(frame,DEBUG_MODE=False)
        
        # FPS Calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            display_fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # print(f"Predicted pose: {output_frame}")

        
        cv2.putText(output_frame, f"FPS: {display_fps:.1f}", (output_frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)


        cv2.imshow('Exercise Coach', output_frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('n'): # For debugging: skip to next pose
            print("Skipping to next pose...")
            coach.start_next_pose()
        elif key == ord('s'):
            # Save landmarks to CSV for data collection
            with open("pose_landmarks_data_labeled.csv", 'a') as file_for_data_collection:
                coach.save_landmarks(coach.mp_landmarks_for_pose, coach.current_pose_name, file_for_data_collection)
                print(f"Landmarks saved for {coach.current_pose_name}.")


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tts_helper.stop()
    print("Application finished.")

if __name__ == "__main__":
    main()