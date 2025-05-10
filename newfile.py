import csv
import os

DATA_COLLECTION_FILE = "pose_landmarks_data_labeled_2.csv" # New CSV name
NUM_LANDMARKS = 33
NUM_COORDS_PER_LANDMARK = 4

def prepare_header_for_csv():
    header = []
    for i in range(NUM_LANDMARKS):
        header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])
    header.append("pose_label \n")
    return header

file_exists = os.path.isfile(DATA_COLLECTION_FILE)

with open(DATA_COLLECTION_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(DATA_COLLECTION_FILE) == 0:
                header = prepare_header_for_csv()
                writer.writerow(header)