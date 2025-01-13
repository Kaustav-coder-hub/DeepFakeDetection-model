import os
import pandas as pd


test_videos_path='/kaggle/input/deepfake-detection-challenge/test_videos'
train_folder = test_videos_path
video_files = []
for f in os.listdir(train_folder):
    if f.endswith(('.mp4', '.avi', '.mov')):
        video_files.append(f)
print(video_files)
metadata_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
# Convert to dictionary with video names as keys and labels as values
meta_data = {k: v['label'] for k, v in metadata.items()}

video_file = "example_video.mp4"  # Define video_file
predicted_label = 0  # Define predicted_label
accuracy = 0.95  # Example value
precision = 0.90  # Example value
recall = 0.85  # Example value
f1 = 0.87  # Example value

print(f"\n{video_file}\n Predicted Label-{'Real' if predicted_label == 0 else 'Fake'} \n Accuracy-{accuracy}\nPrecision-{precision}\nRecall-{recall}\nF1 Score-{f1}")