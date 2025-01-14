from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the model
model = load_model('./models/Deep-Fake_Detection_FINAL_ACC-0.9017_Efficientnet_model.pth')

# Define transformations (Data Augmentation and Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(20),  # Random rotation between -20 and 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Preprocess single image
def preprocess_img(image_path):
    img = Image.open(image_path)
    img = transform(img)  # Apply augmentation and transformation
    return np.expand_dims(img.numpy(), axis=0)

# Preprocess video and extract every 20th frame
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Extract every 20th frame
        if frame_count % 20 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame).resize((224, 224))  # Resize frame
            frame = transform(frame)  # Apply transformations
            frames.append(frame.numpy())  # Add the transformed frame to the list
    
    cap.release()
    return np.array(frames)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if file.filename.endswith(('.jpg', '.png')):
        # Process image
        data = preprocess_img(filepath)
        prediction = model.predict(data)[0][0]
    else:
        # Process video frames
        frames = preprocess_video(filepath)
        if frames.size > 0:  # Ensure frames were extracted
            frames = np.expand_dims(frames, axis=0)  # Add batch dimension for prediction
            prediction = model.predict(frames)[0][0]
        else:
            return render_template('result.html', result="No valid frames extracted", filename=file.filename)

    result = "Fake" if prediction > 0.5 else "Real"
    return render_template('result.html', result=result, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
