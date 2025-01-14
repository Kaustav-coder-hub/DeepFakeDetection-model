from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import cv2
import os
import torch
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the PyTorch model
model = torch.load('./models/Deep-Fake_Detection_FINAL_ACC-0.9017_Efficientnet_model.pth')
model.eval()  # Set the model to evaluation mode


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
    img = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    img = transform(img)  # Apply transformations
    return img.unsqueeze(0)  # Add batch dimension


# Preprocess video and extract every 20th frame
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 20 == 0:  # Extract every 20th frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)  # Convert to PIL Image
            frame = transform(frame)  # Apply transformations
            frames.append(frame.numpy())  # Store as numpy array

    cap.release()
    if frames:
        return torch.tensor(np.stack(frames))  # Stack frames and convert to tensor
    return torch.empty(0)  # Return empty tensor if no frames


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

    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')):
        return render_template('result.html', result="Unsupported file type", filename=file.filename)

    try:
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            data = preprocess_img(filepath)
            with torch.no_grad():
                prediction = model.predict(data)[0][0]
        else:
            frames = preprocess_video(filepath)
            if frames.size == 0:
                return render_template('result.html', result="No valid frames extracted", filename=file.filename)
            with torch.no_grad():
                prediction = model.predict(frames).mean()
        
        result = "Fake" if prediction > 0.5 else "Real"
    except Exception as e:
        return render_template('result.html', result=f"Error processing file: {str(e)}", filename=file.filename)

    return render_template('result.html', result=result, filename=file.filename)


if __name__ == '__main__':
    app.run(debug=True)
