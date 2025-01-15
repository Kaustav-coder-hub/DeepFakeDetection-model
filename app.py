from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model_path = 'models\Deep-Fake_Detection_FINAL_ACC-0.9017_Efficientnet_model.pth'

# Load the trained EfficientNet model
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)  # Binary classification
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Ensure the 'uploads' directory exists
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Supported video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    
    # Supported image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    
    # Get the file extension
    file_extension = os.path.splitext(filepath)[1].lower()
    
    # Check if the file is a video
    if file_extension in video_extensions:
        result = analyze_video(filepath)
    
    # Check if the file is an image
    elif file_extension in image_extensions:
        result = analyze_image(filepath)
    
    # If the file is neither a video nor an image
    else:
        result = f"Error: Unsupported file type '{file_extension}'. Please upload a video or image."
    
    return render_template('result.html', result=result, filename=file.filename)

def analyze_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    output = model(input_tensor)
    prediction = torch.argmax(output, 1).item()
    return 'Fake' if prediction == 1 else 'Real'

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0
    total_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        if frame_count % 20 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            input_tensor = preprocess(image).unsqueeze(0)
            output = model(input_tensor)
            prediction = torch.argmax(output, 1).item()
            if prediction == 1:
                fake_count += 1
            total_frames += 1
    
    cap.release()
    if fake_count / total_frames > 0.5:
        return 'Fake'
    else:
        return 'Real'

if __name__ == '__main__':
    app.run(debug=True)
