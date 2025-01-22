from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from torch import nn
import face_recognition
import os

from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Image transformation settings
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def frame_extract(video_path):
    vidObj = cv2.VideoCapture(video_path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def prepare_frames(video_path, sequence_length, transform):
    frames = []
    for i, frame in enumerate(frame_extract(video_path)):
        faces = face_recognition.face_locations(frame)
        try:
            top, right, bottom, left = faces[0]
            frame = frame[top:bottom, left:right, :]
        except:
            pass
        frames.append(transform(frame))
        if len(frames) == sequence_length:
            break
    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames.unsqueeze(0)

# Prediction function
def predict(model, video_tensor):
    with torch.no_grad():
        _, output = model(video_tensor.to(device))
        prediction = torch.argmax(output, dim=1)
        confidence = torch.max(torch.nn.functional.softmax(output, dim=1))
        return prediction.item(), confidence.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def uplaod():
    return render_template('upload.html')
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    video_path = f'static/uploaded_videos/{video_file.filename}'
    video_file.save(video_path)

    model_file_path = 'static/models/iteration2.pt'  # Replace with your model path
    sequence_length = 20  # Set your desired sequence length

    
    model = Model(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    model.eval()

    
    video_tensor = prepare_frames(video_path, sequence_length, train_transforms)

    
    prediction, confidence = predict(model, video_tensor)
    output = "REAL" if prediction == 1 else "FAKE"

    return jsonify({'output': output, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
