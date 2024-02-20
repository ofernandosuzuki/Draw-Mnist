from flask import Flask, render_template, request, jsonify

from model.model_training_script.model import LeNeT
from torchvision import transforms
import torch

from PIL import Image
import numpy as np
import base64
import io

model = LeNeT()
model.load_state_dict(torch.load('./model/trained_models/LeNet.pth'))
model.eval()

app = Flask(__name__, template_folder='./app/templates', static_folder='./app/static')

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  image = data['image']
  image = base64.b64decode(image.split(',')[1])
  image = Image.open(io.BytesIO(image)).convert('L')

  transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor()
  ])
  
  image = transform(image)
  image = image.unsqueeze(0)

  with torch.no_grad():
    pred = model(image)
    pred = pred.argmax(1)

    return jsonify({
      'prediction': pred.item()
    })

if __name__ == '__main__':
  app.run(debug=True)
