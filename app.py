import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO('models/best2.pt')


def process_image(image):
    results = model.predict(image)
    annotated_image = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return encoded_image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' in request.files:
        file = request.files['file']
        image = Image.open(file.stream)
        processed_image = process_image(image)

        return jsonify({'image': processed_image})

    elif 'image' in request.json:
        image_data = request.json['image'].split(
            ',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = process_image(image)

        return jsonify({'image': processed_image})

    return jsonify({'error': 'No image provided'}), 400


if __name__ == '__main__':
    app.run(debug=True)
