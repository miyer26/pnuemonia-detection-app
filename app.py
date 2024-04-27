from flask import Flask, request, jsonify
from torchvision import transforms
import os
from flasgger import Swagger

from src.image_preprocessing import Transformation
from src.model_inference import Inference

app = Flask(__name__)
swagger = Swagger(app)

# Load the model and transformations
model_path = os.path.join('model', 'best_model_elastic.pth')
transforms = Transformation(model_id_timm="timm/tf_efficientnetv2_b0.in1k")
inference = Inference(model_path=model_path, transforms=transforms)

@app.route('/')
def home():
    """
    This is the home endpoint.

    ---
    responses:
        200:
            description: A welcome message
    """
    return 'Welcome to the inference service!'

@app.route('/predict', methods=['POST'])
def predict():
    """
    This predicts the label for an uploaded image

    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The image file to classify

    responses:
        200:
            description: classification label
    """
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Perform inference
    label = inference.predict(file)

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)