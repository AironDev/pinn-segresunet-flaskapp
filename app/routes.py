import zipfile
from flask import Blueprint, request, render_template, jsonify, url_for
from monai.transforms import Compose, Activations, AsDiscrete
import io
import torch

from app.utils.inference import inference
from app.utils.preprocess import preprocess_multiple_images, preprocess_single_image
from app.utils.visualization import visualize_multiple_modality_prediction, visualize_single_modality_prediction

main = Blueprint('main', __name__)


@main.route('/')
def home():
    return render_template('index.html')

@main.route('/smodality-predict', methods=['POST'])
def upload_file():
    # Process the uploaded file and perform inference
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Preprocess and run inference
    input_tensor = preprocess_single_image(file)
    output_tensor = inference(input_tensor)
    
    # Visualize and save results
    image_filename = visualize_single_modality_prediction(input_tensor, output_tensor)

    # Pass the filename to the template
    return render_template('result.html', result_image=image_filename)

@main.route('/mmodality-predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    zip_file = request.files['file']
    if not zip_file or not zip_file.filename.endswith('.zip'):
        return jsonify({'error': 'Invalid file format. Please upload a ZIP file.'}), 400

    try:
        # Read the ZIP file into memory
        zip_bytes = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            # Preprocess the images from the zip archive
            image_tensor = preprocess_multiple_images(zip_ref)  # Pass the zip_ref to preprocess_images

            # Perform prediction
            with torch.no_grad():
                output = inference(image_tensor)
                output = output.cpu().numpy()
                response = {'prediction': output.tolist()}

            # Call visualization function
            image_filename = visualize_multiple_modality_prediction(image_tensor.cpu().numpy(), output)

            return render_template('result.html', result_image=image_filename)

            # return jsonify(response)
    except Exception as e:
        raise e
        # return jsonify({'error': str(e)}), 500
