from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms, models
import time # Import time for unique filenames

app = Flask(__name__)

# --- Modification 1: Save uploads INSIDE the static folder ---
# This makes the uploaded images accessible via URL like /static/uploads/image.jpg
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists within static
# --- Modification 2: Ensure static/uploads directory exists ---
# Check if the static directory exists first, then the uploads directory within it
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def predict_image(image_path, model_path='diabetic_retinopathy_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Add try-except for model loading in case the file is missing ---
    try:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        # Use map_location='cpu' when loading models trained on GPU but running on CPU
        # Ensure the model file exists before attempting to load
        if not os.path.exists(model_path):
             print(f"Error: Model file not found at {model_path}")
             return -1 # Indicate an error

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model = model.to(device)
    except Exception as e: # Catch potential errors during model loading
        print(f"Error loading model: {e}")
        return -1 # Indicate an error


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Add try-except for image loading and processing ---
    try:
        # Ensure the image file exists before opening
        if not os.path.exists(image_path):
             print(f"Error: Image file not found at {image_path}")
             return -1 # Indicate an error

        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    except Exception as e: # Catch potential errors during image processing
        print(f"Error processing image {image_path}: {e}")
        return -1 # Indicate an error


@app.route('/')
def home():
    # --- Modification 3: Remove getting data from query params ---
    # Data will now be passed directly when rendering from /upload
    # These lines are removed:
    # prediction = request.args.get('prediction')
    # condition = request.args.get('condition')
    # return render_template('home.html', prediction=prediction, condition=condition)

    # Simply render the home page initially
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        # Render home page, possibly with an error message
        return render_template('home.html', error="No file selected.")

    file = request.files['file']
    # --- Use secure_filename and add a unique identifier ---
    # A simple timestamp helps prevent overwriting files with the same name
    timestamp = int(time.time())
    original_filename = secure_filename(file.filename)
    filename = f"{timestamp}_{original_filename}" # Prefix with timestamp

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # --- Add error handling for file saving ---
    try:
        file.save(file_path)
    except Exception as e:
        print(f"Error saving file {filename}: {e}")
        return render_template('home.html', error="Error saving file.")


    # --- Perform Prediction ---
    prediction_result = predict_image(file_path)

    # --- Handle prediction errors ---
    if prediction_result == -1:
         # Clean up the saved file if prediction failed
         if os.path.exists(file_path):
             os.remove(file_path) # Remove the potentially problematic file
         return render_template('home.html', error="Error processing image or model.")

    # Determine condition based on prediction (adjust as per your model's output)
    # Assuming 0 is no DR, and > 0 indicates some level of DR
    # You might want a more detailed mapping for different prediction levels (0-4)
    condition_map = {
        0: "No Diabetic Retinopathy",
        1: "Mild Non-Proliferative Diabetic Retinopathy",
        2: "Moderate Non-Proliferative Diabetic Retinopathy",
        3: "Severe Non-Proliferative Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }
    condition_text = condition_map.get(prediction_result, "Unknown Condition")


    # --- Modification 4: Generate the URL for the saved image ---
    # Flask's url_for helps generate the correct path to static files
    # The URL will look like /static/uploads/1678888888_myimage.jpg
    image_url = url_for('static', filename=f'uploads/{filename}')

    # --- Modification 5: Render the template directly with results and image_url ---
    return render_template('home.html',
                           prediction=prediction_result,
                           condition=condition_text,
                           image_url=image_url) # Pass the image URL to the template


if __name__ == '__main__':
    # Run in debug mode for development. Set debug=False for production.
    # Ensure debug=True only during development.
    app.run(debug=True)