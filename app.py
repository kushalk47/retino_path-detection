from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms, models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def predict_image(image_path, model_path='diabetic_retinopathy_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

@app.route('/')
def home():
    prediction = request.args.get('prediction')
    condition = request.args.get('condition')
    return render_template('home.html', prediction=prediction, condition=condition)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('home'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prediction = predict_image(file_path)
    condition = "Diabetic Retinopathy" if prediction > 0 else "No Diabetic Retinopathy"

    # Redirect to home with prediction results as query params
    return redirect(url_for('home', prediction=prediction, condition=condition))

if __name__ == '__main__':
    app.run(debug=True)
