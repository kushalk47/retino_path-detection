from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms, models
import time
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- Flask Configuration ---
# Use a strong secret key for session management
app.config['SECRET_KEY'] = 'your_super_secret_key_here' # CHANGE THIS TO A RANDOM STRING IN PRODUCTION
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db' # SQLite database named site.db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Disable modification tracking

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Initialize Extensions ---
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # Route name for the login page if login_required is used
login_manager.login_message_category = 'info' # Category for the default login required message

# --- Ensure upload folder exists within static ---
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- User Model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False) # Store hashed passwords

    def __repr__(self):
        return f"User('{self.username}')"

# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    # Check if user_id is not None and is an integer before querying
    if user_id is not None and user_id.isdigit():
        return User.query.get(int(user_id))
    return None # Return None if user_id is invalid


# --- AI Model Prediction Function (from your original code) ---
# (Kept as is, assuming its internal logic is correct)
def predict_image(image_path, model_path='diabetic_retinopathy_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return -1 # Indicate an error

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return -1

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return -1

        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return -1

# --- Routes ---

@app.route('/')
def home():
    # current_user is available in the template due to Flask-Login
    # Flash messages are available due to import and usage in routes
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home')) # Redirect if already logged in

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
            # Keep the username in the form for user convenience
            return render_template('signup.html', username=username)

        # Hash the password before saving
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during signup. Please try again.', 'danger')
            print(f"Signup error: {e}") # Log the error for debugging
            return redirect(url_for('signup'))


    return render_template('signup.html') # Render blank form on GET

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home')) # Redirect if already logged in

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') # Check if 'remember me' is checked

        user = User.query.filter_by(username=username).first()

        # Check if user exists and password is correct
        if user and check_password_hash(user.password, password):
            login_user(user, remember=bool(remember)) # Log the user in
            next_page = request.args.get('next') # Get the page user tried to access
            flash(f'Welcome back, {user.username}!', 'success')
            # Redirect to next page if available, otherwise home
            return redirect(next_page if next_page else url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
            # Keep the username in the form for user convenience
            return render_template('login.html', username=username)


    return render_template('login.html') # Render blank form on GET

@app.route('/logout')
@login_required # Requires user to be logged in to access this route
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/upload', methods=['POST'])
@login_required # Protect this route - user must be logged in
def upload_file():
    # Ensure request method is POST, although the decorator handles GET redirect
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("No file selected.", 'warning')
            # When redirecting after an error, pass data via flash or query params if needed
            # For simplicity here, just flash and render home without results
            return render_template('home.html')

        file = request.files['file']
        timestamp = int(time.time())
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(file_path)
        except Exception as e:
            flash("Error saving file.", 'danger')
            print(f"Error saving file {filename}: {e}")
            return render_template('home.html')

        prediction_result = predict_image(file_path)

        if prediction_result == -1:
            if os.path.exists(file_path):
                os.remove(file_path) # Clean up the potentially problematic file
            flash("Error processing image or model.", 'danger')
            return render_template('home.html')

        condition_map = {
            0: "No Diabetic Retinopathy",
            1: "Mild Non-Proliferative Diabetic Retinopathy",
            2: "Moderate Non-Proliferative Diabetic Retinopathy",
            3: "Severe Non-Proliferative Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }
        condition_text = condition_map.get(prediction_result, "Unknown Condition")

        image_url = url_for('static', filename=f'uploads/{filename}')

        # Render the template directly with results and image_url
        return render_template('home.html',
                               prediction=prediction_result,
                               condition=condition_text,
                               image_url=image_url)
    else:
        # Should not be reached with @login_required on POST, but good practice
        return redirect(url_for('home'))


# --- Helper function to create the database ---
# Run this function ONCE to create the database tables.
# You can uncomment it, run app.py, then comment it back out.
def create_database():
    with app.app_context():
        db.create_all()
        print("Database tables created.")


if __name__ == '__main__':
    # IMPORTANT: Run create_database() ONCE to create the tables
    create_database() # Uncomment this line, run app.py once, then comment it back.

    # Ensure SECRET_KEY is set before running the app in production
    if app.config['SECRET_KEY'] == 'your_super_secret_key_here' and not app.debug:
         print("WARNING: SECRET_KEY is not set. Please change it for production.")
         # In production, you might want to exit or handle this differently

    app.run(debug=True) # Set debug=False for production