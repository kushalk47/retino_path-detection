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
login_manager.login_message = 'Please log in to access this page.' # Custom message

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
    # --- New Columns ---
    # Store the last prediction level (0-4)
    last_prediction_level = db.Column(db.Integer, nullable=True)
    # Store the recommended cure text
    recommended_cure = db.Column(db.String(200), nullable=True)
    # Store the URL of the last uploaded image for this user
    last_image_url = db.Column(db.String(100), nullable=True)


    def __repr__(self):
        return f"User('{self.username}')"

# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    # Check if user_id is not None and is a digit before querying
    if user_id is not None and user_id.isdigit():
        return User.query.get(int(user_id))
    return None # Return None if user_id is invalid


# --- AI Model Prediction Function ---
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

# --- Helper function to get condition text ---
def get_condition_text(prediction_level):
    condition_map = {
        0: "No Diabetic Retinopathy",
        1: "Mild Non-Proliferative Diabetic Retinopathy",
        2: "Moderate Non-Proliferative Diabetic Retinopathy",
        3: "Severe Non-Proliferative Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }
    # Return "Unknown Condition" if level is None or outside 0-4 range
    if prediction_level is None or prediction_level not in condition_map:
        return "Unknown Condition"
    return condition_map[prediction_level]

# --- Helper function to get recommended cure ---
def get_recommended_cure(prediction_level):
    cure_map = {
        0: "Maintain good blood sugar and blood pressure control. Regular eye exams are recommended.",
        1: "Maintain good blood sugar and blood pressure control. Regular dilated eye exams (usually annually) are important.",
        2: "Closer monitoring by an ophthalmologist is needed. Treatment options like laser therapy or anti-VEGF injections may be considered.",
        3: "Prompt treatment is necessary, often involving laser treatment (panretinal photocoagulation) and/or anti-VEGF injections. Close monitoring is crucial.",
        4: "Aggressive treatment is required, typically involving anti-VEGF injections, laser treatment, and potentially surgery (vitrectomy) to remove blood or scar tissue. Close follow-up is essential."
    }
     # Return a general recommendation if level is None or outside 0-4 range
    if prediction_level is None or prediction_level not in cure_map:
        return "Consult with an ophthalmologist for diagnosis and treatment plan."
    return cure_map[prediction_level]


# --- Routes ---

@app.route('/')
def home():
    # current_user is available in the template due to Flask-Login
    # Flash messages are available due to import and usage in routes

    # Fetch last prediction, cure, and image URL from the user object if logged in
    last_prediction_level = None
    recommended_cure = None
    last_image_url = None

    if current_user.is_authenticated:
        last_prediction_level = current_user.last_prediction_level
        recommended_cure = current_user.recommended_cure
        last_image_url = current_user.last_image_url


    return render_template('home.html',
                           # Pass the stored data to the template
                           prediction=last_prediction_level,
                           condition=get_condition_text(last_prediction_level),
                           recommended_cure=recommended_cure,
                           image_url=last_image_url # Pass the stored image URL
                           )

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('signup.html', username=username)

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        # Initialize new columns to None
        new_user = User(username=username, password=hashed_password,
                        last_prediction_level=None, recommended_cure=None, last_image_url=None)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during signup. Please try again.', 'danger')
            print(f"Signup error: {e}")
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user, remember=bool(remember))
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.username}!', 'success')
            # Redirect to next page if available and safe, otherwise home
            is_safe = True
            if next_page:
                 is_safe = next_page.startswith('/') and not next_page.startswith('//')

            return redirect(next_page if next_page and is_safe else url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
            return render_template('login.html', username=username)

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("No file selected.", 'warning')
            return render_template('home.html') # Render home without new results

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
            return render_template('home.html') # Render home without new results

        prediction_result = predict_image(file_path)

        # Generate the URL for the saved image regardless of prediction success for display
        image_url = url_for('static', filename=f'uploads/{filename}')

        if prediction_result == -1:
            # Do NOT remove the file here, we still want to display it with the error message
            # if os.path.exists(file_path):
            #     os.remove(file_path) # Removed this line
            flash("Error processing image or model.", 'danger')
            # Render home, passing the image URL so it can be displayed
            return render_template('home.html', image_url=image_url)


        # --- Determine condition text and recommended cure based on prediction ---
        condition_text = get_condition_text(prediction_result)
        recommended_cure_text = get_recommended_cure(prediction_result)

        # --- Save prediction, cure, and image URL to the current user's record ---
        current_user.last_prediction_level = prediction_result
        current_user.recommended_cure = recommended_cure_text
        current_user.last_image_url = image_url # Save the image URL
        try:
            db.session.commit()
            flash("Prediction saved to your profile.", "success")
        except Exception as e:
            db.session.rollback()
            flash("Could not save prediction to your profile.", "warning")
            print(f"Error saving prediction to user: {e}")


        # Render the template directly with results and image_url
        return render_template('home.html',
                               prediction=prediction_result,
                               condition=condition_text,
                               image_url=image_url, # Pass the image URL
                               recommended_cure=recommended_cure_text
                               )
    else:
         flash("Please log in to upload files.", "info")
         return redirect(url_for('login'))


# --- Helper function to create the database ---
# Run this function ONCE to create the database tables.
# You can uncomment it, run app.py, then comment it back out.
def create_database():
    with app.app_context():
        db.create_all()
        print("Database tables created.")


if __name__ == '__main__':
    # IMPORTANT: Run create_database() ONCE to create the tables
    # Uncomment the line below, run app.py once, then comment it back or remove it.
    create_database()

    # Ensure SECRET_KEY is set before running the app in production
    if app.config['SECRET_KEY'] == 'your_super_secret_key_here' and not app.debug:
         print("WARNING: SECRET_KEY is not set. Please change it for production.")

    app.run(debug=True)
