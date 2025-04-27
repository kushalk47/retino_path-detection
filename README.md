

 Synopsis

This project is a simple web application designed to predict the presence and severity of Diabetic Retinopathy (DR) from uploaded fundus images. 
It utilizes a pre-trained deep learning model (like ResNet18, finetuned for DR detection) running on a Flask backend. 
Users can upload an eye image through a web interface, and the application will return a predicted class representing the stage of Diabetic Retinopathy (e.g., No DR, Mild, Moderate, Severe, Proliferative). 
The uploaded image is temporarily displayed alongside the prediction results. The application features a futuristic, dark-themed user interface.

---

## README: Diabetic Retinopathy Prediction Web App

# Diabetic Retinopathy Prediction Web App

A web application built with Flask and PyTorch to predict Diabetic Retinopathy from fundus images.

## Synopsis

*(See the Synopsis section above)*

## Features

* **Image Upload:** Users can easily upload fundus images in common formats (.png, .jpg, .jpeg).
* **AI Prediction:** Utilizes a deep learning model (requires `diabetic_retinopathy_model.pth`) to analyze the uploaded image.
* **Prediction Output:** Displays the predicted class (severity level) and a corresponding condition status.
* **Temporary Image Display:** Shows the uploaded image on the results page alongside the prediction (image file is removed from the server after display).
* **Futuristic UI:** Features a dark-themed interface with modern CSS styling and animations.

## Technologies Used

* **Backend:**
    * Python 3.x
    * Flask: Web framework
    * PyTorch: Deep learning framework
    * torchvision: For image transformations and models
    * Pillow (PIL): For image processing
* **Frontend:**
    * HTML5
    * CSS3

## Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    # Activate the virtual environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate

    # Install required Python packages
    pip install Flask torch torchvision Pillow
    ```
    *(Note: You might need to install PyTorch and torchvision separately depending on your OS and CUDA requirements. Refer to the official PyTorch website for specific instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))*

3.  **Obtain the Trained Model:**
    This project requires a pre-trained PyTorch model file named `diabetic_retinopathy_model.pth`.
    * **You need to provide this file.** If you trained your own model, ensure it's saved as `diabetic_retinopathy_model.pth` and matches the architecture expected in `app.py` (a ResNet18 with a 5-class output layer).
    * **Place the `diabetic_retinopathy_model.pth` file in the root directory of the project** (the same directory as `app.py`).

4.  **Create Required Directories:**
    Ensure the static directories for uploads exist. The Flask app attempts to create these, but you can manually check or create them:
    ```bash
    mkdir static
    mkdir static/uploads
    ```

## Usage

1.  **Run the Flask Application:**
    Navigate to the root directory of the project in your terminal (where `app.py` is located) and run:
    ```bash
    python app.py
    ```
    The application will start, and you should see output indicating the server is running, typically on `http://127.0.0.1:5000/`.

2.  **Access the Web App:**
    Open your web browser and go to the address:
    ```
    http://127.0.0.1:5000/
    ```

3.  **Upload and Predict:**
    * Use the file input field to select a fundus image from your computer.
    * Click the "Upload and Predict" button.
    * The page will update to show the uploaded image and the predicted Diabetic Retinopathy condition.

## File Structure

```
.
├── app.py                     # Flask application backend
├── diabetic_retinopathy_model.pth # Your trained PyTorch model file (REQUIRED)
├── templates/
│   └── home.html              # HTML template for the web interface
└── static/
    ├── styles.css             # Stylesheet for the application
    └── uploads/               # Directory where uploaded images are temporarily stored
```

## Future Enhancements

* Add more detailed information or visualizations (e.g., heatmaps).
* Implement user authentication and prediction history.
* Improve error handling and user feedback for invalid inputs or model errors.
* Add support for other image formats.
* Provide confidence scores for predictions.
* Containerize the application using Docker.
* Add comprehensive documentation.

