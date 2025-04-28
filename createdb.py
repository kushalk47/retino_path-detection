# create_db.py

# Import the Flask application instance and the SQLAlchemy database instance
# from your main application file (app.py)
try:
    from app import app, db, User # Also import User to ensure models are loaded
    print("Successfully imported app, db, and User from app.py")
except ImportError as e:
    print(f"Error importing app, db, or User from app.py: {e}")
    print("Please ensure app.py exists in the same directory and is configured correctly.")
    print("Also check for any syntax errors in app.py that prevent it from being imported.")
    exit(1) # Exit if import fails

# Ensure you are within the application context before creating tables
print("Entering application context...")
with app.app_context():
    print("Creating database tables...")
    # Create all database tables defined in models (like your User model)
    db.create_all()
    print("Database tables created successfully!")