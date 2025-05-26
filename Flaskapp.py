from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, firestore
from cryptography.fernet import Fernet
import os

# Initialize Flask app and static folder
app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'

# Firebase initialization
FIREBASE_KEY_PATH = r"D:\Drone detection command files\firebase_key.json"
if not os.path.exists(FIREBASE_KEY_PATH):
    raise FileNotFoundError(f"Firebase key file not found at {FIREBASE_KEY_PATH}")

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load encryption key
SECRET_KEY_PATH = "secret.key"
if not os.path.exists(SECRET_KEY_PATH):
    raise FileNotFoundError("secret.key not found. Please run generate_key.py first.")
key = open(SECRET_KEY_PATH, "rb").read()
fernet = Fernet(key)

@app.route('/', methods=['GET', 'POST'])
def user_form():
    message = None
    message_category = None

    if request.method == 'POST':
        email = request.form.get('email')
        app_password = request.form.get('app_password')

        if not email or not app_password:
            message = "Please provide both email and app password"
            message_category = "error"
        else:
            try:
                # Encrypt credentials
                encrypted_email = fernet.encrypt(email.encode()).decode()
                encrypted_password = fernet.encrypt(app_password.encode()).decode()

                # Save encrypted data to Firestore using email as document ID
                db.collection('users').document(email).set({
                    'email': encrypted_email,
                    'app_password': encrypted_password
                })

                message = "Credentials saved successfully!"
                message_category = "success"

            except Exception as e:
                message = f"An error occurred: {str(e)}"
                message_category = "error"

    return render_template('form.html', message=message, category=message_category)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
