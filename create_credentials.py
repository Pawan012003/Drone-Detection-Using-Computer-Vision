from cryptography.fernet import Fernet
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Load encryption key
if not os.path.exists("secret.key"):
    raise FileNotFoundError("secret.key not found. Please run generate_key.py first.")
key = open("secret.key", "rb").read()
fernet = Fernet(key)

# Initialize Firebase (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate(r'D:\Drone detection command files\firebase_key.json')
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Get user input
email = input("Enter your Gmail ID: ")
password = input("Enter your Gmail App Password: ")

# Encrypt credentials
encrypted_email = fernet.encrypt(email.encode()).decode()
encrypted_password = fernet.encrypt(password.encode()).decode()

# Store encrypted credentials in Firestore
db.collection('users').document(email).set({
    'email': encrypted_email,
    'app_password': encrypted_password
})

print("[INFO] Encrypted credentials saved to Firestore.")
