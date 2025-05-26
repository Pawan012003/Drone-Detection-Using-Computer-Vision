import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import firebase_admin
from firebase_admin import credentials, firestore
from cryptography.fernet import Fernet
import os

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(r'D:\Drone detection command files\firebase_key.json')
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load encryption key
if not os.path.exists("secret.key"):
    raise FileNotFoundError("secret.key not found. Please run generate_key.py first.")
key = open("secret.key", "rb").read()
fernet = Fernet(key)

def send_alert_to_users(alert_message):
    users_ref = db.collection('users').stream()

    for user_doc in users_ref:
        user_data = user_doc.to_dict()

        encrypted_email = user_data.get('email')
        encrypted_password = user_data.get('app_password')

        if not encrypted_email or not encrypted_password:
            print(f"Missing encrypted credentials for user {user_doc.id}")
            continue

        try:
            # Decrypt credentials
            sender_email = fernet.decrypt(encrypted_email.encode()).decode()
            sender_password = fernet.decrypt(encrypted_password.encode()).decode()

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = sender_email
            msg['Subject'] = "Drone Alert Notification"
            msg.attach(MIMEText(f"Alert: {alert_message}", 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, sender_email, msg.as_string())
            server.quit()

            print(f"Alert email sent to {sender_email}")
        except Exception as e:
            print(f"Failed to send alert to {user_doc.id}. Error: {e}")

# Example usage (call this function when drone detected):
# send_alert_to_users("Drone detected at 2025-05-26 14:30")
