from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import uuid
import numpy as np
import face_recognition
from flask_sqlalchemy import SQLAlchemy
import os
import base64

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:lcbisa88@173.212.232.47:3307/python_test'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    encoding = db.Column(db.BLOB, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

camera = None  # Global variable to store camera instance
known_face_encodings = []  # List to store known face encodings
image_path = "./static/image.jpg"  # Path to the image file containing known faces
known_face_names = ["Known Person"]  # List to store known face names
is_recognizing = False  # Flag to indicate if face recognition is running
user_new_encoding = None

def load_known_faces():
    global known_face_encodings, known_face_names, is_recognizing, user_new_encoding
    try:
        # Load the known image using OpenCV
        known_image = cv2.imread(image_path)

        # Convert to RGB format if necessary
        if known_image is None:
            raise ValueError("Image not found or unable to load.")
        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

        # Get face encodings
        known_face_encodings = face_recognition.face_encodings(known_image_rgb)
        
        if len(known_face_encodings) == 0:
            print("No face found in the image.")
        else:
            known_face_encodings = [known_face_encodings[0]]  # Take the first face encoding
    except Exception as e:
        print(f"Error loading known faces: {e}")

def load_known_faces_db():
    global known_face_encodings, known_face_names
    try:
        users = User.query.all()
        known_face_encodings = []
        known_face_names = []
        
        for user in users:
            # Convert BLOB to numpy array
            encoding_array = np.frombuffer(user.encoding, dtype=np.float64)
            known_face_encodings.append(encoding_array)
            known_face_names.append(user.name)
        
        print(f"Loaded {len(known_face_encodings)} known face encodings from the database.")

    except Exception as e:
        print(f"Error loading known faces from database: {e}")

def capture_by_frames():
    global camera, user_new_encoding
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure the image is 8-bit and RGB
        if rgb_frame.dtype != np.uint8:
            print("Frame is not 8-bit.")
            continue
        if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
            print("Frame is not RGB.")
            continue

        # Detect face locations using face_recognition
        try:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            continue
        
        if len(face_encodings) > 0:
            user_new_encoding = face_encodings[0]

        # If no faces are found, return the frame as is
        if known_face_encodings is None or len(known_face_encodings) == 0 and not is_recognizing:
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global is_recognizing
    is_recognizing = True
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        load_known_faces_db()
    return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return render_template('stop.html')


@app.route('/form-add-user', methods=['GET'])
def form_add_user():
    global is_recognizing, camera
    is_recognizing = False
    if camera is None:
        camera = cv2.VideoCapture(0)
    return render_template('form_add_user.html')

@app.route('/add-user', methods=['POST'])
def add_user():
    global user_new_encoding
    name = request.form['name']
    print('name:', name)
    print('encoding:', user_new_encoding)
    # Process image data
    try:
        try:
            new_user = User(name=name, encoding=user_new_encoding)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('index'))  # Redirect after success
        except Exception as e:
            return f"Error: Unable to save user data to database: {str(e)}"

    except Exception as e:
        return f"Error processing image: {str(e)}"


@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=True, port=8000)
