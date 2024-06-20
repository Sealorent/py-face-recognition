from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

camera = None  # Global variable to store camera instance
known_face_encodings = []  # List to store known face encodings
image_path = "./static/image.jpg"  # Path to the image file containing known faces
known_face_names = ["Known Person"]  # List to store known face names

def load_known_faces():
    global known_face_encodings, known_face_names
    try:
        # Load the known image using OpenCV
        known_image = cv2.imread(image_path)

        # Convert to RGB format if necessary
        if known_image is None:
            raise ValueError("Image not found or unable to load.")
        if known_image.ndim == 2:
            known_image = cv2.cvtColor(known_image, cv2.COLOR_GRAY2RGB)
        elif known_image.ndim == 3 and known_image.shape[2] == 4:
            known_image = cv2.cvtColor(known_image, cv2.COLOR_RGBA2RGB)
        elif known_image.ndim == 3 and known_image.shape[2] == 3:
            known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

        # Get face encodings
        known_face_encodings = face_recognition.face_encodings(known_image)
        
        if len(known_face_encodings) == 0:
            print("No face found in the image.")
        else:
            known_face_encodings = [known_face_encodings[0]]  # Take the first face encoding
    except Exception as e:
        print(f"Error loading known faces: {e}")

def capture_by_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using CascadeClassifier
        detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert the region of interest (ROI) to RGB for face recognition
            face_rgb = rgb_frame[y:y+h, x:x+w]

            # if face_rgb is None:
            #     raise ValueError("Image not found or unable to load.")
            # if face_rgb.ndim == 2:
            #     face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_GRAY2RGB)
            # elif face_rgb.ndim == 3 and face_rgb.shape[2] == 4:
            #     face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_RGBA2RGB)
            # elif face_rgb.ndim == 3 and face_rgb.shape[2] == 3:
            face_convert_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
            
            print(f"Face RGB shape: {face_rgb.shape}, dtype: {face_rgb.dtype}")

            face_rgb_encodings = face_recognition.face_encodings(face_convert_rgb)

            if(len(known_face_encodings) > 0):
                matches = face_recognition.compare_faces(known_face_encodings, face_rgb_encodings[0])
                name = "Unknown"
                if matches[0]:
                    name = known_face_names[0]
                cv2.putText(frame, name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else :
                cv2.putText(frame, "No known faces", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            

            # try:


            #     # if len(face_encodings) > 0:
            #     #     face_encoding = face_encodings[0]
            #     #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            #     #     name = "Unknown"

            #     #     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #     #     best_match_index = np.argmin(face_distances)
            #     #     if matches[best_match_index]:
            #     #         name = known_face_names[best_match_index]

            #     #     cv2.putText(frame, name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # except Exception as e:
            #     print(f"Error recognizing face: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        load_known_faces()
    return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=8000)
