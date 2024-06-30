from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from io import BytesIO
from scipy.spatial.distance import cosine
import cv2
import numpy as np
import tensorflow as tf
import psycopg2
import os
import uuid

app = FastAPI()

# PostgreSQL connection details
db_host = "localhost"
db_name = "AttendenceSystem"
db_user = "postgres"
db_password = "test123"

# Function to connect to the PostgreSQL database
def get_db_connection():
    conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_password)
    return conn

# Load the FaceNet model
def load_facenet_model(model_path):
    print("Loading FaceNet model...")
    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    sess = tf.compat.v1.Session()
    tf.compat.v1.import_graph_def(graph_def, name='')
    input_tensor = sess.graph.get_tensor_by_name('input:0')
    output_tensor = sess.graph.get_tensor_by_name('embeddings:0')
    print("FaceNet model loaded successfully.")
    return sess, input_tensor, output_tensor

# Preprocess face image
def preprocess_face(face_img):
    preprocessed_img = cv2.resize(face_img, (160, 160))
    preprocessed_img = preprocessed_img.astype('float32') / 255.0
    return preprocessed_img

# Calculate face embedding
def calculate_embedding(sess, input_tensor, output_tensor, face_img):
    preprocessed_img = preprocess_face(face_img)
    embedding = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(preprocessed_img, axis=0), 'phase_train:0': False})
    return embedding

# Initialize FaceNet model
model_path = '20180402-114759.pb'
sess, input_tensor, output_tensor = load_facenet_model(model_path)

@app.post("/upload/")
async def upload_image(person_id: str = Form(...), file: UploadFile = File(...)):
    print("Received image upload request.")

    # Read image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Convert image to OpenCV format
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces using Haar cascade
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    # Check if more than one face is detected
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected in the image")

    # Check if no faces are detected
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No valid face found to process")
    else:
        print(f"Detected faces: {faces}")

    # Process the detected face
    face_processed = False
    inserted_id = None

    for (x, y, w, h) in faces:
        if face_processed:
            break

        # Exclude small and light areas surrounding the face
        face_area = gray_image[y:y+h, x:x+w]
        avg_brightness = np.mean(face_area)
        if avg_brightness > 150:
            continue
        
        # Filter out detected regions with aspect ratio outside of reasonable face proportions
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2:
            continue
        
        # Exclude regions that are too small
        if w < 100 or h < 120:
            continue

        # Extract the face image
        face_img = cv2_image[y:y+h, x:x+w]

        # Calculate face embedding
        embedding = calculate_embedding(sess, input_tensor, output_tensor, face_img)
        print("Embedding calculated successfully.")

        # Draw bounding box around the detected face
        color = (0, 255, 0)  # Green for detected faces
        label = person_id  # Use person_id as the label

        if w >= 100 and h >= 120:  # Only draw bounding box if the size is not too small
            cv2.rectangle(cv2_image, (x, y), (x+w, y+h), color, 5)
            cv2.putText(cv2_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

        face_processed = True

        try:
            output_path = os.path.join("Registered_Persons", f"{person_id}.jpg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2_image)
            print(f"Image saved with bounding box at {output_path}")
        except Exception as e:
            print("Error while saving the image:", e)
            raise HTTPException(status_code=500, detail="Error while saving the image")

        # Connect to the PostgreSQL database
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Convert the image with bounding box to bytes for database insertion
            _, buffer = cv2.imencode('.jpg', cv2_image)
            img_bytes = buffer.tobytes()

            cursor.execute("INSERT INTO users (person_id, frame, embedding) VALUES (%s, %s, %s) RETURNING face_id",
                           (person_id, psycopg2.Binary(img_bytes), psycopg2.Binary(embedding.tobytes())))
            inserted_id = cursor.fetchone()[0]
            conn.commit()
        except Exception as e:
            print("Error occurred during database operation:", e)
            conn.rollback()
            raise HTTPException(status_code=500, detail="Database operation error")
        finally:
            cursor.close()
            conn.close()

    if not face_processed:
        raise HTTPException(status_code=400, detail="No valid face found to process")

    return {"message": "Image uploaded successfully", "person_id": person_id}


@app.post("/search/")
async def search_image_similarity(file: UploadFile = File(...)):
    print("Received image search request.")

    # Read image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Convert image to OpenCV format
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces using Haar cascade
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    # Check if faces are detected
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Connect to the PostgreSQL database
    conn = get_db_connection()
    cursor = conn.cursor()

    faces_objs = []

    try:
        # Retrieve embeddings from the database
        cursor.execute("SELECT face_id, person_id, embedding FROM users")
        rows = cursor.fetchall()

        # Loop through each detected face in the image
        for (x, y, w, h) in faces:
            face_obj = {"face_id": None, "is_recognized": False, "cutted_face": None}

            # Exclude small and light areas surrounding the face
            face_area = gray_image[y:y+h, x:x+w]
            avg_brightness = np.mean(face_area)
            if avg_brightness > 150 or avg_brightness < 50:
                continue
            
            # Exclude regions with unreasonable aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2:
                continue
            
            # Exclude regions that are too small
            if w < 100 or h < 120:
                continue

            # Extract the face image
            face_img = cv2_image[y:y+h, x:x+w]

            # Calculate face embedding
            embedding = calculate_embedding(sess, input_tensor, output_tensor, face_img)

            # Initialize variables to store matching ID and person_id
            matching_id = None
            matching_person_id = None
            max_similarity = 0

            # Calculate cosine similarity with each embedding in the database
            for row in rows:
                db_id = row[0]
                db_person_id = row[1]
                db_embedding = np.frombuffer(row[2], dtype=np.float32)
                similarity = 1 - cosine(embedding.flatten(), db_embedding.flatten())

                if similarity > max_similarity:
                    max_similarity = similarity
                    matching_id = db_id
                    matching_person_id = db_person_id

            if max_similarity >= 0.6:
                face_obj["face_id"] = matching_id
                face_obj["is_recognized"] = True
                face_path = os.path.join("Detected_Persons", matching_person_id, f"{uuid.uuid4()}.jpg")
                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                cv2.imwrite(face_path, face_img)
                face_obj["cutted_face"] = face_path
            else:
                face_path = os.path.join("Detected_Persons", "unmatched", f"{uuid.uuid4()}.jpg")
                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                cv2.imwrite(face_path, face_img)
                face_obj["cutted_face"] = face_path

            if face_obj["cutted_face"] is not None:
                faces_objs.append(face_obj)

        return {"status": "ok", "message": "Image processed successfully", "faces_objs": faces_objs}

    except Exception as e:
        print("Error occurred during database operation:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
