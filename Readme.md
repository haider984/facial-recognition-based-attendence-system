### Documentation

This code implements a web service using FastAPI, a modern, fast (high-performance) web framework for building APIs with Python. The service handles image uploads containing human faces, extracts embeddings (numerical representations) of those faces using a pre-trained FaceNet model, and stores these embeddings along with associated metadata in a PostgreSQL database. Additionally, it provides functionality to search for similar faces in uploaded images by comparing their embeddings with those stored in the database.

### Functionality and Components Breakdown:

1. **Initialization**:
   - **FastAPI app**: The application instance is created using FastAPI.
   - **PostgreSQL connection details**: Connection parameters for PostgreSQL are defined, including the host, database name, user, and password.
   - **FaceNet model loading**: The FaceNet model is loaded using TensorFlow. This model generates embeddings for face images, facilitating face recognition tasks.

2. **Function Definitions**:
   - `get_db_connection()`: Establishes and returns a connection to the PostgreSQL database.
   - `load_facenet_model(model_path)`: Loads the FaceNet model from the specified file path and returns the session and necessary tensors for input and output.
   - `preprocess_face(face_img)`: Preprocesses a face image by resizing and normalizing it before generating its embedding.
   - `calculate_embedding(sess, input_tensor, output_tensor, face_img)`: Calculates the embedding for a face image using the loaded FaceNet model.

3. **API Endpoints**:
   - **`/upload/`**: Handles POST requests for uploading images.
     - **Parameters**:
       - `person_id` (Form): A string parameter used for identifying the person associated with the uploaded image.
       - `file` (UploadFile): The image file containing one or more faces.
     - **Process**:
       - Reads the uploaded image and converts it to an OpenCV format.
       - Detects faces using the Haar cascade classifier.
       - Checks if more than one face is detected and raises an error if so.
       - Preprocesses and calculates embeddings for detected faces.
       - Draws bounding boxes around detected faces and saves the annotated image.
       - Inserts the person_id, annotated image, and face embedding into the PostgreSQL database.
     - **Response**: Returns a success message along with the person_id and the ID of the inserted record.
   
   - **`/search/`**: Handles POST requests for searching for similar faces.
     - **Parameters**:
       - `file` (UploadFile): The image file containing one or more faces.
     - **Process**:
       - Reads the uploaded image and converts it to an OpenCV format.
       - Detects faces using the Haar cascade classifier.
       - Preprocesses and calculates embeddings for detected faces.
       - Retrieves embeddings from the database and compares them with the embeddings of detected faces.
       - If a match is found based on a similarity threshold, it returns their IDs.
       - If no matches are found, it returns a message indicating that no matches were found.
     - **Response**: Returns the matched faces' IDs and paths to the recognized face images.

### Database Structure

The code interacts with a PostgreSQL database named `AttendenceSystem`. Within this database, there is a table named `users`, which stores information about uploaded images, including the associated person (identified by `person_id`), the image, and the face embedding.

**Table Definition**:
```sql
CREATE TABLE users (
  face_id SERIAL PRIMARY KEY,
  person_id VARCHAR(255) NOT NULL,
  frame BYTEA NOT NULL,
  embedding BYTEA
);
```

**Column Details**:
- `face_id`: Auto-incremented primary key that uniquely identifies each record.
- `person_id`: A string (VARCHAR) that identifies the person associated with the uploaded image. It cannot be null.
- `frame`: Binary data (BYTEA) storing the uploaded image file in JPEG format.
- `embedding`: Binary data (BYTEA) storing the serialized NumPy array representing the face embedding.

### Additional Notes:

1. **Face Detection and Filtering**:
   - Faces are detected using Haar cascades.
   - Additional checks are implemented to filter out regions with inappropriate brightness, aspect ratios, and sizes to ensure only valid face regions are processed.

2. **Embedding Calculation and Storage**:
   - Embeddings are calculated using a pre-trained FaceNet model.
   - The calculated embeddings, along with the processed images, are stored in the PostgreSQL database.

3. **Image Saving and Directory Structure**:
   - Images are saved in specific directories based on the person ID.
   - Recognized faces are saved in a designated "Detected_Persons" directory, with each person's faces stored in subdirectories.

This comprehensive structure and documentation ensure that the FastAPI service is robust, scalable, and capable of handling face recognition tasks effectively by leveraging deep learning models and a relational database for storage and retrieval.