import os
import cv2
import face_recognition
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

# Function to load and process training images from a specified folder
def load_training_data(folder_path):
    """
    Load images from a folder and extract face encodings for training.

    Args:
        folder_path (str): Path to the directory containing subfolders of images for each person.

    Returns:
        tuple: List of face encodings and corresponding names.
    """
    known_encodings = []  # List to store face encodings
    known_names = []      # List to store corresponding person names

    # Iterate through each person's folder in the training directory
    for person_name in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_path):
            continue  # Skip non-directory files

        # Process each image in the person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)  # Load image using face_recognition

            # Generate face encodings for the image
            encodings = face_recognition.face_encodings(image)

            # Check if any face encoding is found
            if encodings:
                known_encodings.append(encodings[0])  # Use the first face detected
                known_names.append(person_name)
            else:
                print(f"Warning: No faces detected in {image_path}")

    return known_encodings, known_names

# Function to train the SVM classifier
def train_model(encodings, names):
    """
    Train an SVM classifier using face encodings and corresponding names.

    Args:
        encodings (list): List of face encodings.
        names (list): List of corresponding person names.

    Returns:
        tuple: Trained SVM model and label encoder.

    Raises:
        ValueError: If no encodings or names are provided.
    """
    if not encodings or not names:
        raise ValueError("No encodings or names found. Ensure your training data is valid.")

    # Convert lists to numpy arrays for sklearn compatibility
    encodings = np.array(encodings)
    names = np.array(names)

    # Encode categorical names into numerical labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(names)

    # Initialize and train SVM classifier with linear kernel
    model = SVC(kernel='linear', probability=True)
    model.fit(encodings, labels)

    return model, label_encoder

# Function to save the trained model and label encoder
def save_model(model, label_encoder, model_path="face_model.pkl", label_path="label_encoder.pkl"):
    """
    Save the trained SVM model and label encoder to disk.

    Args:
        model (SVC): Trained SVM classifier.
        label_encoder (LabelEncoder): Fitted label encoder.
        model_path (str): Path to save the SVM model.
        label_path (str): Path to save the label encoder.
    """
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_path)

# Function to mark attendance in a daily log file
def mark_attendance(name):
    """
    Record attendance for a person in a daily log file.

    Args:
        name (str): Name of the person to mark attendance for.

    Returns:
        bool: True if attendance was marked, False if already marked.
    """
    date_today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"attendance_{date_today}.txt"

    # Create attendance file if it doesn't exist
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w") as file:
            file.write("Name,Date,Time\n")

    # Check if the person's attendance is already marked
    with open(attendance_file, "r+") as file:
        lines = file.readlines()
        names_marked = [line.split(",")[0] for line in lines]

        if name not in names_marked:
            time_now = datetime.now().strftime("%H:%M:%S")
            file.write(f"{name},{date_today},{time_now}\n")
            return True
        else:
            return False

# Function to recognize faces in a live video feed
def recognize_faces(model, label_encoder):
    """
    Recognize faces in a live video feed and mark attendance.

    Args:
        model (SVC): Trained SVM classifier.
        label_encoder (LabelEncoder): Fitted label encoder for decoding predictions.
    """
    video_capture = cv2.VideoCapture(0)  # Initialize webcam

    while True:
        ret, frame = video_capture.read()  # Capture frame
        if not ret:
            break  # Exit if frame capture fails

        # Convert frame to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Predict using the trained SVM model
            matches = model.predict_proba([face_encoding])[0]
            label_index = np.argmax(matches)
            name = label_encoder.inverse_transform([label_index])[0]
            confidence = matches[label_index]

            # Mark attendance if confidence exceeds threshold
            if confidence > 0.75:  # Confidence threshold
                attendance_marked = mark_attendance(name)

                if attendance_marked:
                    status_text = f"{name} marked attendance"
                else:
                    status_text = f"{name} already marked attendance"

                # Draw rectangle around face and display name with confidence and status
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the video feed
        cv2.imshow("Video", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

# Main execution block
if __name__ == "__main__":
    """
    Main script to load training data, train the model, save it, and start face recognition.
    """
    # Path to the folder containing training images
    training_folder = "output_faces"

    print("Loading training data...")
    encodings, names = load_training_data(training_folder)

    if not encodings or not names:
        print("No valid training data found. Exiting...")
    else:
        print("Training model...")
        model, label_encoder = train_model(encodings, names)

        print("Saving model...")
        save_model(model, label_encoder)

        print("Starting face recognition...")
        recognize_faces(model, label_encoder)
