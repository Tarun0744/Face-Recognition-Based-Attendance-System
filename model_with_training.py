import os
import cv2
import face_recognition
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

# Step 1: Load and process training images
def load_training_data(folder_path):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)

            # Generate face encodings
            encodings = face_recognition.face_encodings(image)

            # Check if any face encoding is found
            if encodings:
                known_encodings.append(encodings[0])  # Use the first face detected
                known_names.append(person_name)
            else:
                print(f"Warning: No faces detected in {image_path}")

    return known_encodings, known_names

# Step 2: Train the model
def train_model(encodings, names):
    if not encodings or not names:
        raise ValueError("No encodings or names found. Ensure your training data is valid.")

    # Convert lists to numpy arrays for sklearn
    encodings = np.array(encodings)
    names = np.array(names)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(names)

    # Train SVM classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(encodings, labels)

    return model, label_encoder

# Step 3: Save the trained model and label encoder
def save_model(model, label_encoder, model_path="face_model.pkl", label_path="label_encoder.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_path)

# Step 4: Mark attendance
def mark_attendance(name):
    date_today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"attendance_{date_today}.txt"

    if not os.path.exists(attendance_file):
        with open(attendance_file, "w") as file:
            file.write("Name,Date,Time\n")

    with open(attendance_file, "r+") as file:
        lines = file.readlines()
        names_marked = [line.split(",")[0] for line in lines]

        if name not in names_marked:
            time_now = datetime.now().strftime("%H:%M:%S")
            file.write(f"{name},{date_today},{time_now}\n")
            return True
        else:
            return False

# Step 5: Recognize faces in video feed
def recognize_faces(model, label_encoder):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Predict using the trained model
            matches = model.predict_proba([face_encoding])[0]
            label_index = np.argmax(matches)
            name = label_encoder.inverse_transform([label_index])[0]
            confidence = matches[label_index]

            # Mark attendance
            if confidence > 0.75:  # Confidence threshold
                attendance_marked = mark_attendance(name)

                if attendance_marked:
                    status_text = f"{name} marked attendance"
                else:
                    status_text = f"{name} already marked attendance"

                # Draw rectangle and display name with attendance status
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Show the video feed
        cv2.imshow("Video", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    # Path to training folder
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

        
        
