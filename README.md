# Face-Recognition-Based-Attendance-System
## Overview
This project implements a Face Recognition Attendance System using Python. It uses face recognition to identify individuals in a live video feed and automatically marks their attendance in a daily log file. The system leverages the `face_recognition` library for face detection and encoding, and an SVM classifier for recognizing faces based on pre-trained data.

## Features
- Loads and processes training images from a specified folder structure.
- Trains an SVM classifier to recognize faces based on their encodings.
- Saves the trained model and label encoder for reuse.
- Recognizes faces in a live webcam feed with confidence scoring.
- Marks attendance in a daily text file, preventing duplicate entries.
- Displays real-time video with face bounding boxes, names, confidence scores, and attendance status.

## Prerequisites
To run this project, you need the following:
- Python 3.6 or higher
- Required Python libraries:
  - `face_recognition`: For face detection and encoding
  - `opencv-python`: For webcam access and image processing
  - `numpy`: For numerical operations
  - `scikit-learn`: For SVM classifier and label encoding
  - `joblib`: For saving and loading models

Install the required libraries using pip:
```bash
pip install face_recognition opencv-python numpy scikit-learn joblib
```

**Note**: Installing `face_recognition` may require additional dependencies like `dlib`. Follow the library's documentation for platform-specific installation instructions.

## Project Structure
- `model_with_training.py`: The main script for the face recognition and attendance system.
- `output_faces/`: Directory containing subfolders of training images, where each subfolder is named after a person and contains their face images.
- `README.md`: This file, providing an overview and instructions for the project.
- `face_model.pkl`: Generated file storing the trained SVM model (created after running the script).
- `label_encoder.pkl`: Generated file storing the label encoder (created after running the script).
- `attendance_<date>.txt`: Generated daily attendance log files (e.g., `attendance_2025-06-29.txt`).

## Usage
1. **Prepare Training Data**:
   - Create a directory named `output_faces` in the same directory as the script.
   - Inside `output_faces`, create subfolders named after each person (e.g., `output_faces/John_Doe/`, `output_faces/Jane_Smith/`).
   - Place multiple images of each person in their respective subfolder (e.g., `john_doe_1.jpg`, `john_doe_2.jpg`).
   - Ensure images are in a format supported by `face_recognition` (e.g., JPG, PNG).

2. **Run the Script**:
   ```bash
   model_with_training.py
   ```
   - The script performs the following steps:
     1. Loads and processes training images from the `output_faces` directory.
     2. Trains an SVM classifier using face encodings and saves the model and label encoder.
     3. Starts a live webcam feed to recognize faces and mark attendance.

3. **Expected Output**:
   - Console logs:
     - `Loading training data...`
     - `Training model...`
     - `Saving model...`
     - `Starting face recognition...`
     - Warnings if no faces are detected in training images (e.g., `Warning: No faces detected in <image_path>`).
   - A video window displaying the webcam feed with:
     - Green rectangles around detected faces.
     - Text showing the recognized name and confidence score (e.g., `Tarun (0.85)`).
     - Attendance status (e.g., `Tarun marked attendance` or `Tarun already marked attendance`).
   - A daily attendance log file (e.g., `attendance_2025-06-29.txt`) with entries like:
     ```
     Name,Date,Time
     Tarun,2025-06-29,08:15:23
     Vasu,2025-06-29,08:16:45
     ```

4. **Stopping the Program**:
   - Press the `q` key in the video window to stop the program and close the webcam.

## Notes
- **Training Data**:
  - Ensure each person's subfolder contains multiple clear images with visible faces.
  - The system uses the first detected face in each image for training.
  - If no faces are detected in an image, a warning is printed, and the image is skipped.
- **Face Recognition**:
  - The system uses a confidence threshold of 0.75 for marking attendance. Adjust this in the `recognize_faces` function if needed.
  - The SVM classifier uses a linear kernel for simplicity; you can experiment with other kernels for better performance.
- **Attendance Logging**:
  - Attendance is recorded only once per person per day to avoid duplicates.
  - Log files are created daily with the format `attendance_YYYY-MM-DD.txt`.
- **System Requirements**:
  - A functional webcam is required (default camera index is 0).
  - Ensure sufficient lighting and clear face visibility for accurate recognition.
- **Performance**:
  - For large datasets, training may take time. Consider pre-saving the model and loading it for faster startup.
  - The system assumes a reliable webcam feed; add error handling for production use.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built using Python’s `face_recognition`, `opencv-python`, `scikit-learn`, and `joblib` libraries.
- Inspired by automated attendance systems using facial recognition technology.
