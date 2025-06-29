# Import required libraries
import cv2  # OpenCV for webcam capture and image processing
import face_recognition  # Library for face detection and encoding
import os  # For handling file and directory operations
import numpy as np  # For numerical operations and array handling
from datetime import datetime  # For timestamp generation
import pickle  # For potential model persistence (not used in this version)

# Define the path to the folder containing student images for training
path = 'student_images'

# Initialize lists to store images and corresponding names
images = []  # List to store loaded images
classNames = []  # List to store names extracted from image filenames

# Load all images from the student_images folder
mylist = os.listdir(path)  # Get list of all files in the directory
for cl in mylist:
    # Read each image file
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)  # Append image to the images list
    # Extract name from filename (without extension) and append to classNames
    classNames.append(os.path.splitext(cl)[0])

# Function to generate face encodings for a list of images
def findEncodings(images):
    encodeList = []  # List to store face encodings
    for img in images:
        # Convert image from BGR (OpenCV format) to RGB (face_recognition format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Generate face encoding for the image (assumes one face per image)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)  # Append encoding to the list
    return encodeList

# Generate face encodings for the training images
encoded_face_train = findEncodings(images)

# Function to mark attendance and log it to a file
def markAttendance(name):
    # Get today's date in DD-Month-YYYY format
    today_date = datetime.now().strftime('%d-%B-%Y')
    found = False  # Flag to track if attendance is already marked

    # Check if attendance has already been marked for today
    with open('Attendance.txt', 'r+') as f:
        myDataList = f.readlines()  # Read all lines from the attendance file
        for line in myDataList:
            entry = line.strip().split(',')  # Split each line into components
            # Check if the name and date match (indicating attendance already marked)
            if len(entry) > 2 and entry[0] == name and entry[2].strip() == today_date:
                found = True
                break

    # If attendance not found, mark it
    if not found:
        with open('Attendance.txt', 'a') as f:
            now = datetime.now()  # Get current time
            time = now.strftime('%I:%M:%S %p')  # Format time as HH:MM:SS AM/PM
            # Append new attendance record in the format: name, time, date
            f.write(f'\n{name}, {time}, {today_date}')
            print(f"Attendance marked for {name}")  # Log to console
            return "Attendance marked"  # Return status
    else:
        print(f"Attendance already marked for {name} today")  # Log to console
        return "Attendance already marked"  # Return status

# Initialize webcam for real-time face recognition
cap = cv2.VideoCapture(0)  # Open default webcam (index 0)

# Main loop for capturing and processing video frames
while True:
    success, img = cap.read()  # Read frame from webcam
    # Resize frame to 1/4 size for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # Convert resized frame from BGR to RGB for face_recognition
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the resized frame
    faces_in_frame = face_recognition.face_locations(imgS)
    # Generate face encodings for detected faces
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    
    # Iterate through detected faces and their encodings
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        # Compare detected face encoding with training encodings
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        # Calculate distances between detected face and training encodings
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        # Find the index of the closest match
        matchIndex = np.argmin(faceDist)
        
        # If a match is found
        if matches[matchIndex]:
            # Get the name of the matched person (converted to lowercase)
            name = classNames[matchIndex].upper().lower()
            # Extract face location coordinates (scaled back up by 4)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw a green rectangle around the detected face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw a filled green rectangle for the name label background
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            # Display the name on the label
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Mark attendance and get status
            attendance_status = markAttendance(name)
            
            # Display attendance status below the face
            cv2.putText(img, attendance_status, (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the processed frame in a window named 'webcam'
    cv2.imshow('webcam', img)
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
