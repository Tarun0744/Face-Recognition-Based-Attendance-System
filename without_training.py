import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

path = 'student_images'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

def markAttendance(name):
    today_date = datetime.now().strftime('%d-%B-%Y')
    found = False

    # Check if attendance has already been marked for today
    with open('Attendance.txt', 'r+') as f:
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.strip().split(',')
            if len(entry) > 2 and entry[0] == name and entry[2].strip() == today_date:
                found = True
                break

    if not found:
        with open('Attendance.txt', 'a') as f:
            now = datetime.now()
            time = now.strftime('%I:%M:%S %p')
            f.write(f'\n{name}, {time}, {today_date}')
            print(f"Attendance marked for {name}")
        return "Attendance marked"
    else:
        print(f"Attendance already marked for {name} today")
        return "Attendance already marked"

# Capture images from webcam and process
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceloc
            # Scale the coordinates back up by 4
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Check and mark attendance with date verification
            attendance_status = markAttendance(name)
            
            # Display attendance status on the webcam feed
            cv2.putText(img, attendance_status, (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
