import cv2
import face_recognition
from datetime import datetime
import os

# Function to get encodigns of images
def find_encodings(images):
    return [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]


# Function for marking attendance
def mark_attendance(name, filename='Attendance.csv'):
    with open(filename, 'a+') as f:
        name_list = [line.split(',')[0] for line in f]
        if name not in name_list:
            now = datetime.now().strftime('%I:%M:%S:%p, %d-%B-%Y')
            f.write(f'n{name}, {now}\n')


# Variable initializations
path = 'student_images'
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]
class_names = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
encoded_faces = find_encodings(images)
cap = cv2.VideoCapture(0)

# While Loop
while True:
    success, img = cap.read()
    img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(img_s)
    encoded_faces_frame = face_recognition.face_encodings(img_s, faces_in_frame)
    
    for encode_face, face_loc in zip(encoded_faces_frame, faces_in_frame):
        match_index = min(range(len(encoded_faces)), key=lambda i: face_recognition.face_distance([encoded_faces[i]], encode_face))
        if face_recognition.compare_faces([encoded_faces[match_index]], encode_face)[0]:
            name = class_names[match_index].upper().lower()
            x1, y1, x2, y2 = [coord * 4 for coord in face_loc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
