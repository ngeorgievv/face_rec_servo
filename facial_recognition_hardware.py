#This is a modified version of Face Recognition. This modified version contains code from the authors that originally made this program. Licensing can be found in the main Face Recognition folder.
#This version does not support picamera. It is designed to support USB cameras.
#The speed of face_rec will depend on the hardware used. It runs very fast on the RPI5. 

#THIS IS A SCRIPT USED FOR CONTROLLING HARDWARE, CURRENTLY CONFIGURED TO CONTROL THE GPIO PINS USING AN IF STATEMENT ON LINE 74.

import face_recognition
import cv2
import numpy as np
import time
import pickle
from gpiozero import LED
from gpiozero import AngularServo
from time import sleep

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the USB camera (0 for the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the USB camera.")
    exit()

# Initialize GPIO
#output = LED(14)
servo = AngularServo(14, min_angle=0, max_angle=180)

# Initialize our variables
cv_scaler = 4  # This has to be a whole number
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# List of names that will trigger the GPIO pin
authorized_names = ["nikola"]  # Replace with names you wish to authorise (CASE-SENSITIVE)

def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Resize the frame to increase performance (fewer pixels processed)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB (face_recognition uses RGB)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    
    for face_encoding in face_encodings:
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Check if the detected face is in our authorized list
            if name in authorized_names:
                authorized_face_detected = True
        face_names.append(name)
    
    # Control the GPIO pin based on face detection
    if authorized_face_detected:
        servo.angle = 0
    else:
        servo.angle = 180
    
    return frame

def draw_results(frame):
    # Draw bounding boxes and labels on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame was resized
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        # Draw a label with the name below the face
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
        # Add an indicator if the person is authorized
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    # Capture a frame from the USB camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Process the frame for face detection and recognition
    processed_frame = process_frame(frame)
    
    # Draw the results on the frame
    display_frame = draw_results(processed_frame)
    
    # Calculate and display FPS
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the video feed with results
    cv2.imshow('Video', display_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
output.off()  # Ensure the GPIO pin is turned off on exit
