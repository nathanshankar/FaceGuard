import cv2
import os
import numpy as np
from PIL import Image
import my_assistant.shared
from talk import speak
from listen import get_audio
import shutil


def capture_and_train(user_name):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(r"YOUR_PATH\haarcascade_frontalface_default.xml")
    path = r"YOUR_PATH\training_data" # modify this line to point to your training data directory
    
    # Check if a similar face already exists in training data
    existing_users = recognize(1)
    if existing_users!=False:
                speak(f"You look like {existing_users}!")
                return
        
    # If no similar face exists, capture and train new user
    user_id = 1
    while True:
        trained_file = f"face_data_{user_id}.yml"
        if os.path.exists(os.path.join(path, trained_file)):
            user_id += 1
        else:
            break

    # Create directory for user if it doesn't exist
    user_path = os.path.join(path, str("User"+"."+str(user_id)))
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    
    # Capture 500 images of user
    camera = cv2.VideoCapture(0)
    count = 0
    while count < 500:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            count += 1
            # Save image to user's directory
            img_path = os.path.join(user_path, f"User.{user_id}.{count}.jpg")
            cv2.imwrite(img_path, gray[y:y+h,x:x+w])
        cv2.imshow("Capturing Images", frame)
        cv2.waitKey(100)
    camera.release()
    cv2.destroyAllWindows()
    
    # Train recognizer with user's images
    faces, ids = [], []
    for img_path in os.listdir(user_path):
        img_array = np.array(Image.open(os.path.join(user_path, img_path)).convert('L'))
        faces.append(img_array)
        ids.append(user_id)
    ids = np.array(ids, dtype=np.int32)
    recognizer.train(faces, np.array(ids))
    
    # Save recognizer to file
    trained_file = f"face_data_{user_id}.yml"
    recognizer.write(os.path.join(path, trained_file))
    
    # Rename file to include user name
    new_file_name = f"face_data_{user_name}.yml"
    os.rename(os.path.join(path, trained_file), os.path.join(path, new_file_name))
    user_folder = r"YOUR_PATH\training_data\User." + str(user_id)
    shutil.rmtree(user_folder)


def recognize(e):
    threshold=50
    detector = cv2.CascadeClassifier(r"YOUR_PATH\haarcascade_frontalface_default.xml")
    path = r"YOUR_PATH\training_data" # modify this line to point to your training data directory
    
    # Load recognizer for each user
    recognizers = {}
    for trained_file in os.listdir(path):
        if trained_file.startswith("face_data_") and trained_file.endswith(".yml"):
            user_id = trained_file.split("_")[2].split(".")[0]
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(os.path.join(path, trained_file))
            recognizers[user_id] = recognizer
    
    # Capture video from default camera
    camera = cv2.VideoCapture(0)
    
    # Flag to keep track of whether a user has already been detected
    user_detected = False
    old_user = ""
    
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        # Detect faces in the frame
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in faces:
            # Draw a rectangle around each face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # Recognize user based on face
            face_gray = gray[y:y+h, x:x+w]
            user_id = None
            confidence = np.inf
            for id_, recognizer in recognizers.items():
                _, confidence_ = recognizer.predict(face_gray)
                if confidence_ < confidence:
                    user_id = id_
                    confidence = confidence_

            # Draw user id and confidence on face rectangle
            if confidence < threshold:
                cv2.putText(frame, f"User {user_id} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if e==1:
                    return user_id
                else:
                    if (not user_detected or old_user!=user_id) and len(faces)==1:
                        speak(f"Hello {user_id}")
                    user_detected = True
                    old_user=user_id
            else:
                cv2.putText(frame, f"Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                user_detected = False
                if e==1:
                    return False
        # Show the frame with face rectangles
        cv2.imshow("Recognizing Users", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release camera and close window
    camera.release()
    cv2.destroyAllWindows()

def main_loop():
    while True:
        print("===== MENU =====")
        print("1. Train")
        print("2. Recognize")
        print("================")
        
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == "1":
            user_name = input("Enter your name: ")
            capture_and_train(user_name)
        elif choice == "2":
            recognize(0)
        else:
            print("Invalid choice. Please try again.")

main_loop()

