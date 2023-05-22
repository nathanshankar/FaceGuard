# FaceGuard
Facial Data Capture and Training along with validation
This software utilizes computer vision techniques to capture facial images, train a facial recognition model, and subsequently recognize users based on their faces. The software is implemented in Python using the OpenCV library.

The main functionalities of the software are as follows:

Capture and Train: The capture_and_train function allows users to capture facial images and train a facial recognition model. It uses the LBPHFaceRecognizer algorithm from OpenCV to create and train the recognizer. The captured images are stored in a user-specific directory within the training data directory.

Recognize: The recognize function performs real-time facial recognition. It uses the trained recognizer models to identify users based on their faces. The software captures video from the default camera, detects faces using the Haar cascade classifier, and compares the detected faces with the trained models to determine the user's identity. If a user is recognized, their ID and confidence level are displayed on the video feed.

User Interface: The main_loop function presents a menu-driven user interface where the user can choose between training and recognition modes. In training mode, the user can provide their name and capture facial images to create a personalized recognizer model. In recognition mode, the software continuously recognizes faces in real-time video feed and displays the identified user's name if recognized.

The code utilizes external libraries such as NumPy, Pillow (PIL), and custom modules (my_assistant.shared, talk, and listen) to provide additional functionalities related to speech synthesis and speech recognition.

Note: The code assumes the availability of Haar cascade XML file for face detection and requires modification of the file paths (path and cascade classifier path) to point to the appropriate directories in the user's system.

Overall, this software provides the foundation for building a facial recognition system capable of training and detecting individuals based on their facial features.
