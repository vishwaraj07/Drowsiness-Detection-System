#Author - Vishwaraj Chavan



# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize the Pygame mixer for sound alerts and load a music file
mixer.init()
mixer.music.load("D:\DSE\CV\Project\music.wav") # Add the path of the music file

# Define a function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set threshold and frame check variables
thresh = 0.25
frame_check = 20

# Initialize face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("D:\DSE\CV\Project\shape_predictor_68_face_landmarks.dat") # Add the path of the shape predictor file

# Define the indices for the left and right eye in the facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open a video capture object for the default camera (0)
cap = cv2.VideoCapture(0)

# Initialize a flag to track consecutive frames with closed eyes
flag = 0

# Main loop for processing video frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Resize the frame for better processing speed
    frame = imutils.resize(frame, width=450)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Loop over detected faces
    for subject in subjects:
        # Predict facial landmarks for the detected face
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates from the facial landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio for left and right eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Calculate the average eye aspect ratio
        ear = (leftEAR + rightEAR) / 2.0

        # Create convex hulls for left and right eyes and draw them on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the average eye aspect ratio is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)

            # If consecutive frames with closed eyes exceed the frame_check, trigger an alert
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Play the alert sound
                mixer.music.play()
        else:
            # Reset the flag if eyes are open
            flag = 0

    # Display the frame with overlays
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()
cap.release()
