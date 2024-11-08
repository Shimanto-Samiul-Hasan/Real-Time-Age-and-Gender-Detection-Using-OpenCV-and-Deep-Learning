import cv2
import math
from tkinter import Tk, filedialog, messagebox
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Model files and configurations
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def process_frame(frame):
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    # Show warning if no face is detected
    if not faceBoxes:
        messagebox.showwarning("No Face Detected", "No face detected or the image is not clear.")
        return None  # Return None to indicate no face was processed
    
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return resultImg

# Function to open webcam with live feed
def open_webcam():
    messagebox.showinfo("Webcam Mode", "Press 'q' to close the webcam feed.")

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        resultImg = process_frame(frame)

        if resultImg is not None:
            # Resize the result image to 800x600 for consistent display size
            resultImg = cv2.resize(resultImg, (800, 600))
            cv2.imshow("Detecting age and gender", resultImg)
        
        # Check if the window was closed
        if cv2.getWindowProperty("Detecting age and gender", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to upload an image
def upload_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filepath:
        frame = cv2.imread(filepath)
        resultImg = process_frame(frame)

        if resultImg is not None:  # Only display if a face was detected
            # Resize the result image to 800x600
            resultImg = cv2.resize(resultImg, (800, 600))
            cv2.imshow("Detecting age and gender", resultImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Function to display options for webcam or image upload
def display_options():
    root = Tk()
    root.withdraw()  # Hide the main window
    choice = messagebox.askyesno("Choose Input", "Do you want to use the webcam? (Yes for webcam, No to upload an image)")
    if choice:
        open_webcam()
    else:
        upload_image()

# Start the program
display_options()
