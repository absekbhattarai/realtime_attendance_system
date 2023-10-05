import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import urllib.request as ur
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

#opencv-python-rolling==4.7.0.20230211

# Using the webcam for capturing frames
cap = cv2.VideoCapture(0)

# loading the model for the object detection
net = cv2.dnn.readNetFromDarknet('/Users/yashmundra/Desktop/Vikram/advance_data_mining/adm_project/face_detection_project/model_yolo3/yolov3.cfg',
                                 '/Users/yashmundra/Desktop/Vikram/advance_data_mining/adm_project/face_detection_project/model_yolo3/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)



# Initializing the model for the arm detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.75,
                      min_tracking_confidence=0.75, max_num_hands=2)


# encoding the images for the face detection
elon_img = face_recognition.load_image_file(
    "/Users/yashmundra/Desktop/Vikram/advance_data_mining/adm_project/face_detection_project/peopleImg/elon.jpeg")
elon_encoding = face_recognition.face_encodings(elon_img)[0]

edsheeran_img = face_recognition.load_image_file(
    "/Users/yashmundra/Desktop/Vikram/advance_data_mining/adm_project/face_detection_project/peopleImg/edsheeran.jpeg")
edsheeran_encoding = face_recognition.face_encodings(edsheeran_img)[0]

mark_img = face_recognition.load_image_file(
    "/Users/yashmundra/Desktop/Vikram/advance_data_mining/adm_project/face_detection_project/peopleImg/mark.jpeg")
mark_encoding = face_recognition.face_encodings(mark_img)[0]

absek_img = face_recognition.load_image_file(
    "/Users/yashmundra/Desktop/Vikram/advance_data_mining/adm_project/face_detection_project/peopleImg/absek.jpg")
absek_encoding = face_recognition.face_encodings(absek_img)[0]

known_face_encoding = [
    elon_encoding,
    edsheeran_encoding,
    mark_encoding,
    absek_encoding
]

known_faces_names = [
    "Elon",
    "Ed Sheeran",
    "Mark",
    "Vikram"
]

students = known_faces_names.copy()
face_locations = []
face_encodings = []
face_names = []
detect_s = True

# Getting the dates for managing the time in the CSV files
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)


ln = net.getLayerNames()
layerOutputs = [ [] for _ in net.getUnconnectedOutLayers()]

# Modify this line to get the length of the output layer indices list
# layerOutputs = [ [] for _ in range(len(ln))] 
layerOutputs = [ [] for _ in net.getUnconnectedOutLayers()]

# Rest of the code remains the same
# for output in layerOutputs:
#     print(len(output))


# ln = net.getLayerNames()

# #Extra line
# layerOutputs = [ [] for _ in net.getUnconnectedOutLayers()]
# for i in net.getUnconnectedOutLayers():

#     layerOutputs[i - 1] = ln[i - 1]

#     #layerOutputs = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#     # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Classes for the object detection
classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
           "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
           "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
           "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]


def arm_detection(arm_detect):
    img = cv2.flip(arm_detect, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:

        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            cv2.putText(img, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)
            return True

        # If any hand present
        else:
            for i in results.multi_handedness:

                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    # Display 'Left Hand' on
                    # left side of window
                    cv2.putText(img, label + ' Hand',
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
                    return True

                if label == 'Right':
                    # Display 'Left Hand'
                    # on left side of window
                    cv2.putText(img, label + ' Hand', (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
                    return True

def face_detection(img):
    if detect_s:
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encoding, face_encoding)
            name = "Unknown"
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            print("This is the person", face_names)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(img, name + ' Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    # print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    f.close()


# Main flow of the program
while cap.isOpened():
    _, img = cap.read()  # Getting the frames from the webcam
    arm_detect = img
    if _ == False:  # If frame is false, break
        break

    # Image processing for object detection
    img = cv2.resize(img, (1280, 640), fx=1, fy=1)
    hight, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    layer_names = net.getLayerNames()
    # output_layers_name = []
    output_layers_name = net.getUnconnectedOutLayersNames()
    # print("This is the output layer", output_layers_name)
    layerOutputs = net.forward(output_layers_name)
    

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence,
                        (x, y + 400), font, 2, color, 2)
            # If the subject is person then check for the hand raise, when there is hand raise then only check their faces
            if label == 'person':
                arm_status = arm_detection(arm_detect)
                if arm_status:
                    face_detection(img)

    cv2.imshow('Person Detect', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
