import cv2
import numpy as np
# from google.colab.patches import cv2_imshow
face_proto = "./model/opencv_face_detector.pbtxt"
face_model = "./model/opencv_face_detector_uint8.pb"
age_proto = "./model/age_deploy.prototxt"
age_model = "./model/age_net.caffemodel"

face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

def detect_faces(frame, conf_threshold=0.7):
    net = face_net
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame, face_boxes

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def predict_age(face):
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # BGR mean values for age model
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    return age_list[age_preds[0].argmax()]
def process_image(image_path):
    frame = cv2.imread(image_path)
 
    if frame is None:
        print(f"Error: Image not found at {image_path}")
        return

    frame, face_boxes = detect_faces(frame)
    
    for (x1, y1, x2, y2) in face_boxes:
        face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1), 
                     max(0, x1-20):min(x2+20, frame.shape[1]-1)]
        age = predict_age(face)
        # 0 meat cv2.FONT_HERSHEY_SIMPLEX font face, line tyte cv2.LINE_AA = 16
        cv2.putText(frame, f"Age: {age}", (x1, y1-10), 0, 0.8, (0, 255, 255), 2, 16) 
 
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "./dataset/huy_0.jpg"
process_image(image_path)