import cv2
import numpy as np
import os
import glob


from tensorflow._api import v2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def capture_images(User):
    # Create dataset folder if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # Capture images
    count = 0
    cap = cv2.VideoCapture(2)
    # Capture images
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Draw rectangles around faces and save images
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

            cv2.imwrite(f'dataset/{User}.{count}.jpg', gray[y:y+h, x:x+w])
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break if count is greater than 100 this one will take 100 images from the user
        if count >= 100:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model(label):
    files = glob.glob('dataset/*.jpg')
    faces = []
    labels_list = []
    for f in files:
        img_gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        name = os.path.basename(f).split('.')[0]
        
        if name not in label:
            continue

        detected_faces = face_cascade.detectMultiScale(
            img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )


        if len(detected_faces) > 0:
            x, y, w, h = detected_faces[0]
            face_crop = img_gray[y:y + h, x:x + w]
        else:
            face_crop = img_gray

        face_crop = cv2.resize(face_crop, (200, 200))
        faces.append(face_crop)
        labels_list.append(label[name])

    if not faces:
        raise ValueError("No faces found for training. Ensure dataset images contain detectable faces and labels map is correct.")
    print(faces)
    print(labels_list)
    # Use the global recog
    # nizer defined at module level
    recognizer.train(faces, np.array(labels_list))
    recognizer.write('trainer.yml')
    return recognizer

if __name__ == '__main__':
    # capture_images("huy")
    
    label = {"huy": 0, "dung": 1}
    Recognizer = train_model(label)
    Recognizer.read('trainer.yml')
    
    # label = "dung"
    # Recognizer =train_model(label)
    # Recognizer.read('trainer.yml')

