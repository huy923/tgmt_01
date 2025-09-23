import cv2
from deepface import DeepFace

AGE_LIST = ['(0-2)', '(3-6)', '(7-12)', '(13-17)', '(18-24)', '(25-32)', '(33-39)', '(40-45)', '(46-50)', '(51-56)', '(57-60)', '(61-65)', '(66-70)', '(71-75)', '(76-80)', '(81-85)', '(86-90)', '(91-95)', '(96-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe('./age_deploy.prototxt', './age_net.caffemodel')

def predict_age(face, net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    age_preds = net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return age

def main():
    cam = cv2.VideoCapture(2)
    
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    flags = cv2.WINDOW_NORMAL
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        analyze = DeepFace.analyze(frame, actions=['emotion', 'gender'], enforce_detection=False, detector_backend='skip')
        
        if isinstance(analyze, list) and len(analyze):
            result = analyze[0]
            dominant = result.get('dominant_emotion', "Unknown")
            gender = result.get('dominant_gender', 'Unknown')
            age = predict_age(frame, age_net) if len(faces) > 0 else "Unknown"
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 89, 254), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y+h+20), fontface, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Age: {age}', (x, y+h+40), fontface, 0.5, (255, 0, 0), 2)
                cv2.putText(frame,f"Emotion: {dominant}",(x, y+h+60),fontface,0.5,(255, 0, 0),2)

        cv2.namedWindow('Camera', flags)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == ord('q'): break

    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()