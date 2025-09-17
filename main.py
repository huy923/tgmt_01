import cv2
from deepface import DeepFace

# Load the Haar cascade for face detection and age prediction
AGE_LIST = ['(0-2)', '(3-6)', '(7-12)', '(13-17)', '(18-24)', '(25-32)', '(33-39)', '(40-45)', '(46-50)', '(51-56)', '(57-60)', '(61-65)', '(66-70)', '(71-75)', '(76-80)', '(81-85)', '(86-90)', '(91-95)', '(96-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # BGR mean values for age model

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe('./age_deploy.prototxt', './age_net.caffemodel')

def predict_age(face, net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    age_preds = net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return age

def main():
    # Open the default camera change to 0 for webcam laptop / 2 for webcam external
    cam = cv2.VideoCapture(2) # 0 for laptop webcam, 2 for external webcam
    
    # print(cv2.VideoCapture.getBackendName(cam))
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Frame dimensions: {frame_width}x{frame_height}')
    # change to 'output.avi' for video capture
    out = cv2.VideoWriter('output.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height)) 
    # cv2.namedWindow('frame',cv2.WINDOW_FULLSCREEN)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    flags = cv2.WINDOW_NORMAL
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Analyze face for emotion, gender

        analyze = DeepFace.analyze(frame, actions=['emotion', 'gender'], enforce_detection=False, detector_backend='skip')
        # analysis can be a dict or a list of dicts
        # cv2.putText(frame, f'Fps: {cam.get(cv2.CAP_PROP_FPS)}', (10, 10), fontface, 0.5, (255, 0, 0), 2) # Display FPS but just shows 30 all the time
        # Extract emotion data
        if isinstance(analyze, list) and len(analyze):
            result = analyze[0]
            dominant = result.get('dominant_emotion', "Unknown")
            gender = result.get('dominant_gender', 'Unknown')
            age = predict_age(frame, age_net) if len(faces) > 0 else "Unknown"
            # print({'dominant': dominant, 'score_percent': round(float(dominant_score), 2), 'gender': gender, 'age': age})
            # Draw rectangles around faces and add text
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 89, 254), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y+h+20), fontface, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Age: {age}', (x, y+h+40), fontface, 0.5, (255, 0, 0), 2)
                cv2.putText(frame,f"Emotion: {dominant}",(x, y+h+60),fontface,0.5,(255, 0, 0),2)
            # Write the frame to the output file
        # frame = cv2.resize(frame, (frame_width, frame_height))
        # out.write(frame)
        # Display the camera
        cv2.namedWindow('Camera', flags)
        # cv2.resizeWindow('Camera', 800, 600) # Resize window to 800x600
        cv2.imshow('Camera', frame)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'): break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()