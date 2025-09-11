import cv2
from deepface import DeepFace
# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

def main():
    # Open the default camera change to 0 for webcam laptop / 2 for webcam external
    cam = cv2.VideoCapture(2)
    
    # print(cv2.VideoCapture.getBackendName(cam))
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # change to 'output.avi' for video capture
    out = cv2.VideoWriter('output.avi',  cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height)) 
    # cv2.namedWindow('frame',cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Analyze face for emotion, gender, and age
        analysis = DeepFace.analyze(frame, actions=['emotion', 'gender', 'age'], enforce_detection=False, detector_backend='skip')
        # analysis can be a dict or a list of dicts
        result = analysis[0] if isinstance(analysis, list) and len(analysis) else analysis
        
        # Extract emotion data
        if isinstance(result,dict):
            dominant = result.get('dominant_emotion') 
            gender = result.get('dominant_gender', 'Unknown')
            age = result.get('age', 0)       
            # print({'dominant': dominant, 'score_percent': round(float(dominant_score), 2), 'gender': gender, 'age': age})
            # Draw rectangles around faces and add text
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 89, 254), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Age: {age}', (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame,f"Emotion: {dominant}",(x, y+h+60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)
            # Write the frame to the output file
        # frame = cv2.resize(frame, (frame_width, frame_height))
        # out.write(frame)

        # Display the camera
        cv2.imshow('Camera', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'): break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()