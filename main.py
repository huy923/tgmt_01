import cv2
from deepface import DeepFace
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def main():
    # Open the default camera change to 0 for webcam laptop / 2 for webcam external
    cam = cv2.VideoCapture(2)
    
    # print(cv2.VideoCapture.getBackendName(cam))
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # change to 'output.avi' for video capture
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height)) 

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        emoji = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='skip')
        # emoji can be a dict or a list of dicts
        result = emoji[0] if isinstance(emoji, list) and len(emoji) else emoji
        emotions = result.get('emotion', {}) if isinstance(result, dict) else {}
        dominant = result.get('dominant_emotion') if isinstance(result, dict) else None
        dominant_score = emotions.get(dominant) if dominant else None
        # Print concise info to terminal
        print({'dominant': dominant, 'score_percent': round(float(dominant_score), 2)})
        # Draw rectangles around faces and add text
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 89, 254), 2)
            cv2.putText(frame, f'Human {i+1}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Age: ', (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame,f"Emotion: {dominant}",(x, y+h+60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)
        # Write the frame to the output file
        frame = cv2.resize(frame, (frame_width, frame_height))
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()