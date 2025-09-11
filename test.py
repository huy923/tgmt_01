import cv2
from deepface import DeepFace
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

    # Performance optimization variables
    frame_count = 0
    analysis_interval = 10  
    last_analysis = None  
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Only run expensive DeepFace analysis every N frames
        frame_count += 1
        if frame_count % analysis_interval == 0 and len(faces) > 0:
            try:
                # Analyze face for emotion, gender, and age (only when faces detected)
                analysis = DeepFace.analyze(frame, actions=['emotion', 'gender', 'age'], 
                                          enforce_detection=False, detector_backend='opencv')
                # analysis can be a dict or a list of dicts
                result = analysis[0] if isinstance(analysis, list) and len(analysis) else analysis
                
                # Extract emotion data
                if isinstance(result, dict):
                    last_analysis = {
                        'dominant': result.get('dominant_emotion', 'Unknown'),
                        'gender': result.get('dominant_gender', 'Unknown'),
                        'age': result.get('age', 0)
                    }
                else:
                    last_analysis = {'dominant': 'Unknown', 'gender': 'Unknown', 'age': 0}
            except Exception as e:
                print(f"DeepFace analysis error: {e}")
                last_analysis = {'dominant': 'Unknown', 'gender': 'Unknown', 'age': 0}
        
        # Use last analysis results if available, otherwise use defaults
        if last_analysis:
            dominant = last_analysis['dominant']
            gender = last_analysis['gender']
            age = last_analysis['age']
        else:
            dominant = 'Unknown'
            gender = 'Unknown'
            age = 0
        
        # Print analysis info every 30 frames to avoid spam
        if frame_count % 30 == 0:
            print(f'Frame {frame_count}: Emotion: {dominant}, Gender: {gender}, Age: {age}')
        
        # Draw rectangles around faces and add text
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 89, 254), 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {age}', (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame,f"Emotion: {dominant}",(x, y+h+60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)
        # Write the frame to the output file
        frame = cv2.resize(frame, (frame_width, frame_height))
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