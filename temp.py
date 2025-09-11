
import cv2
from deepface import DeepFace
# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def main():
    # Open the default camera change to 0 for webcam laptop / 2 for webcam external
    cam = cv2.VideoCapture(2)
    
    # Get frame size from the capture to match VideoWriter
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))
    
    # Prepare a larger, resizable preview window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # Set an initial size (no scaling down)
    # Performance controls
    frame_count = 0
    analysis_interval = 10  # analyze every N frames
    last_analysis = {"dominant_emotion": "Unknown", "dominant_gender": "Unknown", "age": 0}

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Run DeepFace less frequently and on a cropped face to speed up
        frame_count += 1
        if len(faces) > 0 and frame_count % analysis_interval == 0:
            x, y, w, h = faces[0]
            face_roi = frame[max(y,0):y+h, max(x,0):x+w]
            if face_roi.size != 0:
                # Resize face to a smaller size; use detector_backend='skip' since it's already a face
                face_resized = cv2.resize(face_roi, (224, 224))
                try:
                    analysis = DeepFace.analyze(
                        face_resized,
                        actions=['age','gender','emotion'],
                        enforce_detection=False,
                        detector_backend='skip',
                        prog_bar=False
                    )
                    result = analysis[0] if isinstance(analysis, list) and len(analysis) else analysis
                    if isinstance(result, dict):
                        last_analysis = {
                            'dominant_emotion': result.get('dominant_emotion', 'Unknown'),
                            'dominant_gender': result.get('dominant_gender', 'Unknown'),
                            'age': result.get('age', 0)
                        }
                except Exception as e:
                    # Keep last_analysis on error
                    pass

        dominant = last_analysis.get('dominant_emotion', 'Unknown')
        gender = last_analysis.get('dominant_gender', 'Unknown')
        age = last_analysis.get('age', 0)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(dominant), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, str(gender), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, str(age), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        out.write(frame)
        # Show full-size frame (window scales it if needed)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
            