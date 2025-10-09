import cv2
from deepface import DeepFace
import torch
import time

# Load the Haar cascade for face detection and age prediction
AGE_LIST = ['(0-2)', '(3-6)', '(7-12)', '(13-17)', '(18-24)', '(25-32)', '(33-39)', '(40-45)', '(46-50)', '(51-56)', '(57-60)', '(61-65)', '(66-70)', '(71-75)', '(76-80)', '(81-85)', '(86-90)', '(91-95)', '(96-100)']
# print(cv2.data.haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cv2.cuda.DeviceInfo_ComputeMode()
# cv2.cuda.setDevice(0) # Set CUDA device and 0 means default device
# Check for GPU availability
def check_gpu_availability():
    # Check OpenCV DNN GPU support
    opencv_gpu = False
    try:
        # Try to get available backends
        backends = cv2.dnn.getAvailableBackends()
        if cv2.dnn.DNN_BACKEND_CUDA in backends:
            opencv_gpu = True
            print("OpenCV DNN CUDA backend available")
    except:
        pass
    
    # Check PyTorch GPU support (for DeepFace)
    pytorch_gpu = torch.cuda.is_available()
    if pytorch_gpu:
        print(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch GPU not available, using CPU")
    
    return opencv_gpu, pytorch_gpu

# Initialize GPU/CPU configuration
opencv_gpu_available, pytorch_gpu_available = check_gpu_availability()

# Load age prediction network
age_net = cv2.dnn.readNetFromCaffe('./model/age_deploy.prototxt', './model/age_net.caffemodel')

# Configure network backend and target
if opencv_gpu_available:
    age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Age prediction network configured for GPU")
else:
    age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Age prediction network configured for CPU")
def predict_age(face):
    # Load the image as a blob 227x227 and resize to 200x200 and 78.4263377603, 87.7689143744, 114.895847746 is values for age prediction
    ageNet = age_net
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746) , swapRB=False) 
    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return age

def main():
    # Open the default camera change to 0 for webcam laptop / 2 for webcam external
    cam = cv2.VideoCapture(0) # 0 for laptop webcam, 2 for external webcam
    
    # Set camera properties for better performance
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    # print(cv2.VideoCapture.getBackendName(cam))
    # cv2.namedWindow('frame',cv2.WINDOW_FULLSCREEN)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    
    # Performance optimization variables
    frame_count = 0
    analysis_skip_frames = 5  # Analyze every 5th frame instead of every frame
    last_analysis_time = 0
    analysis_interval = 0.5  # Analyze every 0.5 seconds
    cached_results = None
    
    # FPS calculation
    # fps_start_time = time.time()
    # fps_frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        frame_count += 1
        # fps_frame_count += 1
        current_time = time.time()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimized face detection with better parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Only run DeepFace analysis periodically to improve performance
        should_analyze = (frame_count % analysis_skip_frames == 0 and 
                        current_time - last_analysis_time > analysis_interval)
        
        if should_analyze and len(faces) > 0:
            try:
                # Analyze face for emotion, gender 
                # DeepFace automatically uses GPU if available for PyTorch models
                analyze = DeepFace.analyze(frame, actions=['emotion', 'gender'], 
                                        enforce_detection=False, detector_backend='opencv')
                last_analysis_time = current_time
                
                # Cache results
                if isinstance(analyze, list) and len(analyze):
                    cached_results = analyze[0]
                else:
                    cached_results = None
            except Exception as e:
                print(f"DeepFace analysis error: {e}")
                cached_results = None
        
        # Use cached results for display
        if cached_results:
            dominant = cached_results.get('dominant_emotion', "Unknown")
            gender = cached_results.get('dominant_gender', 'Unknown')
        else:
            dominant = "Unknown"
            gender = "Unknown"
        
        # Only predict age if faces are detected and we have cached results
        age = "Unknown"
        if len(faces) > 0 and cached_results:
            try:
                age = predict_age(frame)
            except Exception as e:
                print(f"Age prediction error: {e}")
        
        # Draw rectangles around faces and add text
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 89, 254), 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y+h+20), fontface, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {age}', (x, y+h+40), fontface, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Emotion: {dominant}", (x, y+h+60), fontface, 0.5, (255, 0, 0), 2)
        
        # Calculate and display FPS
        # if fps_frame_count % 30 == 0:  # Update FPS every 30 frames
        #     fps = 30 / (current_time - fps_start_time)
        #     fps_start_time = current_time
        #     cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), fontface, 0.7, (0, 255, 0), 2)

        # Display the camera
        flags = cv2.WINDOW_NORMAL
        cv2.namedWindow('Camera', flags)
        # cv2.resizeWindow('Camera', 800, 600) # Resize window to 800x600
        cv2.imshow('Camera', frame)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'): break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()