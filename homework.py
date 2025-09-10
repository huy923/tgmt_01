import cv2

def main():
    cam = cv2.VideoCapture(2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.VideoWriter('test.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            print(f"Saved test.mp4")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()