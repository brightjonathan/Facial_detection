import cv2 as cv
import os

def webcam_face_detect():

    # load the haarcascade for face detection
    cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml');
    haar_cascade = cv.CascadeClassifier(cascade_path)

    # capture video from webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)  # Mirror the webcam
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        print(f'Number of faces detected in frame: {len(face_rects)}')

        for (x, y, w, h) in face_rects:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow('Webcam Face Detection', frame)

        # press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

